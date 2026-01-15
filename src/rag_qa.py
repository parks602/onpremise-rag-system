"""
RAG 질의응답 시스템 모듈
- Ollama LLM 연동
- 프롬프트 템플릿
- 질의응답 체인
"""

from typing import List, Dict
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGSystem:
    """RAG 질의응답 시스템"""
    
    DEFAULT_PROMPT = """당신은 사내 규정을 이해하고 설명하는 전문가입니다. 아래 예시를 참고하여 답변하세요.

예시 1 (위치 질문):
질문: 병가 규정은 어디에 있나요?
답변: 그라비티취업규칙 섹션 3.29에 있습니다.

예시 2 (내용 질문):
질문: 병가는 어떻게 사용하나요?
답변: [섹션 3.29] 병가는 전염병 감염이나 입원이 필요한 경우에 부여됩니다. 사용 기간은 3개월 이내이며, 치유 가능한 경우 1회에 한해 3개월 추가 연장이 가능합니다. 신청 시에는 의사 진단서를 제출해야 합니다.

예시 3 (확인 질문):
질문: 병가를 6개월 쓸 수 있나요?
답변: 네, 가능합니다. [섹션 3.29]에 따르면 최초 3개월 사용 후, 치유 가능한 경우 1회에 한해 3개월 추가 연장이 가능하므로 최대 6개월까지 사용할 수 있습니다.

───────────────────────────

이제 다음 질문에 답변하세요. 위 예시처럼 자연스럽고 2문장 이상으로 답변하되, 문서 원문을 그대로 복사하지 마세요.

문서 내용:
{context}

질문: {question}

답변:"""
    
    def __init__(
        self,
        vectorstore_manager,
        model_name: str = "phi4-mini:3.8b-fp16",
        temperature: float = 0.1,
        prompt_template: str = None,
        pdf_files: dict = None  # PDF 파일 딕셔너리 추가
    ):
        """
        Args:
            vectorstore_manager: VectorStoreManager 인스턴스
            model_name: Ollama 모델명
            temperature: LLM 온도
            prompt_template: 커스텀 프롬프트 (선택)
            pdf_files: {filename: filepath} 딕셔너리 (선택)
        """
        self.vectorstore_manager = vectorstore_manager
        self.llm = Ollama(model=model_name, temperature=temperature)
        self.pdf_files = pdf_files or {}  # PDF 파일 정보 저장
        
        # 프롬프트 설정
        template = prompt_template or self.DEFAULT_PROMPT
        self.prompt = PromptTemplate.from_template(template)
        
        # RAG 체인 구성
        self.retriever = vectorstore_manager.get_retriever(k=3)
        self.rag_chain = self._build_chain()
    
    def _build_chain(self):
        """RAG 체인 생성"""
        def format_docs(docs):
            # LLM에게는 display_text (메타데이터 포함) 전달
            formatted = []
            for doc in docs:
                display_text = doc.metadata.get('display_text', doc.page_content)
                formatted.append(display_text)
            return "\n\n".join(formatted)
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question: str, return_sources: bool = True, chat_history: List = None) -> Dict:
        """
        질문하고 답변 받기
        
        Args:
            question: 질문
            return_sources: 출처 문서 반환 여부
            chat_history: 이전 대화 기록 (선택)
            
        Returns:
            {
                'answer': 답변 텍스트,
                'sources': 출처 문서 리스트 (선택)
            }
        """
        # 질문 유형 판단
        question_type = self._classify_question(question)
        
        # Query Expansion: 짧은 질문을 확장
        expanded_query = self._expand_query(question)
        
        # 대화 기록이 있으면 컨텍스트에 추가
        if chat_history:
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            
            history_context = "\n\n이전 대화:\n"
            for i in range(0, len(recent_history), 2):
                if i+1 < len(recent_history):
                    user_msg = recent_history[i]['content']
                    assistant_msg = recent_history[i+1]['content']
                    if '─' in assistant_msg:
                        assistant_msg = assistant_msg.split('─')[0].strip()
                    history_context += f"Q: {user_msg}\nA: {assistant_msg}\n\n"
            
            enhanced_question = f"{history_context}현재 질문: {question}"
        else:
            enhanced_question = question
        
        # 검색 먼저 수행
        source_docs = self.retriever.invoke(expanded_query)
        
        if not source_docs:
            return {
                'answer': "관련 문서를 찾을 수 없습니다.",
                'sources': []
            }
        
        # 위치 질문이면 간단히 답변
        if question_type == "location":
            answer = self._generate_location_answer(question, source_docs)
        else:
            # 일반 질문은 LLM으로 답변 생성
            answer = self.rag_chain.invoke(enhanced_question)
            
            # 🔥 검증: LLM 답변이 검색 결과와 일치하는지 확인
            answer = self._verify_and_fix_answer(answer, source_docs, question)
        
        result = {'answer': answer}
        
        # 출처 문서 추가
        if return_sources:
            result['sources'] = [
                {
                    'document_name': doc.metadata.get('document_name', 'Unknown'),
                    'section_id': doc.metadata['section_id'],
                    'section_title': doc.metadata['section_title'],
                    'page_start': doc.metadata['page_start'],
                    'page_end': doc.metadata['page_end'],
                    'content': doc.page_content
                }
                for doc in source_docs
            ]
        
        return result
    
    def _verify_and_fix_answer(self, answer: str, docs: List, question: str) -> str:
        """답변이 검색 결과와 일치하는지 검증하고 수정 (심각한 경우만)"""
        if not docs:
            return answer
        
        # 검색된 섹션 정보
        doc = docs[0]
        doc_name = doc.metadata.get('document_name', '알 수 없는 문서')
        section_id = doc.metadata.get('section_id', '')
        section_title = doc.metadata.get('section_title', '')
        display_text = doc.metadata.get('display_text', doc.page_content)
        
        # 패턴 1: "없습니다", "명시되어 있지 않습니다" 등 부정 답변 (심각!)
        negative_patterns = ["없습니다", "명시되어 있지 않", "찾을 수 없", "규정되어 있지 않"]
        is_negative = any(pattern in answer for pattern in negative_patterns)
        
        # 패턴 2: "규정되어 있습니다"만 있고 실제 내용이 거의 없음 (심각!)
        has_no_content = ("규정되어 있습니다" in answer or "명시되어 있습니다" in answer) and len(answer.strip()) < 50
        
        # 심각한 오류만 수정 (부정 답변 또는 내용 없음)
        needs_fix = is_negative or has_no_content
        
        if needs_fix:
            print(f"🔧 답변 검증 실패 - 강제 수정")
            print(f"   - 부정 답변: {is_negative}")
            print(f"   - 내용 없음: {has_no_content}")
            
            # 실제 내용 추출 (메타데이터 제거)
            content = display_text
            lines = content.split('\n')
            actual_content = []
            skip_metadata = True
            
            for line in lines:
                if skip_metadata:
                    if line.strip() and not line.startswith('문서:') and not line.startswith('섹션:'):
                        skip_metadata = False
                        actual_content.append(line)
                else:
                    actual_content.append(line)
            
            content_text = '\n'.join(actual_content).strip()
            
            # 내용이 너무 길면 요약
            if len(content_text) > 300:
                content_preview = content_text[:300] + "..."
            else:
                content_preview = content_text
            
            # 강제 답변 생성
            answer = f"[섹션 {section_id}] {section_title}에 다음과 같이 규정되어 있습니다:\n\n{content_preview}"
        
        return answer
    
    def _classify_question(self, question: str) -> str:
        """질문 유형 분류"""
        # 위치 질문 키워드
        location_keywords = ["어디", "어느", "어떤 문서", "어떤 규정", "찾아", "어디있", "어디에"]
        
        # 내용 질문 키워드 (명시적)
        content_keywords = ["알려줘", "알려주", "설명", "내용", "어떻게", "무엇", "뭐", "무슨"]
        
        # 내용 질문이면 content
        for keyword in content_keywords:
            if keyword in question:
                return "content"
        
        # 위치 질문이면 location
        for keyword in location_keywords:
            if keyword in question:
                return "location"
        
        # 애매하면 content (기본)
        return "content"
    
    def _generate_location_answer(self, question: str, docs: List) -> str:
        """위치 질문에 대한 간단한 답변 생성"""
        if not docs:
            return "문서에서 관련 내용을 찾을 수 없습니다."
        
        # 가장 관련성 높은 문서 정보 추출
        doc = docs[0]
        doc_name = doc.metadata.get('document_name', '알 수 없는 문서')
        
        # 실제 PDF 파일명 찾기
        pdf_filename = self._find_pdf_filename(doc_name)
        
        # 간단한 답변 생성
        answer = f"사규문서 '{pdf_filename}'"
        answer += "에 있습니다."
        
        return answer
    
    def _find_pdf_filename(self, doc_name: str) -> str:
        """문서명으로 실제 PDF 파일명 찾기"""
        import re
        
        if not self.pdf_files:
            return doc_name
        
        # 방법 1: 정확한 매칭
        for pdf_filename in self.pdf_files.keys():
            if doc_name == pdf_filename.replace('.pdf', ''):
                return pdf_filename
        
        # 방법 2: 한글만 추출해서 매칭
        doc_name_korean = re.sub(r'[^가-힣]', '', doc_name)
        
        for pdf_filename in self.pdf_files.keys():
            pdf_korean = re.sub(r'[^가-힣]', '', pdf_filename)
            
            if doc_name_korean and doc_name_korean in pdf_korean:
                return pdf_filename
        
        # 방법 3: 부분 매칭
        for pdf_filename in self.pdf_files.keys():
            if doc_name in pdf_filename:
                return pdf_filename
        
        # 찾지 못하면 원본 반환
        return doc_name
    
    def _expand_query(self, question: str) -> str:
        """질문을 확장하여 검색 정확도 향상"""
        # 짧은 질문을 더 자세하게
        expansions = {
            "역할": "역할과 직무와 업무와 책임",
            "방법": "방법과 절차와 과정",
            "규정": "규정과 규칙과 정책",
            "누구": "담당자와 책임자",
            "어디": "위치와 장소",
            "언제": "시기와 기간",
        }
        
        expanded = question
        for keyword, expansion in expansions.items():
            if keyword in question:
                expanded = expanded.replace(keyword, expansion)
        
        return expanded
    
    def ask_and_print(self, question: str):
        """질문하고 결과 출력"""
        result = self.ask(question, return_sources=True)
        
        print(f"\n질문: {question}")
        print("=" * 60)
        print(f"\n답변:\n{result['answer']}")
        
        if 'sources' in result:
            print("\n" + "=" * 60)
            print("참고한 문서:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n[{i}] 섹션 {source['section_id']}: {source['section_title']}")
                print(f"    페이지: {source['page_start']}-{source['page_end']}")
                print(f"    내용: {source['content'][:200]}...")


class RAGSystemFactory:
    """RAG 시스템 팩토리"""
    
    @staticmethod
    def create_from_vectorstore(
        vectorstore_manager,
        model_name: str = "phi4-mini:3.8b-fp16",
        pdf_files: dict = None
    ) -> RAGSystem:
        """기존 벡터 스토어로부터 RAG 시스템 생성"""
        return RAGSystem(vectorstore_manager, model_name=model_name, pdf_files=pdf_files)
    
    @staticmethod
    def create_from_chunks(
        chunks: List[Dict],
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        llm_model: str = "phi4-mini:3.8b-fp16",
        pdf_files: dict = None
    ) -> RAGSystem:
        """청크로부터 RAG 시스템 생성"""
        from vector_store import VectorStoreManager
        
        # 벡터 스토어 생성
        vectorstore_manager = VectorStoreManager(embedding_model)
        vectorstore_manager.create_vectorstore(chunks)
        
        # RAG 시스템 생성
        return RAGSystem(vectorstore_manager, model_name=llm_model, pdf_files=pdf_files)