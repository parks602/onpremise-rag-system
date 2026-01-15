"""
3단계: Gradio Web UI

사용법:
    python web_ui.py <vectorstore_dir>
    
예시:
    python web_ui.py ./rag_output
"""

import sys
import os
import time
import gradio as gr
from vector_store import VectorStoreManager
from rag_qa import RAGSystem


class RAGWebUI:
    """RAG Web UI"""
    
    def __init__(self, vectorstore_dir: str, pdf_dir: str = None, model_name: str = "phi4-mini:3.8b-fp16"):
        print(f"\n{'='*60}")
        print("RAG 시스템 초기화 중...")
        print(f"{'='*60}")
        
        # PDF 디렉토리 설정 (먼저!)
        self.vectorstore_dir = vectorstore_dir
        self.pdf_dir = pdf_dir or os.path.join(
            os.path.dirname(vectorstore_dir), 
            "pdf_files"
        )
        
        # PDF 파일 검색 (먼저!)
        self.pdf_files = self._find_pdf_files()
        
        # 벡터 스토어 로드
        vectorstore_path = os.path.join(vectorstore_dir, "vectorstore")
        self.vectorstore_manager = VectorStoreManager()
        self.vectorstore_manager.load_vectorstore(vectorstore_path)
        
        # RAG 시스템 생성
        self.rag_system = RAGSystem(
            self.vectorstore_manager,
            model_name=model_name,
            pdf_files=self.pdf_files  # PDF 파일 정보 전달
        )
        
        print(f"✅ RAG 시스템 준비 완료!")
        print(f"📂 PDF 디렉토리: {self.pdf_dir}")
        print(f"📄 발견된 PDF: {len(self.pdf_files)}개\n")
    
    def _find_pdf_files(self):
        """PDF 디렉토리에서 PDF 파일 찾기"""
        import glob
        
        if not os.path.exists(self.pdf_dir):
            print(f"⚠️  경고: PDF 디렉토리를 찾을 수 없습니다: {self.pdf_dir}")
            return {}
        
        # PDF 파일 검색
        pdf_pattern = os.path.join(self.pdf_dir, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        # {파일명: 전체경로} 딕셔너리 생성
        pdf_dict = {}
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            pdf_dict[filename] = pdf_path
            print(f"   - {filename}")
        
        return pdf_dict
    
    def show_pdf_page(self, source_info):
        """출처 정보를 받아 PDF 파일 경로 반환"""
        if not source_info:
            return None
        
        try:
            # source_info 형태: "문서 제목: 문서명 | 섹션 X.X | 페이지 N"
            parts = source_info.split("|")
            doc_name = parts[0].strip().replace("문서 제목: ", "")
            
            print(f"🔍 검색할 문서명: '{doc_name}'")
            
            # 방법 1: 정확한 매칭 시도
            for pdf_filename, pdf_path in self.pdf_files.items():
                if doc_name == pdf_filename.replace('.pdf', ''):
                    print(f"✅ 정확히 매칭: {pdf_filename}")
                    return pdf_path
            
            # 방법 2: 한글만 추출해서 매칭
            import re
            doc_name_korean = re.sub(r'[^가-힣]', '', doc_name)
            
            for pdf_filename, pdf_path in self.pdf_files.items():
                pdf_korean = re.sub(r'[^가-힣]', '', pdf_filename)
                
                # 한글 부분이 포함되어 있으면 매칭
                if doc_name_korean and doc_name_korean in pdf_korean:
                    print(f"✅ 한글 매칭: '{doc_name_korean}' in '{pdf_korean}'")
                    print(f"   → {pdf_filename}")
                    return pdf_path
            
            # 방법 3: 부분 단어 매칭 (최소 3글자 이상)
            doc_words = [w for w in doc_name.split() if len(w) >= 3]
            
            for pdf_filename, pdf_path in self.pdf_files.items():
                match_count = sum(1 for word in doc_words if word in pdf_filename)
                if match_count > 0:
                    print(f"✅ 부분 매칭: {match_count}개 단어 일치")
                    print(f"   → {pdf_filename}")
                    return pdf_path
            
            print(f"❌ PDF를 찾을 수 없음: '{doc_name}'")
            print(f"   사용 가능한 PDF: {list(self.pdf_files.keys())}")
            return None
            
        except Exception as e:
            print(f"⚠️  오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def ask_question(self, question: str, history):
        """질문 처리 (Gradio 채팅 인터페이스용)"""
        if not question.strip():
            return history, ""
        
        # 답변 생성 (대화 기록 포함)
        start_time = time.time()
        result = self.rag_system.ask(
            question, 
            return_sources=True,
            chat_history=history  # 대화 기록 전달
        )
        elapsed = time.time() - start_time
        
        # sources 저장 (상세 정보 표시용)
        self._last_sources = result.get('sources', [])
        
        # 답변 포맷팅 with 출처 하이라이팅
        answer = result['answer']
        
        # 출처 정보 추가 (하이라이팅 포함)
        if 'sources' in result and result['sources']:
            # 답변 끝에 인용 표시
            answer += "\n\n" + "─" * 50 + "\n**📚 참고 문서:**\n\n"
            for i, source in enumerate(result['sources'], 1):
                answer += f"**[{i}] 문서 제목: {source['document_name']}**\n"
                answer += f"   섹션 {source['section_id']}: {source['section_title']}\n"
                answer += f"   📄 페이지: {source['page_start']}-{source['page_end']}\n\n"
        
        answer += f"\n⏱️ *응답 시간: {elapsed:.2f}초*"
        
        # Gradio messages 포맷 (필수)
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        return history, ""
    
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="GRAVITY 사내 규정 검색 시스템") as interface:
            gr.Markdown("# 📚 GRAVITY 사내 규정 검색 시스템")
            
            with gr.Row():
                # 왼쪽: 채팅 영역
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        label="💬 대화",
                        height=700,
                        show_label=True,
                        container=True
                    )
                    
                    question_input = gr.Textbox(
                        label="질문 입력",
                        placeholder="예: 병가에 대해 알려줘",
                        lines=2,
                        container=True
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("전송", variant="primary", scale=3)
                        clear_btn = gr.Button("초기화", scale=1)
                    
                    gr.Examples(
                        examples=[
                            "우리 회사 휴가 관련 규정은 어디있지?",
                            "병가에 대해서 어떻게 규정되어 있는지 알려줘",
                            "정보관리책임자의 역할은 무엇인가요?",
                        ],
                        inputs=question_input,
                        label="💡 예시 질문"
                    )
                
                # 오른쪽: 문서 참고 영역
                with gr.Column(scale=1):
                    # PDF 다운로드 (작게)
                    source_selector = gr.Dropdown(
                        label="📥 PDF 다운로드",
                        choices=[],
                        interactive=True,
                        container=True
                    )
                    
                    pdf_viewer = gr.File(
                        label="PDF 파일",
                        file_count="single",
                        file_types=[".pdf"],
                        type="filepath",
                        height=150,
                        visible=True  # 항상 표시
                    )
                    
                    # 참고 문서 상세 정보 (크게)
                    source_detail = gr.HTML(
                        value="""
                        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; height: 650px; overflow-y: auto;'>
                        <p style='text-align: center; color: #999; padding-top: 200px; font-size: 14px;'>
                        질문하시면 참고 문서가 여기에 표시됩니다
                        </p>
                        </div>
                        """
                    )
            
            # 이벤트 핸들러
            def on_submit(question, history):
                """질문 제출 시"""
                new_history, _ = self.ask_question(question, history)
                
                # 출처 추출 및 상세 정보 생성
                if new_history and len(new_history) > 0:
                    last_answer = new_history[-1]['content']
                    sources = self._extract_sources_from_answer(last_answer)
                    
                    # 드롭다운 선택지
                    source_choices = [
                        f"{s['doc']} | 섹션 {s['section']} | 페이지 {s['page']}"
                        for s in sources
                    ]
                    
                    # 모든 청크 표시 (3개 전부)
                    if sources:
                        detail_html = self._generate_all_sources_detail(sources)
                        # 첫 번째 PDF 자동 로드
                        first_pdf = self.show_pdf_page(source_choices[0]) if source_choices else None
                    else:
                        detail_html = """
                        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; height: 650px;'>
                        <p style='text-align: center; color: #999;'>참고 문서가 없습니다</p>
                        </div>
                        """
                        first_pdf = None
                else:
                    source_choices = []
                    detail_html = ""
                    first_pdf = None
                
                return new_history, "", gr.update(choices=source_choices), detail_html, first_pdf
            
            submit_btn.click(
                fn=on_submit,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input, source_selector, source_detail, pdf_viewer]
            )
            
            question_input.submit(
                fn=on_submit,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input, source_selector, source_detail, pdf_viewer]
            )
            
            clear_btn.click(
                fn=lambda: ([], "", gr.update(choices=[]), None, """
                    <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; height: 650px; overflow-y: auto;'>
                    <p style='text-align: center; color: #999; padding-top: 200px; font-size: 14px;'>
                    질문하시면 참고 문서가 여기에 표시됩니다
                    </p>
                    </div>
                    """),
                outputs=[chatbot, question_input, source_selector, pdf_viewer, source_detail],
                show_progress=False
            )
            
            # PDF 다운로드만 드롭다운에 연동
            source_selector.change(
                fn=lambda x: self.show_pdf_page(x) if x else None,
                inputs=[source_selector],
                outputs=[pdf_viewer]
            )
        
        return interface
        
        return interface
    
    def _generate_all_sources_detail(self, sources: list) -> str:
        """모든 참고 문서(3개 청크)를 한번에 표시"""
        if not sources:
            return """
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; height: 650px;'>
            <p style='text-align: center; color: #999;'>참고 문서가 없습니다</p>
            </div>
            """
        
        html_parts = []
        html_parts.append("<div style='height: 650px; overflow-y: auto; padding: 10px;'>")
        
        for i, source in enumerate(sources, 1):
            doc_name = source['doc']
            section = source['section']
            page = source['page']
            
            # 실제 내용 가져오기
            content = "내용을 불러올 수 없습니다."
            if hasattr(self, '_last_sources') and self._last_sources:
                for src in self._last_sources:
                    if (src['document_name'] == doc_name and 
                        src['section_id'] == section):
                        content = src['content']
                        break
            
            # 각 청크 HTML
            chunk_html = f"""
            <div style='background-color: #fff; border: 2px solid #dee2e6; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                <div style='background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <strong style='color: #0066cc;'>📄 참고 문서 [{i}]</strong><br/>
                    <span style='font-size: 13px;'>
                        <strong>문서:</strong> {doc_name}<br/>
                        <strong>섹션:</strong> {section}<br/>
                        <strong>페이지:</strong> {page}
                    </span>
                </div>
                <div style='background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 4px solid #0066cc; font-size: 13px; line-height: 1.6; white-space: pre-wrap;'>
{content}
                </div>
            </div>
            """
            html_parts.append(chunk_html)
        
        html_parts.append("</div>")
        
        return ''.join(html_parts)
    
    def _update_source_detail(self, source_info: str, history) -> str:
        """참고 문서 선택 시 상세 정보 업데이트"""
        if not source_info:
            return """
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px; min-height: 450px;'>
            <p style='text-align: center; color: #6c757d; padding-top: 100px;'>
            참고 문서를 선택하면 상세 정보가 표시됩니다.
            </p>
            </div>
            """
        
        # source_info 파싱: "문서명 | 섹션 X.X | 페이지 N"
        try:
            if not history or len(history) == 0:
                return "대화 기록이 없습니다."
            
            last_answer = history[-1]['content']
            sources = self._extract_sources_from_answer(last_answer)
            
            # 선택된 source 찾기
            for src in sources:
                if f"{src['doc']} | 섹션 {src['section']} | 페이지 {src['page']}" == source_info:
                    return self._generate_source_detail(src)
            
            return "해당 문서 정보를 찾을 수 없습니다."
        except Exception as e:
            return f"오류: {str(e)}"
    
    def _extract_sources_from_answer(self, answer):
        """답변에서 출처 정보 추출"""
        sources = []
        if "📚 참고 문서:" not in answer:
            return sources
        
        lines = answer.split('\n')
        current_doc = None
        
        for line in lines:
            if line.startswith('**[') and ']' in line:
                # [1] 문서 제목: 문서명 형태
                try:
                    # "**[1] 문서 제목: 그라비티취업규칙년**" 형태
                    doc_name = line.split('문서 제목:')[1].replace('**', '').strip()
                    current_doc = {'doc': doc_name}
                except:
                    continue
            elif '섹션' in line and current_doc:
                # 섹션 X.X: 제목
                try:
                    section = line.split('섹션')[1].split(':')[0].strip()
                    current_doc['section'] = section
                except:
                    pass
            elif '📄 페이지:' in line and current_doc:
                # 페이지: N-M
                try:
                    page_range = line.split('페이지:')[1].strip()
                    page = page_range.split('-')[0].strip()
                    current_doc['page'] = page
                    
                    # content 추가 (저장된 sources에서)
                    if hasattr(self, '_last_sources'):
                        for src in self._last_sources:
                            if (src['document_name'] == current_doc['doc'] and 
                                src['section_id'] == current_doc['section']):
                                current_doc['content'] = src['content']
                                break
                    
                    sources.append(current_doc)
                    current_doc = None
                except:
                    pass
        
        return sources
        
    def launch(self, share=False, server_port=7860):
        """Web UI 실행"""
        interface = self.create_interface()
        
        # PDF 디렉토리를 allowed_paths에 추가
        allowed_paths = [self.pdf_dir]
        
        interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            theme=gr.themes.Soft(),
            allowed_paths=allowed_paths  # PDF 디렉토리 허용
        )


def main():
    if len(sys.argv) < 2:
        print("사용법: python web_ui.py <vectorstore_dir> [pdf_dir] [port]")
        print("예시: python web_ui.py ./output ./pdf_files 7860")
        print("      python web_ui.py ./output  (pdf_dir 기본값: ./pdf_files)")
        sys.exit(1)
    
    vectorstore_dir = sys.argv[1]
    
    # pdf_dir 파라미터 처리
    if len(sys.argv) >= 3 and not sys.argv[2].isdigit():
        pdf_dir = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 7860
    else:
        pdf_dir = None  # 기본값 사용
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 7860
    
    if not os.path.exists(vectorstore_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {vectorstore_dir}")
        sys.exit(1)
    
    try:
        # Web UI 시작
        web_ui = RAGWebUI(vectorstore_dir, pdf_dir=pdf_dir)
        web_ui.launch(share=False, server_port=port)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()