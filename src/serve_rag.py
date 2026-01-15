"""
2단계: RAG 시스템 서버 실행
벡터 스토어와 LLM을 메모리에 상주시키고 질의응답

사용법:
    python serve_rag.py <vectorstore_dir>
    
예시:
    python serve_rag.py ./rag_output
"""

import sys
import os
from vector_store import VectorStoreManager
from rag_qa import RAGSystem


class RAGServer:
    """RAG 서버 (메모리 상주)"""
    
    def __init__(self, vectorstore_dir: str, model_name: str = "phi4-mini:3.8b-fp16"):
        """
        Args:
            vectorstore_dir: 벡터 스토어 디렉토리
            model_name: Ollama 모델명
        """
        print(f"\n{'='*60}")
        print("RAG 시스템 초기화 중...")
        print(f"{'='*60}")
        
        # 벡터 스토어 로드
        print(f"\n1. 벡터 스토어 로드 중...")
        vectorstore_path = os.path.join(vectorstore_dir, "vectorstore")
        self.vectorstore_manager = VectorStoreManager()
        self.vectorstore_manager.load_vectorstore(vectorstore_path)
        
        # RAG 시스템 생성
        print(f"\n2. LLM 로드 중 (모델: {model_name})...")
        self.rag_system = RAGSystem(
            self.vectorstore_manager,
            model_name=model_name
        )
        
        print(f"\n{'='*60}")
        print("✅ RAG 시스템 준비 완료!")
        print(f"{'='*60}")
    
    def ask(self, question: str):
        """질문하기"""
        return self.rag_system.ask_and_print(question)
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n대화형 모드 시작 (종료: 'quit' 또는 'exit')\n")
        
        while True:
            try:
                question = input("\n질문 > ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("종료합니다.")
                    break
                
                self.ask(question)
                
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python serve_rag.py <vectorstore_dir>")
        print("예시: python serve_rag.py ./rag_output")
        sys.exit(1)
    
    vectorstore_dir = sys.argv[1]
    
    if not os.path.exists(vectorstore_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {vectorstore_dir}")
        sys.exit(1)
    
    try:
        # RAG 서버 시작
        server = RAGServer(vectorstore_dir)
        
        # 대화형 모드 실행
        server.interactive_mode()
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
