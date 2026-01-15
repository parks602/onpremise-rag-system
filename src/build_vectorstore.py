"""
1단계: PDF 폴더를 벡터 스토어로 변환

사용법:
    python build_vectorstore.py <pdf_dir> [output_dir]
    
예시:
    python build_vectorstore.py ./pdf_files ./rag_output
"""

import sys
import os
from batch_processor import BatchPDFProcessor


def main():
    if len(sys.argv) < 2:
        print("사용법: python build_vectorstore.py <pdf_dir> [output_dir]")
        print("예시: python build_vectorstore.py ./pdf_files ./rag_output")
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./rag_output"
    
    if not os.path.exists(pdf_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {pdf_dir}")
        sys.exit(1)
    
    print(f"PDF 디렉토리: {pdf_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 배치 처리 실행
    processor = BatchPDFProcessor(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        chunk_size=1024,
        chunk_overlap=150
    )
    
    try:
        vectorstore_dir = processor.process_and_save()
        print(f"\n{'='*60}")
        print(f"✅ 성공!")
        print(f"{'='*60}")
        print(f"벡터 스토어가 생성되었습니다: {vectorstore_dir}")
        print(f"\n다음 단계: python serve_rag.py {output_dir}")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
