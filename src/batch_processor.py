"""
배치 처리 모듈
- 폴더 내 모든 PDF 파일 처리
- 벡터 스토어 통합 생성
"""

import os
import glob
from typing import List, Dict
from pdf_processor import PDFProcessor
from chunker import TextChunker, extract_korean_from_filename
from vector_store import VectorStoreManager, save_chunks_metadata


class BatchPDFProcessor:
    """배치 PDF 처리 클래스"""
    
    def __init__(
        self,
        pdf_dir: str,
        output_dir: str = "./rag_output",
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ):
        """
        Args:
            pdf_dir: PDF 파일들이 있는 디렉토리
            output_dir: 출력 디렉토리
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 컴포넌트 초기화
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker(max_length=chunk_size, overlap=chunk_overlap)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def get_pdf_files(self) -> List[str]:
        """PDF 파일 목록 가져오기"""
        pattern = os.path.join(self.pdf_dir, "*.pdf")
        pdf_files = glob.glob(pattern)
        return sorted(pdf_files)
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """단일 PDF 처리"""
        filename = os.path.basename(pdf_path)
        print(f"\n처리 중: {filename}")
        
        try:
            # 1. PDF 처리 (목차 추출, 텍스트 추출)
            sections = self.pdf_processor.process_pdf(pdf_path)
            print(f"  ✓ {len(sections)}개 섹션 추출")
            
            # 2. 청킹
            chunks = self.chunker.create_rag_chunks(sections)
            print(f"  ✓ {len(chunks)}개 청크 생성")
            
            # 3. 메타데이터 추가
            document_name = extract_korean_from_filename(filename)
            enriched_chunks = self.chunker.enrich_with_metadata(chunks, document_name)
            
            # 4. 중복 체크
            dup_result = self.chunker.check_duplicates(enriched_chunks)
            if dup_result['has_duplicates']:
                print(f"  ⚠️ 중복 발견: {len(dup_result['duplicates'])}개")
            
            return {
                'status': 'success',
                'filename': filename,
                'document_name': document_name,
                'chunks': enriched_chunks,
                'num_sections': len(sections),
                'num_chunks': len(enriched_chunks)
            }
            
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return {
                'status': 'failed',
                'filename': filename,
                'error': str(e)
            }
    
    def process_all(self) -> Dict:
        """모든 PDF 처리"""
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            print(f"PDF 파일이 없습니다: {self.pdf_dir}")
            return {
                'status': 'failed',
                'message': 'No PDF files found'
            }
        
        print(f"\n{'='*60}")
        print(f"배치 처리 시작: {len(pdf_files)}개 PDF 파일")
        print(f"{'='*60}")
        
        all_chunks = []
        results = []
        
        for pdf_path in pdf_files:
            result = self.process_single_pdf(pdf_path)
            results.append(result)
            
            if result['status'] == 'success':
                all_chunks.extend(result['chunks'])
        
        # 통계
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = len(results) - success_count
        
        print(f"\n{'='*60}")
        print(f"배치 처리 완료")
        print(f"{'='*60}")
        print(f"총 파일: {len(pdf_files)}")
        print(f"성공: {success_count}")
        print(f"실패: {failed_count}")
        print(f"총 청크: {len(all_chunks)}")
        
        return {
            'status': 'success',
            'total_files': len(pdf_files),
            'success_count': success_count,
            'failed_count': failed_count,
            'total_chunks': len(all_chunks),
            'chunks': all_chunks,
            'results': results
        }
    
    def process_and_save(self) -> str:
        """모든 PDF 처리 후 벡터 스토어 저장"""
        # 1. 배치 처리
        batch_result = self.process_all()
        
        if batch_result['total_chunks'] == 0:
            raise ValueError("처리된 청크가 없습니다")
        
        # 2. 청크 메타데이터 저장
        chunks_file = os.path.join(self.output_dir, "chunks.json")
        save_chunks_metadata(batch_result['chunks'], chunks_file)
        
        # 3. 벡터 스토어 생성 및 저장
        print("\n벡터 스토어 생성 중...")
        vectorstore_manager = VectorStoreManager()
        vectorstore_manager.create_vectorstore(batch_result['chunks'])
        
        vectorstore_dir = os.path.join(self.output_dir, "vectorstore")
        vectorstore_manager.save_vectorstore(vectorstore_dir)
        
        print(f"\n✅ 모든 처리 완료!")
        print(f"   - 청크 메타데이터: {chunks_file}")
        print(f"   - 벡터 스토어: {vectorstore_dir}")
        
        return vectorstore_dir


# 사용 예시
if __name__ == "__main__":
    # 배치 처리
    processor = BatchPDFProcessor(
        pdf_dir="./pdf_files",
        output_dir="./rag_output"
    )
    
    vectorstore_dir = processor.process_and_save()
    print(f"\n벡터 스토어 경로: {vectorstore_dir}")
