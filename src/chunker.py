"""
텍스트 청킹 및 임베딩 모듈
"""

import re
from typing import List, Dict


class TextChunker:
    """텍스트 청킹 클래스"""
    
    def __init__(self, max_length: int = 1024, overlap: int = 150):
        self.max_length = max_length
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_length
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks
    
    def create_rag_chunks(self, sections: List[Dict]) -> List[Dict]:
        """섹션 리스트를 RAG 청크로 변환"""
        rag_chunks = []
        
        for sec in sections:
            chunks = self.chunk_text(sec["text"])
            for i, chunk_text in enumerate(chunks):
                rag_chunks.append({
                    "id": f'{sec["section_id"]}_{i}',
                    "section_id": sec["section_id"],
                    "section_title": sec["section_title"],
                    "text": chunk_text,
                    "metadata": {
                        "page_start": sec["start_page"] + 1,
                        "page_end": sec["end_page"] + 1,
                    }
                })
        
        return rag_chunks
    
    def enrich_with_metadata(self, chunks: List[Dict], document_name: str) -> List[Dict]:
        """청크에 메타데이터 추가"""
        enriched_chunks = []
        
        for chunk in chunks:
            enriched_text = f"""문서: {document_name}
섹션: {chunk['section_id']} - {chunk['section_title']}

{chunk['text']}"""
            
            enriched_chunks.append({
                **chunk,
                "text": enriched_text,
                "metadata": {
                    **chunk['metadata'],
                    "document_name": document_name  # 문서명 추가
                }
            })
        
        return enriched_chunks
    
    def check_duplicates(self, chunks: List[Dict]) -> Dict:
        """중복 체크"""
        title_to_ids = {}
        for chunk in chunks:
            title = chunk['section_title']
            if title not in title_to_ids:
                title_to_ids[title] = []
            title_to_ids[title].append(chunk['section_id'])
        
        duplicates = {}
        for title, ids in title_to_ids.items():
            unique_ids = list(set(ids))
            if len(unique_ids) > 1:
                duplicates[title] = unique_ids
        
        return {
            'duplicates': duplicates,
            'has_duplicates': bool(duplicates),
            'total_chunks': len(chunks),
            'unique_titles': len(title_to_ids)
        }


def extract_korean_from_filename(filename: str) -> str:
    """파일명에서 한글만 추출"""
    return re.sub(r'[^가-힣]', '', filename)