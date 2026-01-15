"""
벡터 스토어 관리 모듈
- FAISS 인덱스 생성/저장/로드
- 임베딩 생성
"""

import os
import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """벡터 스토어 관리 클래스"""
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            embedding_model: HuggingFace 임베딩 모델명
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
    
    def create_vectorstore(self, chunks: List[Dict]) -> FAISS:
        """청크로부터 벡터 스토어 생성"""
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'id': chunk['id'],
                    'section_id': chunk['section_id'],
                    'section_title': chunk['section_title'],
                    'page_start': chunk['metadata']['page_start'],
                    'page_end': chunk['metadata']['page_end'],
                    'document_name': chunk['metadata'].get('document_name', 'Unknown')  # 문서명 추가
                }
            )
            documents.append(doc)
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore
    
    def save_vectorstore(self, save_dir: str):
        """벡터 스토어 저장"""
        if self.vectorstore is None:
            raise ValueError("저장할 벡터 스토어가 없습니다")
        
        os.makedirs(save_dir, exist_ok=True)
        self.vectorstore.save_local(save_dir)
        print(f"벡터 스토어 저장 완료: {save_dir}")
    
    def load_vectorstore(self, load_dir: str) -> FAISS:
        """벡터 스토어 로드"""
        if not os.path.exists(load_dir):
            raise ValueError(f"벡터 스토어 디렉토리가 없습니다: {load_dir}")
        
        self.vectorstore = FAISS.load_local(
            load_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"벡터 스토어 로드 완료: {load_dir}")
        return self.vectorstore
    
    def get_retriever(self, k: int = 3):
        """Retriever 반환"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 없습니다")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """직접 검색"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 없습니다")
        
        return self.vectorstore.similarity_search(query, k=k)


def save_chunks_metadata(chunks: List[Dict], filepath: str):
    """청크 메타데이터를 JSON으로 저장"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"청크 메타데이터 저장 완료: {filepath}")


def load_chunks_metadata(filepath: str) -> List[Dict]:
    """청크 메타데이터 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"청크 메타데이터 로드 완료: {filepath}")
    return chunks