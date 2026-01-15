"""
PDF 문서 처리 모듈
- PDF에서 목차 추출
- 섹션별 텍스트 추출
- 청킹
"""

import re
import pdfplumber
from typing import List, Dict, Optional


class PDFConfig:
    """PDF 처리 설정"""
    CROP_TOP = 114
    CROP_BOTTOM = 779
    TOC_KEYWORD = '목 차'
    MAX_TOC_PAGES = 3
    
    SECTION_PATTERNS = [
        r'^(\d+(?:\.\d+)+)\.',    # 1.1. / 1.1.1.
        r'^(\d+(?:\.\d+)+)\s+',   # 1.1 / 1.1.1
        r'^(\d+\.\s+\d+)',         # 1. 1 (점 + 공백 + 숫자)
        r'^(\d+)\.',              # 1.
        r'^(\d+)\s+',             # 1
    ]
    
    SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS))
    PAGE_REGEX = re.compile(r'(\d+)\s*$')
    DOTS_REGEX = re.compile(r'\.{2,}')


class PDFProcessor:
    """PDF 문서 처리 클래스"""
    
    def __init__(self, config: PDFConfig = None):
        self.config = config or PDFConfig()
    
    def crop_page(self, page):
        """페이지 크롭"""
        return page.crop((0, self.config.CROP_TOP, page.width, self.config.CROP_BOTTOM))
    
    def find_toc_page(self, pdf) -> Optional[int]:
        """목차 페이지 찾기"""
        for page_num, page in enumerate(pdf.pages):
            cropped = self.crop_page(page)
            text = cropped.extract_text()
            if text and self.config.TOC_KEYWORD in text:
                return page_num
        return None
    
    def parse_toc_line(self, line: str) -> Optional[Dict]:
        """목차 라인 파싱"""
        line = line.strip()
        if not line:
            return None
        
        # 페이지 번호
        page_match = self.config.PAGE_REGEX.search(line)
        if not page_match:
            return None
        page = int(page_match.group(1))
        
        # 섹션 번호
        section_match = self.config.SECTION_REGEX.search(line)
        if not section_match:
            return None
        section_id = next(g for g in section_match.groups() if g is not None)
        section_id = section_id.replace(" ", "")
        
        # 제목 추출
        title_part = line[section_match.end():].strip()
        title_part = self.config.PAGE_REGEX.sub("", title_part).strip()
        title_part = self.config.DOTS_REGEX.sub(" ", title_part).strip()
        
        return {
            "section_id": section_id,
            "section_title": title_part,
            "page": page
        }
    
    def extract_toc(self, pdf, toc_page_num: int) -> List[Dict]:
        """목차 추출"""
        sections = []
        
        end_page = min(toc_page_num + self.config.MAX_TOC_PAGES, len(pdf.pages))
        for i in range(toc_page_num, end_page):
            page = self.crop_page(pdf.pages[i])
            text = page.extract_text()
            
            if not text:
                continue
            
            for line in text.split('\n'):
                result = self.parse_toc_line(line)
                if result:
                    sections.append(result)
        
        return sections
    
    def build_page_ranges(self, toc: List[Dict], total_pages: int) -> List[Dict]:
        """페이지 범위 계산"""
        sections = []
        for i, sec in enumerate(toc):
            start = sec["page"] - 1
            
            if i + 1 < len(toc):
                next_page = toc[i + 1]["page"]
                if next_page - sec["page"] >= 1:
                    end = next_page - 1
                else:
                    end = sec["page"] - 1
            else:
                end = total_pages - 1
            
            sections.append({
                **sec,
                "start_page": start,
                "end_page": end
            })
        
        return sections
    
    def extract_section_text(self, pdf_path: str, sections: List[Dict]) -> List[Dict]:
        """섹션별 텍스트 추출"""
        results = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for n, sec in enumerate(sections):
                # 페이지 텍스트 추출
                raw_texts = ''
                for p in range(sec["start_page"], sec["end_page"] + 1):
                    page = pdf.pages[p]
                    cropped = self.crop_page(page)
                    page_text = cropped.extract_text()
                    if page_text:
                        raw_texts += page_text + "\n"
                
                # 시작 위치 찾기
                pattern = re.escape(sec['section_id']) + r'\s+' + re.escape(sec['section_title'])
                m_start = re.search(pattern, raw_texts)
                start_idx = m_start.end() if m_start else 0
                
                # 끝 위치 찾기
                if n < len(sections) - 1:
                    next_sec_id = sections[n + 1]['section_id']
                    next_pattern = r'\n\s*' + re.escape(next_sec_id) + r'[\.\s]'
                    m_end = re.search(next_pattern, raw_texts[start_idx:])
                    
                    if m_end:
                        end_idx = start_idx + m_end.start()
                        sliced = raw_texts[start_idx:end_idx]
                    else:
                        sliced = raw_texts[start_idx:]
                else:
                    sliced = raw_texts[start_idx:]
                
                results.append({
                    **sec,
                    "text": sliced.strip()
                })
        
        return results
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """PDF 전체 처리 파이프라인"""
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # 1. 목차 페이지 찾기
            toc_page_num = self.find_toc_page(pdf)
            if toc_page_num is None:
                raise ValueError("목차를 찾을 수 없습니다")
            
            # 2. 목차 추출
            toc = self.extract_toc(pdf, toc_page_num)
            
            # 3. 페이지 범위 계산
            sections = self.build_page_ranges(toc, total_pages)
        
        # 4. 섹션별 텍스트 추출
        results = self.extract_section_text(pdf_path, sections)
        
        return results
