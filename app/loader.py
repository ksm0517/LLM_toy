import logging
from typing import List, Dict, Union, Optional
from .utils.vector_store import VectorStore
from .utils.exceptions import RetrievalException
from pathlib import Path
import pymupdf4llm # type: ignore
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter, ExperimentalMarkdownSyntaxTextSplitter
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """문서 로더 클래스: PDF 및 텍스트 문서를 로드하고 벡터화하여 검색 준비"""
    
    def __init__(self, persistence_path: str, documents: Optional[List[str]] = None):
        """
        DocumentLoader 객체를 초기화합니다.
        
        Args:
            persistence_path: VectorStore의 영속성 경로
            documents: 초기 문서 문자열 리스트 (선택사항)
            
        Raises:
            RetrievalException: 벡터 스토어 초기화 실패 시
        """
        try:
            self.doc_store = VectorStore(persistence_path=persistence_path, documents=documents or [])
            logger.info("DocumentLoader 초기화 완료")
        except Exception as e:
            logger.error(f"VectorStore 초기화 실패: {str(e)}")
            raise RetrievalException(f"VectorStore 초기화 실패: {str(e)}")
        

    def chunk_documents(self, docs: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        마크다운 문서 리스트를 청크 단위로 분할

        Args:
            docs: 분할할 문서 리스트
            chunk_size: 청크 최대 길이
            chunk_overlap: 청크 간 중첩 길이

        Returns:
            List[str]: 청크 리스트
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        
        # MarkdownHeaderTextSplitter는 Document 객체의 리스트를 반환합니다. 각 Document는 page_content와 헤더를 포함한 metadata를 가집니다.
        splitter = ExperimentalMarkdownSyntaxTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        # splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        # splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        processed_chunks: list[str] = []
        splitted_docs=[]
        for doc in docs:
            splitted_docs.extend(splitter.split_text(doc))
        if type(splitted_docs[0]) == Document:
            for splitted_doc in splitted_docs:
                # page_content에 이미 헤더 정보가 포함되어 있습니다 (strip_headers=False).
                # 만약 strip_headers=True로 설정했다면, metadata를 조합해야 합니다.
                content = splitted_doc.page_content
                if content.strip():
                    processed_chunks.append(content.strip())
        elif type(splitted_docs[0]) == str:
            processed_chunks = [chunk.strip() for chunk in splitted_docs if chunk.strip()]
            
        return processed_chunks


    def load_pdf(self, pdf_path: Union[str, Path], chunk_size: int = 800, chunk_overlap = 100) -> List[int]:
        """
        PDF 파일을 로드하고 청크 단위로 분할하여 벡터 스토어에 추가합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 각 청크의 최대 문자 수 (기본값: 1000)
            
        Returns:
            List[int]: 추가된 문서 청크들의 ID 리스트
            
        Raises:
            RetrievalException: PDF 로딩 실패 시
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise RetrievalException(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
                
            doc_ids = []
            
            llama_reader = pymupdf4llm.LlamaMarkdownReader()
            llama_docs = llama_reader.load_data(pdf_path)
            
            # 텍스트 추출
            doc_list = [doc.text for doc in llama_docs if doc.text.strip()]
            
            # 청크 분할
            chunks = self.chunk_documents(doc_list, chunk_size, chunk_overlap)

            # 벡터 스토어에 배치로 추가
            doc_ids = self.doc_store.add_documents(chunks)

            logger.info(f"PDF 로드 완료: {pdf_path.name} ({len(doc_ids)}개 청크 생성)")
            return doc_ids
            
        except Exception as e:
            logger.error(f"PDF 로드 실패 ({pdf_path}): {str(e)}")
            raise RetrievalException(f"PDF 로드 실패: {str(e)}")


    def load_pdfs_from_directory(self, 
                               directory: Union[str, Path], 
                               chunk_size: int = 1000) -> Dict[str, List[int]]:
        """
        지정된 디렉토리의 모든 PDF 파일을 로드합니다.
        
        Args:
            directory: PDF 파일들이 있는 디렉토리 경로
            chunk_size: 각 청크의 최대 문자 수 (기본값: 1000)
            
        Returns:
            Dict[str, List[int]]: {파일명: 문서 ID 리스트} 형태의 딕셔너리
            
        Raises:
            RetrievalException: 디렉토리 처리 실패 시
        """
        try:
            directory = Path(directory)
            if not directory.exists():
                raise RetrievalException(f"디렉토리를 찾을 수 없습니다: {directory}")
                
            results = {}
            
            # 모든 PDF 파일 처리
            for pdf_file in directory.glob("*.pdf"):
                try:
                    doc_ids = self.load_pdf(pdf_file, chunk_size)
                    results[pdf_file.name] = doc_ids
                except Exception as e:
                    logger.error(f"PDF 파일 처리 실패 ({pdf_file.name}): {str(e)}")
                    continue
                    
            logger.info(f"디렉토리 처리 완료: {len(results)}개 PDF 파일 로드")
            return results
            
        except Exception as e:
            logger.error(f"디렉토리 처리 실패 ({directory}): {str(e)}")
            raise RetrievalException(f"디렉토리 처리 실패: {str(e)}")


    def search_documents(self, 
                        query: str, 
                        threshold: float = 0.4,
                        top_k: int = 5) -> List[Dict[str, any]]:
        """
        쿼리와 관련된 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            threshold: 유사도 임계값 (기본값: 0.4)
            top_k: 반환할 최대 문서 수 (기본값: 5)
            
        Returns:
            List[Dict]: 검색 결과 리스트. 
                       각 결과는 {'doc_id': id, 'content': text, 'similarity': score} 형태
                       
        Raises:
            RetrievalException: 검색 실패 시
        """
        try:
            if not isinstance(query, str) or not query.strip():
                raise RetrievalException("유효한 검색어가 필요합니다.")
            results = self.doc_store.search(query, threshold, top_k)
            logger.info(f"검색 완료: {len(results)}개 결과 (쿼리: {query})")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            raise RetrievalException(f"문서 검색 실패: {str(e)}")
    
    def add_text_document(self, text: str) -> int:
        """
        텍스트 문서를 추가합니다.
        
        Args:
            text: 추가할 텍스트 문서
            
        Returns:
            int: 추가된 문서의 ID
            
        Raises:
            RetrievalException: 문서 추가 실패 시
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise RetrievalException("유효한 텍스트가 필요합니다.")
                
            doc_id = self.doc_store.add_document(text.strip())
            logger.info(f"텍스트 문서 추가 완료 (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"텍스트 문서 추가 실패: {str(e)}")
            raise RetrievalException(f"텍스트 문서 추가 실패: {str(e)}")