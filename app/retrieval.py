import logging
from typing import List, Optional
from .utils.vector_store import VectorStore
from .utils.exceptions import RetrievalException

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 기본 문서로 초기화
DOCUMENTS = [
    "내 이름은 김성민",
    "나는 서울에 산다",
    "나는 오늘 기분이 좋다",
]

try:
    doc_store = VectorStore(DOCUMENTS)
except Exception as e:
    logger.error(f"VectorStore 초기화 실패: {str(e)}")
    raise RetrievalException(f"VectorStore 초기화 실패: {str(e)}")

def retrieve_docs(query: str, threshold: float = 0.4) -> List[str]:
    """
    주어진 쿼리에 대해 관련 문서를 검색합니다.
    
    Args:
        query: 검색할 질문
        threshold: 벡터 유사도 임계값 (0과 1 사이)
        
    Returns:
        List[str]: 검색된 관련 문서 리스트
        
    Raises:
        RetrievalException: 검색 과정에서 오류 발생 시
    """
    
    try:
        if not isinstance(query, str):
            raise RetrievalException("query는 문자열이어야 합니다.")
        if not isinstance(threshold, float) or threshold <= 0 or threshold > 1:
            raise RetrievalException("threshold는 0과 1 사이의 실수여야 합니다.")
            
        # DocumentStore의 search 메서드 사용
        try:
            filtered_results = doc_store.search(query, threshold)
        except Exception as e:
            logger.error(f"벡터 검색 실패: {str(e)}")
            filtered_results = []

        try:
            # 키워드(문자열) 검색: 쿼리 단어가 포함된 문서 추가
            docs = doc_store.get_documents()
            keyword_results = [doc for doc in docs if query in doc]
        except Exception as e:
            logger.error(f"키워드 검색 실패: {str(e)}")
            keyword_results = []

        # 두 결과를 합치고 중복 제거
        results = list(dict.fromkeys(filtered_results + keyword_results))
        
        logger.info(f"[VectorDB Search Log] 최종 반환 문서: {results}")
        return results
        
    except Exception as e:
        logger.error(f"문서 검색 실패: {str(e)}")
        raise RetrievalException(f"문서 검색 실패: {str(e)}")
