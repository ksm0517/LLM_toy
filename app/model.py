from .retrieval import retrieve_docs
from .utils.exceptions import ModelException
from .models.groq_llm import GroqLLM
from .models.local_llm import LocalLLM

import logging
from typing import Optional, List, Dict, Any, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    """
    RAG 파이프라인에서 LLM을 활용해 답변을 생성하는 모델 클래스입니다.
    - api_key: LLM API 키 (Groq 사용 시)
    - llm_type: 'groq' 또는 'local' (로컬 LLM 사용 시)
    """
    
    def __init__(self, api_key: Optional[str] = None, llm_type: str = "local"):
        """
        Model 객체를 초기화합니다.
        - api_key: Groq API 키
        - llm_type: 'groq' 또는 'local'
        
        Raises:
            ModelException: LLM 초기화 실패 시
            ValueError: 잘못된 llm_type 입력 시
        """
        try:
            self.llm_type = llm_type
            if llm_type == "groq":
                self.llm = GroqLLM(api_key=api_key)
            elif llm_type == "local":
                self.llm = LocalLLM()
            else:
                raise ValueError("llm_type은 'groq' 또는 'local'이어야 합니다.")
        except Exception as e:
            logger.error(f"Model 초기화 실패: {str(e)}")
            raise ModelException(f"Model 초기화 실패: {str(e)}")


    def retrieve(self, query: str, threshold: float = 0.4) -> str:
        """
        주어진 쿼리(query)에 대해 벡터DB에서 관련 문서를 검색합니다.
        
        Args:
            query: 검색할 질문
            threshold: 임베딩 유사도 기준값 (기본값: 0.4)
            
        Returns:
            str: 검색된 문서들을 하나의 문자열로 합친 컨텍스트
            
        Raises:
            ModelException: 문서 검색 실패 시
        """
        try:
            if not isinstance(query, str):
                raise ModelException("query는 문자열이어야 합니다.")
            if not isinstance(threshold, float) or threshold <= 0 or threshold > 1:
                raise ModelException("threshold는 0과 1 사이의 실수여야 합니다.")

            docs = retrieve_docs(query, threshold)
            
            if not docs:
                return "관련 문서가 없습니다."
                
            # 여러 문서를 하나의 context로 합침
            context_lines = [f"문서 {i+1}: {doc}" for i, doc in enumerate(docs)]
            context = "\n".join(context_lines)
            return context
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            raise ModelException(f"문서 검색 실패: {str(e)}")


    def answer(self, 
                query: str, 
                temperature: float = 0.3,
                top_p: float = 0.9,
                top_k: int = 50,
                max_tokens: int = 500) -> str:
        """
        사용자의 질문(query)에 대해 RAG 검색 결과(context)를 활용해 LLM 답변을 생성합니다.
        
        Args:
            query: 사용자 질문
            temperature: 응답의 다양성을 조절하는 값 (0~2, 기본값: 0.3)
            top_p: 누적 확률 기반 샘플링을 위한 임계값 (0~1, 기본값: 0.9)
            top_k: 다음 토큰 선택 시 고려할 상위 k개의 토큰 수 (양의 정수, 기본값: 50)
            max_tokens: 생성할 최대 토큰 수 (양의 정수, 기본값: 500)
            
        Returns:
            str: LLM이 생성한 답변
            
        Raises:
            ModelException: 답변 생성 과정에서 오류 발생 시
        """
        try:
            # 입력값 검증
            if not query or not isinstance(query, str):
                raise ModelException("유효한 질문이 필요합니다.")
            if not isinstance(temperature, float) or temperature < 0 or temperature > 2:
                raise ModelException("temperature는 0과 2 사이의 실수여야 합니다.")
            if not isinstance(top_p, float) or top_p <= 0 or top_p > 1:
                raise ModelException("top_p는 0과 1 사이의 실수여야 합니다.")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ModelException("top_k는 양의 정수여야 합니다.")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ModelException("max_tokens는 양의 정수여야 합니다.")
                
            # RAG 검색 및 답변 생성
            context = self.retrieve(query)
            messages = self.llm.make_message(context, query)
            llm_answer = self.llm.generate_answer(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens
            )
            
            if not llm_answer:
                raise ModelException("LLM이 유효한 답변을 생성하지 못했습니다.")
                
            return llm_answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {str(e)}")
            raise ModelException(f"답변 생성 실패: {str(e)}")
