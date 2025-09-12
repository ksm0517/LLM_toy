"""Base class for LLM implementations"""
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any

class BaseLLM(ABC):
    """LLM 구현을 위한 추상 기본 클래스"""

    @abstractmethod
    def make_message(self, context: str, query: str) -> Union[List[Dict[str, str]], str]:
        """
        LLM에 전달할 메시지/프롬프트를 생성합니다.
        
        Args:
            context: 검색된 문서들의 컨텍스트
            query: 사용자 질문
            
        Returns:
            Union[List[Dict[str, str]], str]: LLM에 전달할 메시지
        """
        pass


    @abstractmethod
    def generate_answer(self, 
                       messages: Union[List[Dict[str, str]], str], 
                       temperature: float,
                       top_p: float,
                       top_k: int,
                       max_tokens: int) -> str:
        """
        LLM을 사용해 답변을 생성합니다.
        
        Args:
            messages: LLM에 전달할 메시지/프롬프트
            temperature: 응답의 다양성을 조절하는 값 (0에 가까울수록 일관된 응답)
            top_p: 누적 확률 기반 샘플링을 위한 임계값
            top_k: 다음 토큰 선택 시 고려할 상위 k개의 토큰 수
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 답변
        """
        pass
