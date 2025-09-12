"""Groq LLM implementation"""
import logging
from typing import Optional, List, Dict, Any, Union

from groq import Groq
from ..utils.exceptions import ModelException
from .base_llm import BaseLLM

# 로깅 설정
logger = logging.getLogger(__name__)

class GroqLLM(BaseLLM):
    """Groq API를 사용하는 LLM 구현"""
    
    def __init__(self, api_key: str):
        """
        GroqLLM 객체를 초기화합니다.
        
        Args:
            api_key: Groq API 키
            
        Raises:
            ModelException: API 키가 없거나 초기화 실패 시
        """
        try:
            if not api_key:
                raise ModelException("Groq API 키가 필요합니다.")
            self.llm = Groq(api_key=api_key)
        except Exception as e:
            logger.error(f"Groq LLM 초기화 실패: {str(e)}")
            raise ModelException(f"Groq LLM 초기화 실패: {str(e)}")


    def make_message(self, context: str, query: str) -> List[Dict[str, str]]:
        """
        Groq API에 전달할 메시지 리스트를 생성합니다.
        
        Args:
            context: 검색된 문서들의 컨텍스트
            query: 사용자 질문
            
        Returns:
            List[Dict[str, str]]: Groq API 형식의 메시지 리스트
            
        Raises:
            ModelException: 메시지 생성 실패 시
        """
        try:
            if not isinstance(query, str):
                raise ModelException("query는 문자열이어야 합니다.")
                
            messages = [
                {
                    "role": "system",
                    "content": context or "관련 문서가 없습니다."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            return messages
            
        except Exception as e:
            logger.error(f"메시지 생성 실패: {str(e)}")
            raise ModelException(f"메시지 생성 실패: {str(e)}")


    def generate_answer(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float,
                       top_p: float,
                       top_k: int,
                       max_tokens: int) -> str:
        """
        Groq API를 사용해 답변을 생성합니다.
        
        Args:
            messages: Groq API 형식의 메시지 리스트
            temperature: 응답의 다양성을 조절하는 값 (0에 가까울수록 일관된 응답)
            top_p: 누적 확률 기반 샘플링을 위한 임계값
            top_k: 다음 토큰 선택 시 고려할 상위 k개의 토큰 수
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 답변
            
        Raises:
            ModelException: API 호출 실패 또는 응답 오류 시
        """
        try:
            # 파라미터 유효성 검사
            if not isinstance(temperature, float) or temperature < 0 or temperature > 2:
                raise ModelException("temperature는 0과 2 사이의 실수여야 합니다.")
            if not isinstance(top_p, float) or top_p <= 0 or top_p > 1:
                raise ModelException("top_p는 0과 1 사이의 실수여야 합니다.")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ModelException("top_k는 양의 정수여야 합니다.")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ModelException("max_tokens는 양의 정수여야 합니다.")
            
            llm_answer = self.llm.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=True,
                stop=None
            )
            
            answer = []
            for chunk in llm_answer:
                if chunk.choices and chunk.choices[0].delta:
                    answer.append(chunk.choices[0].delta.content or "")
            answer = ''.join(answer)
            
            if not answer:
                raise ModelException("Groq API가 빈 응답을 반환했습니다.")
                
            return answer
            
        except Exception as e:
            logger.error(f"Groq API 호출 실패: {str(e)}")
            raise ModelException(f"Groq API 호출 실패: {str(e)}")
