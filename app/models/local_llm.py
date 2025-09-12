"""Local LLM implementation using TinyLlama"""
import logging
from typing import Optional, List, Dict, Any, Union
import torch
from transformers import pipeline

from ..utils.exceptions import ModelException
from .base_llm import BaseLLM

# 로깅 설정
logger = logging.getLogger(__name__)

class LocalLLM(BaseLLM):
    """TinyLlama를 사용하는 로컬 LLM 구현"""
    
    def __init__(self):
        """
        LocalLLM 객체를 초기화합니다.
        
        Raises:
            ModelException: 모델 로딩 실패 시
        """
        try:
            self.llm = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"로컬 LLM 초기화 실패: {str(e)}")
            raise ModelException(f"로컬 LLM 초기화 실패: {str(e)}")


    def make_message(self, context: str, query: str) -> str:
        """
        로컬 LLM에 전달할 프롬프트 문자열을 생성합니다.
        
        Args:
            context: 검색된 문서들의 컨텍스트
            query: 사용자 질문
            
        Returns:
            str: 생성된 프롬프트 문자열
            
        Raises:
            ModelException: 메시지 생성 실패 시
        """
        try:
            if not isinstance(query, str):
                raise ModelException("query는 문자열이어야 합니다.")
                
            context = context if context else "관련 문서가 없습니다."
            prompt = f"{context}\n---\n질문: {query}\n"
            return prompt
            
        except Exception as e:
            logger.error(f"메시지 생성 실패: {str(e)}")
            raise ModelException(f"메시지 생성 실패: {str(e)}")


    def generate_answer(self, 
                       messages: str, 
                       temperature: float,
                       top_p: float,
                       top_k: int,
                       max_tokens: int) -> str:
        """
        로컬 LLM을 사용해 답변을 생성합니다.
        
        Args:
            messages: 프롬프트 문자열
            temperature: 응답의 다양성을 조절하는 값 (0에 가까울수록 일관된 응답)
            top_p: 누적 확률 기반 샘플링을 위한 임계값
            top_k: 다음 토큰 선택 시 고려할 상위 k개의 토큰 수
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 답변
            
        Raises:
            ModelException: 모델 실행 실패 시
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
            
            result = self.llm(
                messages,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                truncation=True
            )
            
            if not result or not result[0]['generated_text']:
                raise ModelException("로컬 LLM이 빈 응답을 반환했습니다.")
            
            return result[0]['generated_text'].strip()
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU 메모리 부족")
            raise ModelException("GPU 메모리 부족")
        except Exception as e:
            logger.error(f"로컬 LLM 실행 실패: {str(e)}")
            raise ModelException(f"로컬 LLM 실행 실패: {str(e)}")
