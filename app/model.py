from .retrieval import retrieve_docs
from groq import Groq

class Model:
    """
    RAG 파이프라인에서 LLM을 활용해 답변을 생성하는 모델 클래스입니다.
    - api_key: LLM API 키
    - llm_client: Groq LLM 클라이언트
    """
    def __init__(self, api_key: str):
        """Model 객체를 초기화합니다. api_key는 LLM 서비스 인증에 사용됩니다."""
        self.llm_client = Groq(api_key=api_key)


    def make_message(self, context: str, query: str):
        """
        LLM에 전달할 메시지 리스트를 생성합니다.
        - context: 검색된 문서들을 하나의 문자열로 합친 값
        - query: 사용자의 질문
        """
        messages = [
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": query
            }
        ]
        return messages
    

    def make_llm_message(self, messages: str):
        """
        groq LLM 클라이언트를 사용해 채팅 완성 메시지를 생성합니다.
        - messages: LLM에 전달할 메시지 리스트
        """
        llm_answer = self.llm_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,  
            max_completion_tokens=3000, 
            top_p=0.9, 
            temperature=0.3, 
            stream=True,
            stop=None
        )

        # groq의 LLM은 스트리밍 응답을 제공하므로, 모든 청크를 합쳐 최종 답변 생성
        answer = []
        for chunk in llm_answer:
            answer.append(chunk.choices[0].delta.content or "")
        answer = ''.join(answer)
        return answer


    def retrieve(self, query: str, threshold: float = 0.4):
        """
        주어진 쿼리(query)에 대해 벡터DB에서 관련 문서를 검색합니다.
        - threshold: 임베딩 유사도 기준값 (기본값: 0.4)
        """
        docs = retrieve_docs(query)
        # 여러 문서를 하나의 context로 합침
        context_lines = [f"문서 {i+1}: {doc}" for i, doc in enumerate(docs)]
        context = "\n".join(context_lines)
        return context


    def answer(self, query: str) -> str:
        """
        사용자의 질문(query)에 대해 RAG 검색 결과(context)를 활용해 LLM 답변을 생성합니다.
        - 검색된 문서들을 context로 합쳐 LLM에 전달
        - LLM의 스트리밍 응답을 모두 합쳐 최종 답변 반환
        """
        context = self.retrieve(query)
        messages = self.make_message(context, query)
        llm_answer = self.make_llm_message(messages)
        return llm_answer
