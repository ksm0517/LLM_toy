from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from .model import Model
import os
from dotenv import load_dotenv

class ChatApp:
    """
    FastAPI 기반 챗봇 웹앱을 관리하는 클래스입니다.
    - 환경 변수 로드
    - FastAPI 및 Jinja2 객체 초기화
    - 모델 객체(Model) 생성
    - 라우트 등록
    """
    def __init__(self):
        """
        ChatApp 클래스의 생성자.
        환경 변수 로드, FastAPI/Jinja2 객체 및 모델 객체 초기화, 라우트 등록을 수행합니다.
        """
        load_dotenv(dotenv_path="./environment.env")
        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="app/templates")
        self.api_key = os.getenv("MY_API_KEY")
        self.model_class = Model(api_key=self.api_key)
        self._add_routes()

    def _add_routes(self):
        """
        FastAPI 라우트(GET/POST)를 등록하는 함수.
        home: 메인 페이지 렌더링
        ask: 질문 입력 시 답변 생성 및 렌더링
        """
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """
            메인 페이지(GET) - 최초 접속 시 채팅 입력 폼만 보여줌
            """
            return self.templates.TemplateResponse("index.html", {"request": request, "answer": None})

        @self.app.post("/", response_class=HTMLResponse)
        async def ask(request: Request, user_input: str = Form(...)):
            """
            질문 입력 시 POST 요청 처리
            모델에 질문(user_input)을 전달해 답변 생성 후 렌더링
            """
            answer = self.model_class.answer(user_input)
            return self.templates.TemplateResponse("index.html", {"request": request, "answer": answer, "user_input": user_input})

# 인스턴스 생성 및 FastAPI 앱 객체 노출
chat_app = ChatApp()  # 챗봇 웹앱 클래스 인스턴스 생성
app = chat_app.app    # FastAPI 앱 객체를 외부에 노출