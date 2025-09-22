from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .model import Model
from .loader import DocumentLoader
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.llm_type = os.getenv("LLM_TYPE", "local")  # 기본값은 "local"
        
        BASE_DIR = Path(__file__).resolve().parent

        # 디렉토리 설정
        self.upload_dir = BASE_DIR / "uploads"
        self.vector_store_path = BASE_DIR / "vector_store"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # 정적 파일 서빙 설정
        self.app.mount(
            "/static", 
            StaticFiles(directory=str(BASE_DIR / "static")), 
            name="static"
        )
        
        # 모델 및 문서 로더 초기화
        self.model_class = Model(api_key=self.api_key, llm_type=self.llm_type)
        self.doc_loader = DocumentLoader(persistence_path=str(self.vector_store_path))
        
        self._add_routes()

    def _get_context(self, query: str) -> Optional[str]:
        """주어진 쿼리에 대한 컨텍스트를 검색합니다."""
        try:
            search_results = self.doc_loader.search_documents(query)
            if not search_results:
                return None
            
            context_lines = [f"문서 {i+1}: {result['content']}" for i, result in enumerate(search_results)]
            return "\n".join(context_lines)
        except Exception as e:
            logger.error(f"컨텍스트 검색 중 오류 발생: {e}")
            return None

    def _add_routes(self):
        """
        FastAPI 라우트(GET/POST)를 등록하는 함수.
        home: 메인 페이지 렌더링
        ask: 질문 입력 시 답변 생성 및 렌더링
        health: 쿠버네티스 헬스 체크용 엔드포인트
        """
        
        @self.app.get("/health")
        async def health_check():
            """쿠버네티스 헬스 체크용 엔드포인트"""
            return {"status": "healthy"}
        
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
            context = self._get_context(user_input)
            answer = self.model_class.answer(user_input, context=context)
            return self.templates.TemplateResponse("index.html", {"request": request, "answer": answer, "user_input": user_input})
            
        @self.app.post("/add_text", response_class=JSONResponse)
        async def add_text(request: Request, text_input: str = Form(...)):
            """
            사용자가 입력한 텍스트를 VectorStore에 추가합니다.
            """
            try:
                if not text_input or not text_input.strip():
                    raise HTTPException(status_code=400, detail="내용이 비어있습니다.")
                
                # DocumentLoader에 텍스트를 추가하는 메서드 호출 (DocumentLoader에 구현 필요)
                self.doc_loader.add_text_document(text_input)
                
                return JSONResponse(content={"message": "텍스트가 성공적으로 추가되었습니다."}, status_code=200)
            except Exception as e:
                logger.error(f"텍스트 추가 중 오류 발생: {e}")
                raise HTTPException(status_code=500, detail=f"텍스트 추가 중 오류가 발생했습니다: {str(e)}")

        @self.app.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            """
            PDF 파일 업로드 처리
            업로드된 파일을 저장하고 문서 처리를 수행
            """
            try:
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
                
                file_path = self.upload_dir / file.filename
                with file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                try:
                    self.doc_loader.load_pdf(str(file_path))
                    return JSONResponse(
                        content={"message": "파일이 성공적으로 업로드되고 처리되었습니다."},
                        status_code=200
                    )
                except Exception as e:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=500, detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}")


# 인스턴스 생성 및 FastAPI 앱 객체 노출
chat_app = ChatApp()  # 챗봇 웹앱 클래스 인스턴스 생성
app = chat_app.app    # FastAPI 앱 객체를 외부에 노출