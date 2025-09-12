# 베이스 이미지: 공식 Python 3.9 slim
FROM python:3.9-slim

# 작업 디렉토리 생성 및 이동
WORKDIR /app

# 시스템 패키지 및 빌드 툴 설치
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y git && \
    apt-get clean

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 환경 변수 예시 (필요시 수정)
ENV PYTHONUNBUFFERED=1

# uvicorn으로 FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
