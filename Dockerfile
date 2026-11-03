FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存：代码改动不用重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝代码（.env / chroma_db / data 由 .dockerignore 排除，运行时挂载或调 /ingest 构建）
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
