FROM python:3.11-slim

WORKDIR /app

# 先装 CPU 版 torch（PyPI 默认 wheel 含 CUDA 约 900MB，云服务器无 GPU 纯浪费；CPU 版约 190MB）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖（清华镜像加速；torch 已满足依赖，不会再拉 CUDA 版）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷贝代码（.env / chroma_db / data 由 .dockerignore 排除，运行时挂载或调 /ingest 构建）
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
