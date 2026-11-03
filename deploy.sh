#!/usr/bin/env bash
# deploy.sh — 腾讯云轻量应用服务器一键部署/更新 RAG 服务
# 前置：服务器已安装 Docker；仓库已 clone；已执行 cp .env.example .env 并填好 DEEPSEEK_API_KEY 与 API_TOKEN
# 用法：bash deploy.sh
set -euo pipefail

IMAGE_NAME=rag-kb-assistant
CONTAINER_NAME=rag-api
PORT=8000

command -v docker >/dev/null || { echo "❌ 请先安装 Docker：curl -fsSL https://get.docker.com | bash"; exit 1; }
[ -f .env ] || { echo "❌ 缺少 .env：请先 cp .env.example .env 并填写 DEEPSEEK_API_KEY 和 API_TOKEN"; exit 1; }
grep -q "^API_TOKEN=.\+" .env || { echo "❌ .env 里 API_TOKEN 为空：公网部署必须设置鉴权 token"; exit 1; }

echo "[1/3] 构建镜像..."
docker build -t "$IMAGE_NAME" .

echo "[2/3] 重启容器..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file .env \
  -p "$PORT":8000 \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/chroma_db":/app/chroma_db \
  "$IMAGE_NAME"

echo "[3/3] 健康检查（模型冷启动约 20-60 秒）..."
for i in $(seq 1 30); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "✅ 部署成功：http://localhost:$PORT/health"
    echo "后续调用 /ingest /ask 需带请求头：X-API-Token: <你 .env 里的 API_TOKEN>"
    echo "首次使用先建索引：curl -X POST http://localhost:$PORT/ingest -H \"X-API-Token: \$TOKEN\" -H \"Content-Type: application/json\" -d '{\"folder\":\"data\"}'"
    exit 0
  fi
  sleep 2
done
echo "❌ 健康检查超时，排查：docker logs $CONTAINER_NAME"
exit 1
