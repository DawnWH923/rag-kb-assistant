# REST API 参考

启动：`uvicorn api:app --reload --port 8000`，交互文档 `http://localhost:8000/docs`。

## 鉴权

`.env` 设置 `API_TOKEN` 后，`/ingest`、`/ask` 要求请求头 `X-API-Token` 匹配（`secrets.compare_digest` 恒时比较，防时序侧信道）；未设置时本地开放。`GET /health` 永远免鉴权。

## GET /health

健康检查。

```json
{"status": "ok"}
```

## POST /ingest

把目录下的 PDF / Markdown / TXT 切块入库。

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" -H "X-API-Token: your_token" \
  -d '{"folder": "data"}'
```

| 字段 | 必填 | 说明 |
|------|------|------|
| folder | 是 | 文档目录（相对路径） |

流程：DirectoryLoader（PyPDF/Text 按扩展名）→ RecursiveCharacterTextSplitter 切分 → BGE 向量化 → Chroma 持久化（`chroma_db/`）。

## POST /ask

提问并返回带引用来源的回答。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" -H "X-API-Token: your_token" \
  -d '{"question": "什么是RFM模型？"}'
```

| 字段 | 必填 | 说明 |
|------|------|------|
| question | 是 | 用户问题 |

流程：问题向量化 → Chroma 召回 top-8 → BGE Cross-Encoder 精排 top-4 → 拼上下文 → DeepSeek 生成。上下文不足时返回"不知道"（拒答不编造），并附候选来源。

## 部署

`Dockerfile` + `deploy.sh` 一键部署；公网部署务必设置 `API_TOKEN`。
