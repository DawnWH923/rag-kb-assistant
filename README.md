# RAG 智能知识库问答助手

基于 LangChain + Chroma + DeepSeek 大模型的 Retrieval-Augmented Generation (RAG) 系统，支持多格式文档入库、向量检索 + Cross-Encoder 精排、带引用来源的问答，并提供 FastAPI RESTful 接口。

## 功能
- 📄 支持 PDF / Markdown / TXT 文档自动加载与切分
- 🔍 BGE 中文 Embedding 向量召回 + BGE Reranker Cross-Encoder 精排（召回 8 → 精排 4）
- 💬 接入 DeepSeek 大模型，生成带引用来源的回答，上下文不足时拒答不编造
- 🌐 FastAPI 封装 RESTful 接口（`/ingest`、`/ask`），自带交互文档
- ⌨️ 同样支持命令行问答（`ask.py`）

## 技术栈
| 模块 | 技术 |
|------|------|
| LLM | DeepSeek-Chat (OpenAI 兼容 API) |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Rerank | BAAI/bge-reranker-base (Cross-Encoder) |
| 向量库 | ChromaDB |
| 框架 | LangChain (LCEL) |
| 后端 | FastAPI + Uvicorn |

## 检索架构
```
用户问题 ──► BGE 向量化 ──► Chroma 向量召回 top-8（双塔，快但粗）
                                  │
                                  ▼
                    （问题, 文档）两两拼接送入 Cross-Encoder 打相关分
                                  │
                                  ▼
                    精排取 top-4（慢但准）──► 拼提示词 ──► DeepSeek 生成
```
双塔召回负责覆盖面，Cross-Encoder 精排负责准确度——用便宜的召回保上限，用昂贵的精排保质量。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
创建 `.env` 文件：
```
DEEPSEEK_API_KEY=your_api_key
```

### 3. 构建索引
将文档放入 `data/` 目录，运行：
```bash
python ingest.py
```

### 4. 提问（两种方式）

**命令行：**
```bash
python ask.py
```

**REST API：**
```bash
uvicorn api:app --reload --port 8000
# 浏览器打开 http://localhost:8000/docs 可在线调试

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RFM模型？"}'

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder": "data"}'
```

**Docker 一键部署：**
```bash
docker build -t rag-kb-assistant .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  rag-kb-assistant
# 容器起来后先 POST /ingest 构建索引，再 POST /ask
```

## 项目结构
```
rag-kb-assistant/
├── data/               # 文档目录（PDF/MD/TXT，不入库）
├── chroma_db/          # 向量库（自动生成，不入库）
├── .env                # API 密钥（不入库）
├── ingest.py           # 构建索引（离线一次）
├── ask.py              # 命令行问答
├── rerank.py           # Cross-Encoder 精排
├── api.py              # FastAPI 服务（/ingest /ask /health）
├── test_query.py       # 快速冒烟测试（3个固定问题）
└── requirements.txt
```

## 使用示例
```
你：什么是RFM模型？
AI：RFM模型是按最近一次消费（Recency）、消费频率（Frequency）、消费金额（Monetary）三个维度给用户打分。
```

## 作者
吴晗 | [GitHub](https://github.com/DawnWH923)
