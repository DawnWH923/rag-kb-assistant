# RAG 智能知识库问答助手

基于 LangChain + Chroma + DeepSeek 大模型的 Retrieval-Augmented Generation (RAG) 系统，支持多格式文档入库、向量检索与带引用的问答。

## 功能
- 📄 支持 PDF / Markdown / TXT 文档自动加载与切分
- 🔍 基于 BGE 中文 Embedding 的向量检索 + Rerank 精排
- 💬 接入 DeepSeek 大模型，生成带引用来源的回答
- 🌐 FastAPI 封装 RESTful 接口

## 技术栈
| 模块 | 技术 |
|------|------|
| LLM | DeepSeek-Chat (OpenAI 兼容 API) |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量库 | ChromaDB |
| 框架 | LangChain |
| 后端 | FastAPI + Uvicorn |

## 快速开始

### 1. 安装依赖
```bash
pip install langchain langchain-community langchain-openai langchain-huggingface chromadb sentence-transformers openai fastapi pypdf python-dotenv
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

### 4. 启动问答
```bash
python ask.py
```

## 项目结构
```
rag-kb-assistant/
├── data/               # 文档目录（PDF/MD/TXT）
├── chroma_db/          # 向量库（自动生成）
├── .env                # API 密钥
├── ingest.py           # 构建索引
├── ask.py              # 命令行问答
└── README.md
```

## 使用示例
```
你：什么是RFM模型？
AI：RFM模型是按最近一次消费（Recency）、消费频率（Frequency）、消费金额（Monetary）三个维度给用户打分。
```

## 作者
吴晗 | [GitHub](https://github.com/DawnWH923)
