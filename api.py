"""
api.py — FastAPI 封装 REST 接口
启动：uvicorn api:app --reload --port 8000
交互文档：http://localhost:8000/docs
接口：
  GET  /health  健康检查（免鉴权）
  POST /ingest  把指定目录下的 PDF/Markdown 切块入库
  POST /ask     提问（向量召回 recall_k -> Cross-Encoder 精排 top_n -> DeepSeek 生成）
鉴权：
  环境变量 API_TOKEN 未设置时接口开放（本地开发）；
  设置后 /ingest /ask 要求请求头 X-API-Token 匹配，用于公网部署防护。
"""
import os
import secrets
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rerank import rerank

load_dotenv()

EMB_MODEL = "BAAI/bge-small-zh-v1.5"
CHROMA_DIR = "chroma_db"

emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

PROMPT = ChatPromptTemplate.from_template(
    """请根据以下上下文回答问题。如果上下文里没有答案，就说不知道，不要编造。
上下文：{context}
问题：{question}
回答："""
)

app = FastAPI(title="RAG 知识库问答 API", version="1.2.0")


def verify_token(x_api_token: str | None = Header(default=None, alias="X-API-Token")):
    """公网部署防护：校验固定 token；compare_digest 防时序侧信道"""
    expected = os.getenv("API_TOKEN", "")
    if expected and not secrets.compare_digest(x_api_token or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token")


class IngestRequest(BaseModel):
    folder: str = Field(default="data", description="文档所在目录")


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    recall_k: int = Field(default=8, ge=1, le=20, description="向量召回条数")
    top_n: int = Field(default=4, ge=1, le=10, description="精排后送入 LLM 的条数")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", dependencies=[Depends(verify_token)])
def ingest(req: IngestRequest):
    """读取目录下的 PDF/Markdown，切分后写入向量库（可重复调用，增量追加）"""
    try:
        docs = (
            DirectoryLoader(req.folder, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
            + DirectoryLoader(req.folder, glob="**/*.md", loader_cls=TextLoader).load()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文档读取失败: {e}")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    if chunks:
        Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR)
    return {"status": "ok", "documents": len(docs), "chunks": len(chunks)}


@app.post("/ask", dependencies=[Depends(verify_token)])
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    candidates = vs.as_retriever(search_kwargs={"k": req.recall_k}).invoke(req.question)
    top_texts = rerank(req.question, [d.page_content for d in candidates], top_n=req.top_n)
    context = "\n\n".join(top_texts)
    answer = (PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": req.question}
    )
    return {"answer": answer, "references": top_texts}
