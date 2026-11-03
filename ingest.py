"""
ingest.py — 把 data/ 文件夹里的 PDF/Markdown 切块、向量化，存进 Chroma
用法：先把文档放进 data/ 文件夹，然后运行 python ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. 读取 data/ 下所有 pdf 和 md
docs = []
pdfs = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader).load()
mds = DirectoryLoader("data", glob="**/*.md", loader_cls=TextLoader).load()
docs = pdfs + mds
print(f"读到 {len(docs)} 个文档块")

# 2. 切分
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
print(f"切分成 {len(chunks)} 个片段")

# 3. BGE 中文 Embedding + Chroma
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
print("✅ 入库完成！向量已存到 chroma_db/")
