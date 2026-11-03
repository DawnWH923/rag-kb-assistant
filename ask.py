"""
ask.py — 命令行和你的文档聊天
先运行 ingest.py 建好索引，再运行 python ask.py
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

prompt = ChatPromptTemplate.from_template("""请根据以下上下文回答问题。如果上下文里没有答案，就说不知道。
上下文：{context}
问题：{question}
回答：""")

chain = (
    {"context": vs.as_retriever(search_kwargs={"k": 4}), "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print("开始聊天，输入问题；输 exit 退出")
while True:
    q = input("\n你：")
    if q.strip() in ("exit", "quit", ""):
        break
    print("\nAI：", chain.invoke(q))
