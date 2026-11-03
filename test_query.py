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
llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
prompt = ChatPromptTemplate.from_template("根据上下文回答，不知道就说不知道。\n上下文：{context}\n问题：{question}\n回答：")
chain = ({"context": vs.as_retriever(search_kwargs={"k":4}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

for q in ["公司是做什么的？", "怎么绑定车辆？", "DAU怎么定义？"]:
    print("问：", q)
    print("答：", chain.invoke(q))
    print()
