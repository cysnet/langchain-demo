import os  # 导入os模块，用于操作系统相关功能
import sys  # 导入sys模块，用于操作Python运行时环境

sys.path.append(os.getcwd())  # 将当前工作目录添加到模块搜索路径

from llm_set import llm_env  # 导入自定义的llm_env对象
from langchain_openai import OpenAIEmbeddings  # 导入OpenAIEmbeddings用于生成向量
from langchain_postgres import PGVector  # 导入PGVector用于向量存储
from langchain_community.document_loaders import WebBaseLoader  # 导入网页加载器
from langchain_core.documents import Document  # 导入Document文档对象
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入文本分割器
from langgraph.graph import START,StateGraph  # 导入图相关类
from typing_extensions import List,TypedDict,Annotated  # 导入类型扩展
from typing import Literal  # 导入Literal类型
from langgraph.checkpoint.postgres import PostgresSaver  # 导入PostgresSaver用于检查点保存
import time  # 导入time模块
from langgraph.graph.message import add_messages  # 导入消息添加工具
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
)  # 导入消息相关类
from langchain_core.prompts import ChatPromptTemplate  # 导入聊天提示模板

llm = llm_env.llm  # 获取大语言模型对象

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 初始化OpenAI嵌入模型

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_rag_docs",
    connection="postgresql+psycopg2://postgres:123456@localhost:5433/langchainvector",
)  # 初始化PGVector向量存储

url = "https://python.langchain.com/docs/tutorials/qa_chat_history/"  # 目标网页URL
loader = WebBaseLoader(
    web_paths=(url,),
)  # 创建网页加载器

docs = loader.load()  # 加载网页文档
for doc in docs:
    doc.metadata["source"] = url  # 为每个文档添加来源元数据


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)  # 创建递归文本分割器
all_splits = text_splitter.split_documents(docs)  # 分割文档

total_documents = len(all_splits)  # 获取总分割文档数
third = total_documents // 3  # 计算三分之一

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"  # 前三分之一标记为beginning
    elif i < 2 * third:
        document.metadata["section"] = "middle"  # 中间三分之一标记为middle
    else:
        document.metadata["section"] = "end"  # 最后三分之一标记为end

existing = vector_store.similarity_search(url, k=1, filter={"source": url})  # 检查向量库中是否已存在
if not existing:
    _ = vector_store.add_documents(documents=all_splits)  # 不存在则添加文档向量
    print("文档向量化完成")  # 输出完成提示


class Search(TypedDict):
    query:Annotated[str, "The question to be answered"]  # 查询问题
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]  # 查询的文档部分


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 消息列表
    query:Search  # 查询结构
    context:List[Document]  # 检索到的上下文文档
    answer:set  # 回答内容


def analyze(state:State):
    structtured_llm = llm.with_structured_output(Search)  # 使用结构化输出
    query = structtured_llm.invoke(state["messages"])  # 生成查询
    return {"query":query}  # 返回查询


def retrieve(state:State):
    query = state["query"]  # 获取查询
    if hasattr(query,'section'):
        filter={"section":query["section"]}  # 按section过滤
    else:
        filter=None  # 不过滤
    retrieved_docs = vector_store.similarity_search(query["query"],filter=filter)  # 相似度检索
    return {"context":retrieved_docs}  # 返回检索结果


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "尽你所能按照上下文:{context}，回答问题：{question}。"),
    ]
)  # 创建聊天提示模板


def generate(state:State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])  # 拼接上下文内容

    messages = prompt_template.invoke({
        "question":state["query"]["query"],
        "context": docs_content, 
    })  # 生成提示消息

    response = llm.invoke(messages)  # 调用大模型生成回答
    return {"answer":response.content,"messages": [response]}  # 返回回答和消息


graph_builder = StateGraph(State).add_sequence([analyze,retrieve,generate])  # 构建状态图
graph_builder.add_edge(START,"analyze")  # 添加起始边


DB_URI = "postgresql://postgres:123456@localhost:5433/langchaindemo?sslmode=disable"  # 检查点数据库连接
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:  # 创建检查点保存器
    checkpointer.setup()  # 初始化检查点

    graph = graph_builder.compile(checkpointer=checkpointer)  # 编译状态图
    input_thread_id = input("输入thread_id:")  # 输入线程ID
    time_str = time.strftime("%Y%m%d", time.localtime())  # 获取当前日期字符串
    config = {"configurable": {"thread_id": f"rag-{time_str}-demo-{input_thread_id}"}}  # 配置线程ID


    print("输入问题，输入 exit 退出。")  # 提示用户输入
    while True:
        query = input("你: ")  # 用户输入问题
        if query.strip().lower() == "exit":  # 输入exit则退出
            break
        input_messages = [HumanMessage(query)]  # 构造消息
        response = graph.invoke({"messages":input_messages}, config=config)  # 执行RAG流程
        print(response["answer"])  # 输出答案

