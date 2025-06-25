import os
import sys

sys.path.append(os.getcwd())


from llm_set import llm_env
from langchain.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState,StateGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.postgres import PostgresSaver 
import time  # 导入time模块



llm = llm_env.llm

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_rag_agent_docs",
    connection="postgresql+psycopg2://postgres:123456@localhost:5433/langchainvector",
)


url = "https://www.cnblogs.com/chenyishi/p/18926783"
loader = WebBaseLoader(
    web_paths=(url,),
)

docs = loader.load()
for doc in docs:
    doc.metadata["source"] = url 


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)


existing = vector_store.similarity_search(url, k=1, filter={"source": url})
if not existing:
    _ = vector_store.add_documents(documents=all_splits)
    print("文档向量化完成")


@tool(response_format="content_and_artifact")
def retrieve(query: str) -> tuple[str, dict]:
    """Retrieve relevant documents from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    if not retrieved_docs:
        return "No relevant documents found.", {}
    return "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    ), retrieved_docs


def query_or_respond(state:MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}


tools = ToolNode([retrieve])

def generate(state:MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    system_message_content = "\n\n".join(doc.content for doc in tool_messages)

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}



graph_builder= StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    path_map={END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)



DB_URI = "postgresql://postgres:123456@localhost:5433/langchaindemo?sslmode=disable"  # 检查点数据库连接
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:  # 创建检查点保存器
    checkpointer.setup()  # 初始化检查点

    graph = graph_builder.compile(checkpointer=checkpointer)  # 编译图并设置检查点


    input_thread_id = input("输入thread_id:")  # 输入线程ID
    time_str = time.strftime("%Y%m%d", time.localtime())  # 获取当前日期字符串
    config = {"configurable": {"thread_id": f"rag-{time_str}-demo-{input_thread_id}"}}  # 配置线程ID

    print("输入问题，输入 exit 退出。")  # 提示用户输入
    while True:
        query = input("你: ")  # 用户输入问题
        if query.strip().lower() == "exit":  # 输入exit则退出
            break
        response = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)  # 执行RAG流程
        print(response["messages"][-1][""])  # 输出答案

    # input_message = "文章的介绍了什么？?"

    # response = graph.invoke({"messages": [HumanMessage(content=input_message)]})
    # print(response)