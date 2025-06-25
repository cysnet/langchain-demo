import os
import sys
import time

sys.path.append(os.getcwd())


from llm_set import llm_env

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


search = TavilySearchResults(max_results=5)

tools = [add, search]


DB_URI = "postgresql://postgres:123456@localhost:5432/langchaindemo?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    input_thread_id = input("输入thread_id:")
    time_str = time.strftime("%Y%m%d", time.localtime())
    config = {"configurable": {"thread_id": f"{time_str}-{input_thread_id}-agent-demo"}}

    agent_excuter = create_react_agent(llm_env.llm, tools, checkpointer=checkpointer)

    print("输入问题，输入 exit 退出。")
    while True:
        query = input("你: ")
        if query.strip().lower() == "exit":
            break
        input_messages = [HumanMessage(query)]
        response = agent_excuter.invoke({"messages": input_messages}, config=config)
        for message in response["messages"]:
            if hasattr(message, "content") and message.content:
                print(f"{message.type}:{message.content}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"{message.type}:{message.tool_calls}")
