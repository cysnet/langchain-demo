import os
import sys

sys.path.append(os.getcwd())

from llm_set import llm_env

model = llm_env.llm

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    trim_messages,
    SystemMessage,
)

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, TypedDict


trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你说话像个卡通人物。尽你所能按照语言{language}回答所有问题。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "call_model")
workflow.add_node("call_model", call_model)

DB_URI = "postgresql://postgres:123456@localhost:5432/langchaindemo?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    app = workflow.compile(checkpointer=checkpointer)

    input_thread_id = input("输入thread_id:")
    config = {"configurable": {"thread_id": input_thread_id}}
    language = "中文"

    print("输入问题，输入 exit 退出。")
    while True:
        query = input("你: ")
        if query.strip().lower() == "exit":
            break
        input_messages = [HumanMessage(query)]
        output = app.invoke({"messages": input_messages, "language": language}, config)
        for message in output["messages"]:
            print(f"{message.type}: {message.content}")
