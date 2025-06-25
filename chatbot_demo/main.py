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


from langgraph.checkpoint.memory import MemorySaver
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
        ("system", "你说话像个海盗。尽你所能按照语言{language}回答所有问题。"),
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


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "chester_123"}}

    query = "你好，我是Chester"

    messages = [
        SystemMessage(content="你是一个好的助手"),
        HumanMessage(content="请帮我解答一些问题"),
        AIMessage(content="hi!"),
        HumanMessage(content="我喜欢冰淇淋"),
        AIMessage(content="太好了"),
        HumanMessage(content="2=2=?"),
        AIMessage(content="4"),
        HumanMessage(content="谢谢"),
        AIMessage(content="没问题!"),
    ]

    input_messages = messages + [HumanMessage(query)]

    language = "中文"
    output = app.invoke({"messages": input_messages, "language": language}, config)

    print("Output Messages:")
    for message in output["messages"]:
        print(f"{message.type}: {message.content}")

    query = "我是谁？"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke({"messages": input_messages, "language": language}, config)

    print("Output Messages:")
    for message in output["messages"]:
        print(f"{message.type}: {message.content}")
