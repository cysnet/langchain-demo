import os
import sys

sys.path.append(os.getcwd())

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
import time


from typing import TypedDict, Annotated

from llm_set import llm_env

llm = llm_env.llm


# mysql
db = SQLDatabase.from_uri(
    "mysql+pymysql://root:123456@localhost:3306/javademo",
    engine_args={"pool_size": 5, "max_overflow": 10},
)


class State(TypedDict):
    """State for the demo."""

    question: str
    query: str
    result: str
    answer: str
    approved: bool


system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""


user_prompt = "Question:{input}"


query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("human", user_prompt)],
)


class QueryOutput(TypedDict):
    """Generated the SQL query."""

    query: Annotated[str, "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 5,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )

    structured_llm = llm.with_structured_output(QueryOutput)

    result = structured_llm.invoke(prompt)

    return {"query": result["query"]}


def wait_for_user_approve(state: State):
    """Pause here and wait for user approval before executing query."""
    try:
        user_approval = input("Do you want to go to execute query? (yes/no): ")
    except Exception:
        user_approval = "no"

    if user_approval.lower() == "yes":
        return {
            "query": state["query"],
            "approved": True,
        }
    else:
        return {
            "query": state["query"],
            "approved": False,
        }


def excute_query(state: State):
    """Execute the SQL query and return the result."""
    if state["approved"]:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}
    else:
        return {"result": "excute denied."}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    if state["approved"]:
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}

    else:
        prompt = f'{"同意" if state["approved"] else "拒绝"} 用户拒绝当前执行'
        response = llm.invoke(prompt)
        return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence(
    [write_query, wait_for_user_approve, excute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")

DB_URI = "postgresql://postgres:123456@localhost:5432/langchaindemo?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    input_thread_id = input("输入thread_id:")
    time_str = time.strftime("%Y%m%d", time.localtime())
    config = {"configurable": {"thread_id": f"{time_str}-{input_thread_id}-agent-demo"}}

    graph = graph_builder.compile(checkpointer=checkpointer)

    print("输入问题，输入 exit 退出。")
    while True:
        query = input("你: ")
        if query.strip().lower() == "exit":
            break
        response = graph.invoke(
            {"question": query},
            config,
        )

        print(response)
