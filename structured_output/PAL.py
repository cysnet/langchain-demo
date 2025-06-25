import os
from langchain_community.utilities import SQLDatabase


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"


db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")


from typing import TypedDict


class State(TypedDict):
    """State for the agent."""

    question: str
    query: str
    result: str
    answer: str


from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


from langchain_core.prompts import ChatPromptTemplate

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

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

# for message in query_prompt_template.messages:
#     message.pretty_print()


from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Output of the query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )

    structured_query = llm.with_structured_output(QueryOutput)
    query = structured_query.invoke(prompt)
    state["query"] = query["query"]

    return state


from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


def execute_query(state: State) -> State:
    """Execute the query and return the result."""
    query_tool = QuerySQLDataBaseTool(db=db)
    result = query_tool.invoke(state["query"])
    state["result"] = result
    return state


def generate_answer(state: State) -> State:
    """Generate the answer from the result."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])


config = {"configurable": {"thread_id": "1"}}
for step in graph.stream(
    {"question": "How many employees are there?"},
    config,
    stream_mode="updates",
):
    print(step)
    try:
        user_approval = input("Do you want to go to execute query? (yes/no): ")
    except Exception:
        user_approval = "no"

    if user_approval.lower() == "yes":
        for step in graph.stream(None, config, stream_mode="updates"):
            print(step)
    else:
        print("Operation cancelled by user.")
