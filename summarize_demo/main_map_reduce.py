import os
import sys

sys.path.append(os.getcwd())

from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import operator
from typing import Annotated, List, Literal, TypedDict
from langchain.chains.combine_documents.reduce import collapse_docs, split_list_of_docs
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from llm_set import llm_env


llm = llm_env.llm

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")

docs = loader.load()


map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following: \\n\\n{context}")]
)


reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

reduce_prompt = ChatPromptTemplate([("human", reduce_template)])


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks")


token_max = 1000


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents"""
    return sum(llm.get_num_tokens(d.page_content) for d in documents)


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


def generate_summary(state: SummaryState):
    prompt = map_prompt.invoke(state["content"])
    response = llm.invoke(prompt)
    return {"summaries": [response.content]}


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = llm.invoke(prompt)
    return response.content


def collapse_summaries(state: OverallState):
    docs_lists = split_list_of_docs(
        state["collapsed_summaries"],
        length_function,
        token_max,
    )

    results = []
    for doc_list in docs_lists:
        combined = collapse_docs(doc_list, _reduce)
        results.append(combined)

    return {"collapsed_summaries": results}


def should_collapse(state: OverallState):
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


def generate_final_summary(state: OverallState):
    response = _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


graph = StateGraph(OverallState)

graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()


for step in app.stream(
    {"contents": [doc.page_content for doc in split_docs]},
    {"recursion_limit": 10},
):
    print(list(step.keys()))
