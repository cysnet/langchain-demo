import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"


from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)


import bs4

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing import List, TypedDict, Literal, Annotated

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(docs)

total_documents = len(all_splits)
third = total_documents // 3
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

for a in all_splits:
    print(a.metadata)


_ = vector_store.add_documents(all_splits)


class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    """State for the agent."""

    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State) -> State:
    structured_query = llm.with_structured_output(Search)
    query = structured_query.invoke(state["question"])
    state["query"] = query
    return state


def retrieve(state: State) -> State:
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    state["context"] = retrieved_docs
    return state


def generate(state: State) -> State:
    docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
    message = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(message)
    state["answer"] = response.content
    return state


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")
