import os
from langchain_community.utilities import SQLDatabase


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

docs = loader.load()

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write a concise summary of the following:\\n\\n{context}"),
    ]
)

chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

# result = chain.invoke({"context": docs})
# print(result)

for token in chain.stream({"context": docs}):
    print(token, end="|", flush=True)
