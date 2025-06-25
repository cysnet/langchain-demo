import os
import sys

sys.path.append(os.getcwd())

from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from llm_set import llm_env


llm = llm_env.llm

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")

docs = loader.load()


prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following: \\n\\n{context}")]
)


chain = create_stuff_documents_chain(llm, prompt)

result = chain.invoke({"context": docs})
print(result)
