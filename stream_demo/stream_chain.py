import os
import sys

sys.path.append(os.getcwd())


from llm_set import llm_env


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | llm_env.llm | parser

for chunk in chain.stream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)
