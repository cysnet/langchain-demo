from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticToolsParser
import os
import sys

sys.path.append(os.getcwd())


class Add(BaseModel):
    a: int = Field(..., description="First number to add")
    b: int = Field(..., description="Second number to add")


class Multiply(BaseModel):
    a: int = Field(..., description="First number to multiply")
    b: int = Field(..., description="Second number to multiply")


tools = [Add, Multiply]

from llm_set import llm_env

llm = llm_env.llm

llm_with_tools = llm.bind_tools(tools)


query = "What is the sum of 5 and 10?"

chain = llm_with_tools | PydanticToolsParser(tools=[Add, Multiply])
response = chain.invoke(query)

print(response)
