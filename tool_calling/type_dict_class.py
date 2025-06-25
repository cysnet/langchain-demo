import os
import sys
from typing_extensions import Annotated, TypedDict


sys.path.append(os.getcwd())


class add(TypedDict):
    """add two integres"""

    a: Annotated[int, "the first number to add"]
    b: Annotated[int, "the second number to add"]


class multiply(TypedDict):
    """multiply two integers"""

    a: Annotated[int, "the first number to multiply"]
    b: Annotated[int, "the second number to multiply"]


tools = [add, multiply]


from llm_set import llm_env


llm = llm_env.llm

llm_with_tools = llm.bind_tools(tools)


query = "What is 2 + 3 and 4 * 5?"
response = llm_with_tools.invoke(query)
print(response)
