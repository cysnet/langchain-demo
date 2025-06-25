import llm_set.llm_env as llm_env

from typing import Optional, Annotated, TypedDict


class Joke(TypedDict):
    setup: Annotated[str, "The setup of the joke"]
    punchline: Annotated[str, "The punchline of the joke"]
    rating: Annotated[Optional[int], "How funny the joke is, from 1 to 10"]


structured_llm = llm_env.llm.with_structured_output(Joke)
structured_response = structured_llm.invoke(
    "Tell me a joke about cats",
)

print(structured_response)
