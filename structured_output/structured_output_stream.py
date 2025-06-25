import llm_set.llm_env as llm_env

from typing import Annotated, TypedDict, Optional


class Joke(TypedDict):
    setup: Annotated[str, "the setup of the joke"]
    punchline: Annotated[str, "the punchline of the joke"]
    rating: Annotated[Optional[int], "the rating of the joke, from 0 to 10"]


structured_llm = llm_env.llm.with_structured_output(Joke)


for chunk in structured_llm.stream("Tell me a joke about dog"):
    print(chunk)
