import sys
import os

sys.path.append(os.getcwd())


import llm_set.llm_env as llm_env


from typing import Annotated, TypedDict, Optional


class Joke(TypedDict):
    setup: Annotated[str, "the setup of the joke"]
    punchline: Annotated[str, "the punchline of the joke"]
    rating: Annotated[Optional[int], "the rating of the joke, from 0 to 10"]


structured_llm = llm_env.llm.with_structured_output(Joke)


from langchain_core.prompts import ChatPromptTemplate

system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""


prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

few_shot_llm = prompt | structured_llm

print(few_shot_llm.invoke("Tell me a joke about The UK"))
