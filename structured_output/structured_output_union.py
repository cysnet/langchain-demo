import llm_set.llm_env as llm_env


from typing import Union
from pydantic import BaseModel, Field


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Union[int, None] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    response: str = Field(description="A conversational response to a user query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]


structured_llm = llm_env.llm.with_structured_output(FinalResponse)


print(structured_llm.invoke("Tell me a joke about cat"))


print(structured_llm.invoke("How are you?"))
