import llm_set.llm_env as llm_env

json_schema = {
    "title": "Joke",
    "type": "object",
    "description": "Joke to tell user.",
    "properties": {
        "setup": {"type": "string", "description": "The setup of the joke"},
        "punchline": {"type": "string", "description": "The punchline of the joke"},
        "rating": {
            "type": ["integer"],
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}

structured_llm = llm_env.llm.with_structured_output(json_schema)
structured_response = structured_llm.invoke(
    "Tell me a joke about cats",
)

print(structured_response)
