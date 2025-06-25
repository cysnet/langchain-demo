import os
import sys

sys.path.append(os.getcwd())

from llm_set import llm_env


chunks = []

for chunk in llm_env.llm.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

print(chunks[0])
