import os


os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "default"
# os.environ["OPENAI_API_KEY"] = ""
# os.environ["TAVILY_API_KEY"] = ""
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
