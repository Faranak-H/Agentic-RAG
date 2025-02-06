import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from smolagents import OpenAIServerModel  # Assuming you're using this for local models

load_dotenv()

def get_model(model_id: str):
    """Returns a Hugging Face model via LangChain or a local OpenAI model."""
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

    if using_huggingface:
        return HuggingFaceHub(
            repo_id=model_id,  # Hugging Face model ID
            huggingfacehub_api_token=huggingface_api_token,
            model_kwargs={"temperature": 0.7, "max_length": 512}  # Adjust params as needed
        )
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )
