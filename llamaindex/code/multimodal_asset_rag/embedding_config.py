import os

from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from .config import load_env


def configure_embedding() -> str:
    load_env()
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()
    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL")

    if provider == "mock":
        Settings.embed_model = MockEmbedding(embed_dim=384)
        return "MockEmbedding(embed_dim=384)"

    if api_key and base_url and model:
        Settings.embed_model = OpenAIEmbedding(
            model_name=model,
            api_key=api_key,
            api_base=base_url,
        )
        return f"OpenAIEmbedding({model})"

    Settings.embed_model = MockEmbedding(embed_dim=384)
    return "MockEmbedding(embed_dim=384)"

