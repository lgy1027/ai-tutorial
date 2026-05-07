import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "example_docs"
STORAGE_DIR = ROOT_DIR / "storage"


def configure_llamaindex() -> None:
    """从 .env 读取 OpenAI 兼容接口配置。"""
    load_dotenv(ROOT_DIR / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    base_url = os.environ.get("OPENAI_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL") or base_url

    missing = [
        name
        for name, value in {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "OPENAI_MODEL": model,
            "EMBEDDING_MODEL": embedding_model,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"请先在 llamaindex/.env 中配置: {', '.join(missing)}")

    Settings.llm = OpenAILike(
        model=model,
        api_key=api_key,
        api_base=base_url,
        temperature=0.2,
        is_chat_model=True,
        is_function_calling_model=True,
        system_prompt="请直接给出最终答案，不要输出思考过程，也不要输出 <think> 标签。",
    )
    Settings.embed_model = OpenAIEmbedding(
        model_name=embedding_model,
        api_key=api_key,
        api_base=embedding_base_url,
    )


def load_example_documents():
    """加载教程目录下的示例文档。"""
    return SimpleDirectoryReader(str(DATA_DIR)).load_data()


def build_example_index() -> VectorStoreIndex:
    """基于示例文档构建一个内存向量索引。"""
    documents = load_example_documents()
    return VectorStoreIndex.from_documents(documents)


def print_nodes(nodes) -> None:
    """打印检索命中的节点，便于观察 Retriever 行为。"""
    for index, node_with_score in enumerate(nodes, start=1):
        score = getattr(node_with_score, "score", None)
        text = node_with_score.node.get_content().replace("\n", " ")
        metadata = node_with_score.node.metadata
        print(f"\n[{index}] score={score}")
        print(f"metadata={metadata}")
        print(text[:220])
