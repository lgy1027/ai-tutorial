import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import VectorStoreIndex

from common import configure_llamaindex, load_example_documents


def main() -> None:
    """用例 1：用最短路径跑通本地文档问答。"""
    configure_llamaindex()

    documents = load_example_documents()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    response = query_engine.query("LlamaIndex 在 RAG 系统中负责什么？")

    print("=== Response ===")
    print(response)

    print("\n=== Source Nodes ===")
    for item in response.source_nodes:
        print(f"score={item.score}")
        print(item.node.get_content()[:200])


if __name__ == "__main__":
    main()
