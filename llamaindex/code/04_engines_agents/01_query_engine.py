import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_example_index, configure_llamaindex


def main() -> None:
    """实验 1：QueryEngine 适合稳定的单轮知识问答。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)

    response = query_engine.query("LlamaIndex 在 RAG 系统中主要负责什么？")
    print(response)
    print("\nsource_nodes:", len(response.source_nodes))


if __name__ == "__main__":
    main()
