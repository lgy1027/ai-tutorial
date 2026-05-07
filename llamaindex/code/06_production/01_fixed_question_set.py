import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_example_index, configure_llamaindex, print_nodes


QUESTIONS = [
    "LlamaIndex 在 RAG 系统里负责什么？",
    "为什么 ingestion 要单独设计？",
    "Retriever 和 QueryEngine 的区别是什么？",
]


def main() -> None:
    """实验 1：固定问题集，记录回答和来源上下文。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)

    for question in QUESTIONS:
        print(f"\n\n=== Question ===\n{question}")
        response = query_engine.query(question)
        print("\n=== Response ===")
        print(response)
        print("\n=== Source Nodes ===")
        print_nodes(response.source_nodes)


if __name__ == "__main__":
    main()
