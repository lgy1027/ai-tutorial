import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_example_index, configure_llamaindex, print_nodes


def main() -> None:
    """实验 2：绕过 QueryEngine，单独观察 Retriever。"""
    configure_llamaindex()

    index = build_example_index()
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("RAG 质量下降时应该先排查什么？")

    print_nodes(nodes)


if __name__ == "__main__":
    main()
