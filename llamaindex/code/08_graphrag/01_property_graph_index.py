import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader

from common import ROOT_DIR, configure_llamaindex, print_nodes


GRAPH_DATA_DIR = ROOT_DIR / "data" / "advanced_docs"
QUESTION = "Agent 工作台依赖哪些服务？"


def build_relation_graph_index() -> PropertyGraphIndex:
    """从关系型文档构建最小图索引。"""
    documents = SimpleDirectoryReader(
        input_files=[str(GRAPH_DATA_DIR / "team_graph.md")]
    ).load_data()
    return PropertyGraphIndex.from_documents(documents, show_progress=True)


def inspect_graph_retrieval(index: PropertyGraphIndex, question: str) -> None:
    """观察图索引返回的关系上下文。"""
    retriever = index.as_retriever(include_text=True, similarity_top_k=3)
    nodes = retriever.retrieve(question)
    print_nodes(nodes)


def main() -> None:
    """进阶 2：用 PropertyGraphIndex 观察实体关系检索。"""
    configure_llamaindex()

    index = build_relation_graph_index()
    inspect_graph_retrieval(index, QUESTION)


if __name__ == "__main__":
    main()
