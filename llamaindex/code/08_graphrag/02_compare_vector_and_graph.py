import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, VectorStoreIndex

from common import ROOT_DIR, configure_llamaindex, print_nodes


GRAPH_DATA_DIR = ROOT_DIR / "data" / "advanced_docs"


def main() -> None:
    """进阶 2：同一个问题，对比向量索引和图索引返回结果。"""
    configure_llamaindex()

    documents = SimpleDirectoryReader(
        input_files=[str(GRAPH_DATA_DIR / "team_graph.md")]
    ).load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    graph_index = PropertyGraphIndex.from_documents(documents, show_progress=False)

    question = "张明维护哪些能力？这些能力和哪个服务有关？"

    print("\n=== Vector Retriever ===")
    vector_nodes = vector_index.as_retriever(similarity_top_k=3).retrieve(question)
    print_nodes(vector_nodes)

    print("\n=== Property Graph Retriever ===")
    graph_nodes = graph_index.as_retriever(include_text=True, similarity_top_k=3).retrieve(
        question
    )
    print_nodes(graph_nodes)


if __name__ == "__main__":
    main()
