import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever

from common import ROOT_DIR, Settings, configure_llamaindex, print_nodes


ADVANCED_DATA_DIR = ROOT_DIR / "data" / "advanced_docs"


def main() -> None:
    """进阶 1：用 QueryFusionRetriever 做多路召回合并。"""
    configure_llamaindex()

    documents = SimpleDirectoryReader(str(ADVANCED_DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)

    base_retriever = index.as_retriever(similarity_top_k=3)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[base_retriever],
        llm=Settings.llm,
        similarity_top_k=3,
        num_queries=3,
        use_async=False,
        verbose=True,
    )

    nodes = fusion_retriever.retrieve("故障码 E-503 和索引任务有什么关系？")
    print_nodes(nodes)


if __name__ == "__main__":
    main()
