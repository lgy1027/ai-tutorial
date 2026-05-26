import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from common import ROOT_DIR, configure_llamaindex, print_nodes


ADVANCED_DATA_DIR = ROOT_DIR / "data" / "advanced_docs"
QUESTION = "E-503 故障码通常说明什么？"


def build_diagnostic_index() -> VectorStoreIndex:
    """构建诊断器使用的最小知识库索引。"""
    documents = SimpleDirectoryReader(str(ADVANCED_DATA_DIR)).load_data()
    return VectorStoreIndex.from_documents(documents)


def inspect_retrieval(index: VectorStoreIndex, question: str, top_k: int) -> None:
    """只检查召回结果，不生成最终回答。"""
    print(f"\n=== top_k={top_k} ===")
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    print_nodes(nodes)


def main() -> None:
    """进阶 1：RAG 检索诊断器的第一步，观察 top_k 对召回的影响。"""
    configure_llamaindex()

    index = build_diagnostic_index()

    for top_k in [1, 3, 5]:
        inspect_retrieval(index, QUESTION, top_k)


if __name__ == "__main__":
    main()
