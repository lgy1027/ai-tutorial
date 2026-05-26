import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from common import ROOT_DIR, configure_llamaindex, print_nodes


ADVANCED_DATA_DIR = ROOT_DIR / "data" / "advanced_docs"


def keyword_score(text: str, keywords: list[str]) -> int:
    """用关键词命中数模拟一个最小 rerank。"""
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def rerank_by_keywords(nodes, keywords: list[str]):
    """在候选 Node 内重排；前提是正确内容已经被召回。"""
    return sorted(
        nodes,
        key=lambda item: keyword_score(item.node.get_content(), keywords),
        reverse=True,
    )


def main() -> None:
    """进阶 1：演示 rerank 解决的是排序问题，不是召回问题。"""
    configure_llamaindex()

    documents = SimpleDirectoryReader(str(ADVANCED_DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)

    question = "E-503 和索引构建任务有什么关系？"
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(question)

    print("\n=== Before rerank ===")
    print_nodes(nodes)

    keywords = ["E-503", "索引", "构建", "任务"]
    reranked_nodes = rerank_by_keywords(nodes, keywords)

    print("\n=== After simple keyword rerank ===")
    print_nodes(reranked_nodes)


if __name__ == "__main__":
    main()
