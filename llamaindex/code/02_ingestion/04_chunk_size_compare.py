import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from common import configure_llamaindex, load_example_documents, print_nodes


def build_nodes(chunk_size: int, chunk_overlap: int):
    documents = load_example_documents()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        ]
    )
    return pipeline.run(documents=documents)


def main() -> None:
    """示例 4：对比不同 chunk 参数对 Node 数量和检索结果的影响。"""
    configure_llamaindex()

    question = "LlamaIndex 的核心组件和生产化关注点是什么？"
    configs = [
        {"chunk_size": 80, "chunk_overlap": 10},
        {"chunk_size": 180, "chunk_overlap": 30},
    ]

    for config in configs:
        print(f"\n=== chunk_size={config['chunk_size']} overlap={config['chunk_overlap']} ===")
        nodes = build_nodes(**config)
        print(f"nodes={len(nodes)}")

        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes_with_score = retriever.retrieve(question)
        print_nodes(nodes_with_score)


if __name__ == "__main__":
    main()
