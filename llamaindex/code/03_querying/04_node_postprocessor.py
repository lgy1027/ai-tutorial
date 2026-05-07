import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.postprocessor import SimilarityPostprocessor

from common import build_example_index, configure_llamaindex, print_nodes


def main() -> None:
    """实验 4：使用 Node Postprocessor 过滤低分候选。"""
    configure_llamaindex()

    index = build_example_index()
    question = "LlamaIndex 的架构边界是什么？"

    retriever = index.as_retriever(similarity_top_k=5)
    retrieved_nodes = retriever.retrieve(question)

    print("=== Raw Retrieved Nodes ===")
    print_nodes(retrieved_nodes)

    processor = SimilarityPostprocessor(similarity_cutoff=0.5)
    filtered_nodes = processor.postprocess_nodes(retrieved_nodes, query_str=question)

    print("\n=== Filtered Nodes ===")
    print_nodes(filtered_nodes)


if __name__ == "__main__":
    main()
