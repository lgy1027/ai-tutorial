import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from common import configure_llamaindex, load_example_documents


def main() -> None:
    """示例 1：使用 IngestionPipeline 显式切分文档。"""
    configure_llamaindex()

    documents = load_example_documents()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=120, chunk_overlap=20),
        ]
    )
    nodes = pipeline.run(documents=documents)

    print(f"documents={len(documents)}")
    print(f"nodes={len(nodes)}")
    for index, node in enumerate(nodes, start=1):
        print(f"\n--- Node {index} ---")
        print(node.get_content()[:220])


if __name__ == "__main__":
    main()
