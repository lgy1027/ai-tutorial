import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.node_parser import SentenceSplitter

from common import configure_llamaindex, load_example_documents


def main() -> None:
    """用例 2：单独观察 Document 到 Node 的转换。"""
    configure_llamaindex()

    documents = load_example_documents()
    splitter = SentenceSplitter(chunk_size=160, chunk_overlap=30)
    nodes = splitter.get_nodes_from_documents(documents)

    print("=== Documents ===")
    for document in documents:
        print(f"doc_id={document.doc_id}")
        print(f"metadata={document.metadata}")
        print(document.text[:120].replace("\n", " "))

    print("\n=== Nodes ===")
    for index, node in enumerate(nodes, start=1):
        print(f"\n--- Node {index} ---")
        print(f"node_id={node.node_id}")
        print(f"ref_doc_id={node.ref_doc_id}")
        print(f"metadata={node.metadata}")
        print(node.get_content()[:220])


if __name__ == "__main__":
    main()
