import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Settings, VectorStoreIndex

from common import configure_llamaindex, load_example_documents


def main() -> None:
    """用例 3：观察 Settings 和 source_nodes，建立调试入口。"""
    configure_llamaindex()

    print("=== Settings ===")
    print(f"llm={type(Settings.llm).__name__}")
    print(f"embed_model={type(Settings.embed_model).__name__}")

    documents = load_example_documents()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    question = "生产环境的 RAG 应用需要关注哪些问题？"
    response = query_engine.query(question)

    print("\n=== Question ===")
    print(question)
    print("\n=== Response ===")
    print(response)

    print("\n=== Debug Source Nodes ===")
    for index, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node
        print(f"\n[{index}] score={source_node.score}")
        print(f"node_id={node.node_id}")
        print(f"ref_doc_id={node.ref_doc_id}")
        print(f"metadata={node.metadata}")
        print(node.get_content()[:260])


if __name__ == "__main__":
    main()
