import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from common import configure_llamaindex, load_example_documents, print_nodes


def main() -> None:
    """示例 2：为 Document 注入业务元数据，并用 metadata filter 检索。"""
    configure_llamaindex()

    documents = load_example_documents()
    for document in documents:
        file_name = document.metadata.get("file_name", "")
        document.metadata["course"] = "llamaindex"
        document.metadata["source_type"] = "local_markdown"
        document.metadata["document_name"] = file_name
        document.metadata["tenant_id"] = "demo_tenant"

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=140, chunk_overlap=20),
        ]
    )
    nodes = pipeline.run(documents=documents)

    index = VectorStoreIndex(nodes)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="course", value="llamaindex"),
            MetadataFilter(key="tenant_id", value="demo_tenant"),
        ]
    )
    retriever = index.as_retriever(similarity_top_k=3, filters=filters)
    nodes_with_score = retriever.retrieve("RAG 系统生产化要关注什么？")

    print_nodes(nodes_with_score)


if __name__ == "__main__":
    main()
