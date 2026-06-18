from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage

from .config import TEXT_INDEX_DIR
from .document_store import read_documents
from .embedding_config import configure_embedding
from .schema import SearchHit


def build_text_index() -> tuple[int, str]:
    embedding_name = configure_embedding()
    parsed_documents = read_documents()
    documents = [
        Document(text=document.text, metadata=document.metadata)
        for document in parsed_documents
        if document.text.strip()
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(TEXT_INDEX_DIR))
    return len(documents), embedding_name


def load_text_index():
    configure_embedding()
    storage_context = StorageContext.from_defaults(persist_dir=str(TEXT_INDEX_DIR))
    return load_index_from_storage(storage_context)


def search_text(query: str, top_k: int = 5) -> list[SearchHit]:
    index = load_text_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    hits = []
    for node_with_score in retriever.retrieve(query):
        node = node_with_score.node
        hits.append(
            SearchHit(
                route="text",
                score=float(node_with_score.score or 0.0),
                asset_id=str(node.metadata.get("asset_id", "")),
                title=str(node.metadata.get("asset_title", "")),
                source_type=str(node.metadata.get("source_type", "")),
                source_path=str(node.metadata.get("source_path", "")),
                evidence=node.get_content()[:800],
                metadata=dict(node.metadata),
            )
        )
    return hits

