import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from .assets import ASSET_DIR
from .config import INDEX_DIR
from .document_store import read_documents
from .providers import EmbeddingProvider, ImageEmbeddingProvider
from .schema import SearchHit
from .image_index import text_score


TEXT_COLLECTION_BASE = os.environ.get("QDRANT_TEXT_COLLECTION", "multimodal_text")
IMAGE_COLLECTION_BASE = os.environ.get("QDRANT_IMAGE_COLLECTION", "multimodal_image")


def text_collection(vector_size: int | None = None) -> str:
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_TEXT_COLLECTION", TEXT_COLLECTION_BASE)
    name = f"{TEXT_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_TEXT_COLLECTION"] = name
    return name


def image_collection(vector_size: int | None = None) -> str:
    if vector_size is None:
        return os.environ.get("QDRANT_ACTIVE_IMAGE_COLLECTION", IMAGE_COLLECTION_BASE)
    name = f"{IMAGE_COLLECTION_BASE}_{vector_size}d"
    os.environ["QDRANT_ACTIVE_IMAGE_COLLECTION"] = name
    return name


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=str(INDEX_DIR / "qdrant"))


def recreate_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def build_qdrant_text_index(batch_size: int | None = None) -> tuple[int, str]:
    documents = read_documents()
    embedder = EmbeddingProvider()
    batch_size = batch_size or max(1, int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "16")))
    first_vector = embedder.embed_text(documents[0].text)
    client = get_qdrant_client()
    collection_name = text_collection(len(first_vector))
    recreate_collection(client, collection_name, len(first_vector))

    points: list[models.PointStruct] = []
    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        texts = [document.text for document in batch]
        vectors = [first_vector] + embedder.embed_texts(texts[1:]) if offset == 0 else embedder.embed_texts(texts)
        for index, (document, vector) in enumerate(zip(batch, vectors), start=offset):
            payload = {**document.metadata, "text": document.text}
            point_id = stable_point_id(f"text:{document.metadata.get('asset_id')}:{document.metadata.get('page')}:{index}")
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(documents), f"qdrant:{collection_name}"


def build_qdrant_image_index(provider_name: str = "lite") -> tuple[int, str]:
    documents = read_documents()
    image_documents = [document for document in documents if document.metadata.get("source_type") == "image"]
    provider = ImageEmbeddingProvider(provider=provider_name)
    first_path = ASSET_DIR / str(image_documents[0].metadata["source_path"])
    first_vector = provider.embed_image(first_path)
    client = get_qdrant_client()
    collection_name = image_collection(len(first_vector))
    recreate_collection(client, collection_name, len(first_vector))

    points = []
    for index, document in enumerate(image_documents):
        image_path = ASSET_DIR / str(document.metadata["source_path"])
        vector = first_vector if index == 0 else provider.embed_image(image_path)
        payload = {**document.metadata, "text": document.text, "image_provider": provider_name}
        point_id = stable_point_id(f"image:{document.metadata.get('asset_id')}")
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points), f"qdrant:{collection_name}:{provider_name}"


def qdrant_text_search(query: str, top_k: int = 5) -> list[SearchHit]:
    embedder = EmbeddingProvider()
    client = get_qdrant_client()
    query_vector = embedder.embed_text(query)
    results = client.query_points(
        collection_name=text_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_text", point) for point in results]


def qdrant_text_to_image_search(query: str, top_k: int = 5, provider_name: str = "lite") -> list[SearchHit]:
    provider = ImageEmbeddingProvider(provider=provider_name)
    client = get_qdrant_client()
    if provider_name == "lite":
        collection_name = image_collection(26)
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        hits = []
        for point in records:
            payload = point.payload or {}
            score = text_score(query, "\n".join([str(payload.get("asset_title", "")), str(payload.get("text", ""))]))
            hits.append(_payload_to_hit("qdrant_payload_text_to_image", score, payload))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    query_vector = provider.embed_text(query)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_text_to_image", point) for point in results]


def qdrant_image_to_image_search(image_path: Path, top_k: int = 5, provider_name: str = "lite") -> list[SearchHit]:
    provider = ImageEmbeddingProvider(provider=provider_name)
    client = get_qdrant_client()
    query_vector = provider.embed_image(image_path)
    results = client.query_points(
        collection_name=image_collection(len(query_vector)),
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [_point_to_hit("qdrant_image_to_image", point) for point in results]


def _point_to_hit(route: str, point) -> SearchHit:
    payload = point.payload or {}
    return _payload_to_hit(route, float(point.score or 0.0), payload)


def _payload_to_hit(route: str, score: float, payload: dict[str, object]) -> SearchHit:
    return SearchHit(
        route=route,
        score=score,
        asset_id=str(payload.get("asset_id", "")),
        title=str(payload.get("asset_title") or payload.get("title") or ""),
        source_type=str(payload.get("source_type", "")),
        source_path=str(payload.get("source_path", "")),
        evidence=str(payload.get("text", ""))[:1000],
        metadata=dict(payload),
    )
