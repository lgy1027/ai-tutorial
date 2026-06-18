from pathlib import Path

from .image_index import search_images_by_image, search_images_by_text
from .qdrant_store import (
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .schema import SearchHit
from .text_index import search_text


def normalize_scores(hits: list[SearchHit]) -> list[SearchHit]:
    if not hits:
        return []
    max_score = max(hit.score for hit in hits) or 1.0
    for hit in hits:
        hit.score = hit.score / max_score
    return hits


def merge_hits(groups: list[list[SearchHit]], weights: list[float], top_k: int) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    for group, weight in zip(groups, weights):
        for hit in normalize_scores(group):
            if hit.score <= 0:
                continue
            key = hit.asset_id
            weighted_score = hit.score * weight
            if key not in merged:
                hit.score = weighted_score
                hit.metadata = {**hit.metadata, "routes": [hit.route]}
                merged[key] = hit
            else:
                current = merged[key]
                current.score += weighted_score
                routes = list(current.metadata.get("routes", []))
                routes.append(hit.route)
                current.metadata["routes"] = sorted(set(routes))
                if len(hit.evidence) > len(current.evidence):
                    current.evidence = hit.evidence
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:top_k]


def hybrid_search(
    query: str,
    image_path: Path | None = None,
    top_k: int = 5,
    backend: str = "qdrant",
    image_provider: str = "lite",
) -> list[SearchHit]:
    if backend == "qdrant":
        groups = [
            qdrant_text_search(query, top_k=top_k),
            qdrant_text_to_image_search(query, top_k=top_k, provider_name=image_provider),
        ]
    else:
        groups = [
            search_text(query, top_k=top_k),
            search_images_by_text(query, top_k=top_k),
        ]
    weights = [0.55, 0.30]
    if image_path:
        if backend == "qdrant":
            groups.append(qdrant_image_to_image_search(image_path, top_k=top_k, provider_name=image_provider))
        else:
            groups.append(search_images_by_image(image_path, top_k=top_k))
        weights.append(0.15)
    return merge_hits(groups, weights, top_k=top_k)
