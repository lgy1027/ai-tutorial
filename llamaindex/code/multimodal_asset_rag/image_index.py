import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from .assets import Asset, load_assets
from .config import ASSET_DIR, IMAGE_INDEX_PATH
from .document_store import read_documents
from .schema import SearchHit


def tokenize(text: str) -> list[str]:
    english = re.findall(r"[a-zA-Z0-9_.:-]+", text.lower())
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    chinese_terms = []
    for chunk in chinese_chunks:
        chinese_terms.append(chunk)
        chinese_terms.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        chinese_terms.extend(chunk[index : index + 3] for index in range(len(chunk) - 2))
    return english + chinese_terms


def text_score(query: str, text: str) -> float:
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0
    doc_terms = Counter(tokenize(text))
    hits = sum(1 for term in query_terms if doc_terms[term] > 0)
    fuzzy_hits = sum(
        1
        for term in query_terms
        if len(term) >= 2 and any(term in doc_term or doc_term in term for doc_term in doc_terms)
    )
    return (hits + 0.5 * fuzzy_hits) / max(len(query_terms), 1)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def lite_image_feature(path: Path) -> list[float]:
    image = Image.open(path).convert("RGB").resize((96, 96))
    bins = [0.0] * 24
    for red, green, blue in image.getdata():
        bins[red // 32] += 1
        bins[8 + green // 32] += 1
        bins[16 + blue // 32] += 1
    total = float(96 * 96)
    vector = [value / total for value in bins]
    width, height = Image.open(path).size
    vector.extend([width / max(height, 1), height / max(width, 1)])
    return vector


def sentence_transformers_clip_records(image_assets: list[Asset], text_by_asset: dict[str, str]) -> list[dict[str, object]]:
    from PIL import Image as PILImage
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("CLIP_MODEL", "clip-ViT-B-32")
    model = SentenceTransformer(model_name)
    records = []
    for asset in image_assets:
        image = PILImage.open(asset.file_path).convert("RGB")
        vector = model.encode(image, normalize_embeddings=True).tolist()
        records.append(
            {
                "asset_id": asset.asset_id,
                "title": asset.title,
                "source_type": asset.source_type,
                "source_path": asset.relative_path,
                "source_url": asset.source_url,
                "tags": asset.tags,
                "search_text": text_by_asset.get(asset.asset_id, ""),
                "image_vector": vector,
                "provider": f"sentence-transformers:{model_name}",
                "clip_model": model_name,
            }
        )
    return records


def build_image_index(provider: str = "lite") -> tuple[int, str]:
    image_assets = [asset for asset in load_assets() if asset.source_type == "image"]
    documents = read_documents()
    text_by_asset: dict[str, str] = {}
    for document in documents:
        asset_id = str(document.metadata.get("asset_id", ""))
        text_by_asset.setdefault(asset_id, "")
        text_by_asset[asset_id] += "\n" + document.text

    if provider == "sentence_transformers":
        records = sentence_transformers_clip_records(image_assets, text_by_asset)
    else:
        records = []
        for asset in image_assets:
            records.append(
                {
                    "asset_id": asset.asset_id,
                    "title": asset.title,
                    "source_type": asset.source_type,
                    "source_path": asset.relative_path,
                    "source_url": asset.source_url,
                    "tags": asset.tags,
                    "search_text": text_by_asset.get(asset.asset_id, ""),
                    "image_vector": lite_image_feature(asset.file_path),
                    "provider": "lite-color-histogram",
                }
            )
    IMAGE_INDEX_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records), str(records[0]["provider"] if records else provider)


def load_image_index() -> list[dict[str, object]]:
    if not IMAGE_INDEX_PATH.exists():
        raise RuntimeError(f"Image index not found: {IMAGE_INDEX_PATH}")
    return json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))


def search_images_by_text(query: str, top_k: int = 5) -> list[SearchHit]:
    records = load_image_index()
    if records and str(records[0].get("provider", "")).startswith("sentence-transformers:"):
        from sentence_transformers import SentenceTransformer

        model_name = str(records[0].get("clip_model") or os.environ.get("CLIP_MODEL", "clip-ViT-B-32"))
        model = SentenceTransformer(model_name)
        query_vector = model.encode(query, normalize_embeddings=True).tolist()
        hits = [
            SearchHit(
                route="clip_text_to_image",
                score=cosine(query_vector, [float(value) for value in record["image_vector"]]),
                asset_id=str(record["asset_id"]),
                title=str(record["title"]),
                source_type=str(record["source_type"]),
                source_path=str(record["source_path"]),
                evidence=str(record.get("search_text", ""))[:500],
                metadata=record,
            )
            for record in records
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    hits = []
    for record in records:
        weighted_text = "\n".join(
            [
                str(record["title"]),
                str(record["title"]),
                str(record.get("tags", "")),
                str(record.get("search_text", "")),
            ]
        )
        hits.append(
            SearchHit(
                route="text_to_image",
                score=text_score(query, weighted_text),
                asset_id=str(record["asset_id"]),
                title=str(record["title"]),
                source_type=str(record["source_type"]),
                source_path=str(record["source_path"]),
                evidence=str(record.get("search_text", ""))[:500],
                metadata=record,
            )
        )
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def search_images_by_image(image_path: Path, top_k: int = 5) -> list[SearchHit]:
    records = load_image_index()
    if records and str(records[0].get("provider", "")).startswith("sentence-transformers:"):
        from PIL import Image as PILImage
        from sentence_transformers import SentenceTransformer

        model_name = str(records[0].get("clip_model") or os.environ.get("CLIP_MODEL", "clip-ViT-B-32"))
        model = SentenceTransformer(model_name)
        image = PILImage.open(image_path).convert("RGB")
        query_vector = model.encode(image, normalize_embeddings=True).tolist()
        hits = [
            SearchHit(
                route="clip_image_to_image",
                score=cosine(query_vector, [float(value) for value in record["image_vector"]]),
                asset_id=str(record["asset_id"]),
                title=str(record["title"]),
                source_type=str(record["source_type"]),
                source_path=str(record["source_path"]),
                evidence=str(record.get("search_text", ""))[:300],
                metadata=record,
            )
            for record in records
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    query_vector = lite_image_feature(image_path)
    hits = []
    for record in records:
        hits.append(
            SearchHit(
                route="image_to_image",
                score=cosine(query_vector, [float(value) for value in record["image_vector"]]),
                asset_id=str(record["asset_id"]),
                title=str(record["title"]),
                source_type=str(record["source_type"]),
                source_path=str(record["source_path"]),
                evidence=str(record.get("search_text", ""))[:300],
                metadata=record,
            )
        )
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def asset_path_from_source(source_path: str) -> Path:
    return ASSET_DIR / source_path
