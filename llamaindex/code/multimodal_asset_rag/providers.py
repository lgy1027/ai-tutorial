import hashlib
import os
import time
from pathlib import Path

import requests
from PIL import Image


class EmbeddingProvider:
    def __init__(self) -> None:
        self.provider = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()
        self.api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = os.environ.get("EMBEDDING_MODEL")
        self.mock_dim = int(os.environ.get("MOCK_EMBEDDING_DIM", "384"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "mock" or not (self.api_key and self.base_url and self.model):
            return [self._mock_embedding(text) for text in texts]

        batch_size = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "5")))
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), batch_size):
            vectors.extend(self._embed_remote_batch(texts[offset : offset + batch_size]))
            interval = float(os.environ.get("EMBEDDING_REQUEST_INTERVAL", "0.25"))
            if interval > 0:
                time.sleep(interval)
        return vectors

    def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        retry_count = int(os.environ.get("EMBEDDING_RETRY_COUNT", "5"))
        timeout = float(os.environ.get("EMBEDDING_TIMEOUT", "120"))
        last_error: Exception | None = None
        for attempt in range(retry_count):
            response = requests.post(
                self.base_url.rstrip("/") + "/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": texts},
                timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min(5 * (attempt + 1), 30)
                time.sleep(wait_seconds)
                continue
            try:
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))
                continue
            data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
            return [[float(value) for value in item["embedding"]] for item in data]
        if last_error:
            raise last_error
        raise RuntimeError("Embedding request failed after retries, probably due to rate limit")

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _mock_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.mock_dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.mock_dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class ImageEmbeddingProvider:
    def __init__(self, provider: str = "lite") -> None:
        self.provider = provider
        self.model_name = os.environ.get("CLIP_MODEL", "clip-ViT-B-32")
        self._model = None

    def embed_image(self, image_path: Path) -> list[float]:
        if self.provider == "sentence_transformers":
            return self._embed_image_clip(image_path)
        return self._lite_image_feature(image_path)

    def embed_text(self, text: str) -> list[float]:
        if self.provider == "sentence_transformers":
            return self._embed_text_clip(text)
        from .image_index import tokenize

        vector = [0.0] * 384
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % len(vector)
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def _load_clip(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_image_clip(self, image_path: Path) -> list[float]:
        model = self._load_clip()
        image = Image.open(image_path).convert("RGB")
        return [float(value) for value in model.encode(image, normalize_embeddings=True).tolist()]

    def _embed_text_clip(self, text: str) -> list[float]:
        model = self._load_clip()
        return [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]

    def _lite_image_feature(self, image_path: Path) -> list[float]:
        image = Image.open(image_path).convert("RGB").resize((96, 96))
        bins = [0.0] * 24
        for red, green, blue in image.getdata():
            bins[red // 32] += 1
            bins[8 + green // 32] += 1
            bins[16 + blue // 32] += 1
        total = float(96 * 96)
        width, height = Image.open(image_path).size
        vector = [value / total for value in bins]
        vector.extend([width / max(height, 1), height / max(width, 1)])
        return vector
