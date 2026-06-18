import json
from dataclasses import dataclass
from pathlib import Path

from .config import ASSET_DIR, MANIFEST_PATH


@dataclass(frozen=True)
class Asset:
    asset_id: str
    title: str
    source_type: str
    relative_path: str
    source_url: str
    tags: list[str]

    @property
    def file_path(self) -> Path:
        return ASSET_DIR / self.relative_path


def load_assets(limit: int = 0) -> list[Asset]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = [
        Asset(
            asset_id=str(item["id"]),
            title=str(item["title"]),
            source_type=str(item["type"]),
            relative_path=str(item["path"]),
            source_url=str(item.get("source_url", "")),
            tags=[str(tag) for tag in item.get("tags", [])],
        )
        for item in payload["records"]
    ]
    if limit > 0:
        return assets[:limit]
    return assets

