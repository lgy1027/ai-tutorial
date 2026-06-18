import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "data" / "media_project"
ASSET_DIR = PROJECT_DIR / "chapter11_assets"
MANIFEST_PATH = ASSET_DIR / "asset_manifest.json"
WORK_DIR = PROJECT_DIR / "multimodal_asset_rag"
PARSED_DIR = WORK_DIR / "parsed"
CAPTION_DIR = WORK_DIR / "captions"
INDEX_DIR = WORK_DIR / "indexes"
TEXT_INDEX_DIR = INDEX_DIR / "text"
IMAGE_INDEX_PATH = INDEX_DIR / "image_index.json"
DOCUMENTS_JSONL = WORK_DIR / "documents.jsonl"
RUN_REPORT_PATH = WORK_DIR / "run_report.json"


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_workspace() -> None:
    for path in [WORK_DIR, PARSED_DIR, CAPTION_DIR, INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)

