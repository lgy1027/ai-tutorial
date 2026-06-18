from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answer import answer_question
from .assets import load_assets
from .cli import command_index, command_parse
from .config import DOCUMENTS_JSONL, IMAGE_INDEX_PATH, TEXT_INDEX_DIR, ensure_workspace, load_env
from .evaluation import run_eval
from .qdrant_store import qdrant_image_to_image_search, qdrant_text_search, qdrant_text_to_image_search
from .retrieval import hybrid_search


app = FastAPI(title="Multimodal Asset RAG", version="0.1.0")


class IngestRequest(BaseModel):
    limit: int = 0
    pdf_parser: str = Field(default="auto", pattern="^(auto|pymupdf|paddleocr_vl)$")
    ocr: bool = False
    vlm: bool = False
    image_provider: str = Field(default="lite", pattern="^(lite|sentence_transformers)$")


class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(text|text-to-image|image-to-image|hybrid)$")
    image_path: str | None = None
    top_k: int = 5
    image_provider: str = Field(default="lite", pattern="^(lite|sentence_transformers)$")


class AnswerRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, object]:
    load_env()
    return {
        "status": "ok",
        "assets": len(load_assets()),
        "documents_jsonl_exists": DOCUMENTS_JSONL.exists(),
        "text_index_exists": TEXT_INDEX_DIR.exists(),
        "image_index_exists": IMAGE_INDEX_PATH.exists(),
        "vector_backend": "qdrant",
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    import argparse

    ensure_workspace()
    command_parse(
        argparse.Namespace(
            limit=request.limit,
            pdf_parser=request.pdf_parser,
            ocr=request.ocr,
            vlm=request.vlm,
        )
    )
    command_index(
        argparse.Namespace(
            backend="qdrant",
            image_provider=request.image_provider,
        )
    )
    return {
        "status": "ok",
        "documents_jsonl": str(DOCUMENTS_JSONL),
        "backend": "qdrant",
        "image_provider": request.image_provider,
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    if request.mode == "text":
        hits = qdrant_text_search(request.query, top_k=request.top_k)
    elif request.mode == "text-to-image":
        hits = qdrant_text_to_image_search(
            request.query, top_k=request.top_k, provider_name=request.image_provider
        )
    elif request.mode == "image-to-image":
        if not request.image_path:
            raise ValueError("image_path is required for image-to-image search")
        hits = qdrant_image_to_image_search(
            Path(request.image_path), top_k=request.top_k, provider_name=request.image_provider
        )
    else:
        hits = hybrid_search(
            request.query,
            image_path=Path(request.image_path) if request.image_path else None,
            top_k=request.top_k,
            backend="qdrant",
            image_provider=request.image_provider,
        )
    return {"query": request.query, "hits": [hit.__dict__ for hit in hits]}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    return answer_question(request.question, top_k=request.top_k, backend="qdrant")


@app.post("/eval")
def eval_endpoint() -> dict[str, object]:
    results = run_eval(backend="qdrant")
    return {"results": [result.__dict__ for result in results]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8011,
    )

