import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .answer import answer_json
from .assets import load_assets
from .config import DOCUMENTS_JSONL, IMAGE_INDEX_PATH, TEXT_INDEX_DIR, ensure_workspace, load_env
from .document_store import write_documents
from .evaluation import run_eval, write_eval_report
from .image_index import build_image_index, search_images_by_image, search_images_by_text
from .image_parser import parse_image
from .pdf_parser import parse_pdf
from .qdrant_store import (
    build_qdrant_image_index,
    build_qdrant_text_index,
    qdrant_image_to_image_search,
    qdrant_text_search,
    qdrant_text_to_image_search,
)
from .retrieval import hybrid_search
from .text_index import build_text_index, search_text


def command_parse(args: argparse.Namespace) -> None:
    load_env()
    ensure_workspace()
    assets = load_assets(limit=args.limit)
    documents = []
    for asset in assets:
        if asset.source_type == "pdf":
            parsed = parse_pdf(asset, parser=args.pdf_parser)
        elif asset.source_type == "image":
            parsed = parse_image(asset, enable_ocr=args.ocr, enable_vlm=args.vlm)
        else:
            parsed = []
        documents.extend(parsed)
        print(f"parsed asset={asset.asset_id} type={asset.source_type} documents={len(parsed)}")
    write_documents(documents)
    print(f"documents={len(documents)}")
    print(f"documents_jsonl={DOCUMENTS_JSONL}")


def command_index(args: argparse.Namespace) -> None:
    load_env()
    ensure_workspace()
    if args.backend == "qdrant":
        text_count, embedding_name = build_qdrant_text_index()
        image_count, image_provider = build_qdrant_image_index(provider_name=args.image_provider)
    else:
        text_count, embedding_name = build_text_index()
        image_count, image_provider = build_image_index(provider=args.image_provider)
    print(f"text_documents={text_count}")
    print(f"embedding={embedding_name}")
    print(f"text_index={TEXT_INDEX_DIR}")
    print(f"image_records={image_count}")
    print(f"image_provider={image_provider}")
    print(f"image_index={IMAGE_INDEX_PATH}")


def print_hits(hits) -> None:
    rows = [asdict(hit) for hit in hits]
    safe_print(json.dumps(rows, ensure_ascii=False, indent=2))


def safe_print(text: str) -> None:
    print(text.encode("utf-8", errors="replace").decode("utf-8").encode("gbk", errors="replace").decode("gbk"))


def command_search(args: argparse.Namespace) -> None:
    load_env()
    if args.backend == "qdrant" and args.mode == "text":
        hits = qdrant_text_search(args.query, top_k=args.top_k)
    elif args.backend == "qdrant" and args.mode == "text-to-image":
        hits = qdrant_text_to_image_search(args.query, top_k=args.top_k, provider_name=args.image_provider)
    elif args.backend == "qdrant" and args.mode == "image-to-image":
        if not args.image:
            raise RuntimeError("--image is required for image-to-image search")
        hits = qdrant_image_to_image_search(Path(args.image), top_k=args.top_k, provider_name=args.image_provider)
    elif args.mode == "text":
        hits = search_text(args.query, top_k=args.top_k)
    elif args.mode == "text-to-image":
        hits = search_images_by_text(args.query, top_k=args.top_k)
    elif args.mode == "image-to-image":
        if not args.image:
            raise RuntimeError("--image is required for image-to-image search")
        hits = search_images_by_image(Path(args.image), top_k=args.top_k)
    elif args.mode == "hybrid":
        hits = hybrid_search(
            args.query,
            image_path=Path(args.image) if args.image else None,
            top_k=args.top_k,
            backend=args.backend,
            image_provider=args.image_provider,
        )
    else:
        raise RuntimeError(f"Unsupported search mode: {args.mode}")
    print_hits(hits)


def command_eval(args: argparse.Namespace) -> None:
    load_env()
    results = run_eval(top_k=args.top_k, backend=args.backend)
    write_eval_report(results)
    safe_print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def command_answer(args: argparse.Namespace) -> None:
    load_env()
    safe_print(answer_json(args.question, top_k=args.top_k, backend=args.backend))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal multimodal asset RAG project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse PDF and image assets")
    parse_cmd.add_argument("--limit", type=int, default=0)
    parse_cmd.add_argument("--pdf-parser", choices=["auto", "pymupdf", "paddleocr_vl"], default="auto")
    parse_cmd.add_argument("--ocr", action="store_true", help="Run local OCR HTTP for images")
    parse_cmd.add_argument("--vlm", action="store_true", help="Run OpenAI-compatible VLM captions for images")
    parse_cmd.set_defaults(func=command_parse)

    index_cmd = subparsers.add_parser("index", help="Build text and image indexes")
    index_cmd.add_argument("--backend", choices=["qdrant", "llamaindex"], default="qdrant")
    index_cmd.add_argument(
        "--image-provider",
        choices=["lite", "sentence_transformers"],
        default="lite",
        help="lite is a runnable fallback; sentence_transformers uses a CLIP model if installed",
    )
    index_cmd.set_defaults(func=command_index)

    search_cmd = subparsers.add_parser("search", help="Search indexed assets")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--mode", choices=["text", "text-to-image", "image-to-image", "hybrid"], default="hybrid")
    search_cmd.add_argument("--image", default="")
    search_cmd.add_argument("--top-k", type=int, default=5)
    search_cmd.add_argument("--backend", choices=["qdrant", "llamaindex"], default="qdrant")
    search_cmd.add_argument("--image-provider", choices=["lite", "sentence_transformers"], default="lite")
    search_cmd.set_defaults(func=command_search)

    eval_cmd = subparsers.add_parser("eval", help="Run a small retrieval evaluation set")
    eval_cmd.add_argument("--top-k", type=int, default=5)
    eval_cmd.add_argument("--backend", choices=["qdrant", "llamaindex"], default="qdrant")
    eval_cmd.set_defaults(func=command_eval)

    answer_cmd = subparsers.add_parser("answer", help="Answer with retrieved multimodal evidence")
    answer_cmd.add_argument("question")
    answer_cmd.add_argument("--top-k", type=int, default=5)
    answer_cmd.add_argument("--backend", choices=["qdrant", "llamaindex"], default="qdrant")
    answer_cmd.set_defaults(func=command_answer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
