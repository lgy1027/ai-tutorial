import json
import os
import time
from pathlib import Path

import fitz
import requests

from .assets import Asset
from .config import PARSED_DIR, env_bool
from .schema import ParsedDocument


def parse_pdf_with_pymupdf(asset: Asset) -> list[ParsedDocument]:
    """Local fallback parser for text-based PDFs."""
    output_dir = PARSED_DIR / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    docs: list[ParsedDocument] = []

    with fitz.open(asset.file_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            markdown_path = output_dir / f"page_{page_index}.md"
            markdown_path.write_text(text, encoding="utf-8")
            docs.append(
                ParsedDocument(
                    text=text,
                    metadata={
                        "asset_id": asset.asset_id,
                        "asset_title": asset.title,
                        "source_type": asset.source_type,
                        "source_path": asset.relative_path,
                        "source_url": asset.source_url,
                        "page": page_index,
                        "parser": "pymupdf",
                        "markdown_path": str(markdown_path),
                        "tags": asset.tags,
                    },
                )
            )
    return docs


def submit_paddleocr_vl_job(file_path: Path) -> str:
    token = os.environ.get("PADDLEOCR_VL_API_TOKEN")
    if not token:
        raise RuntimeError("PADDLEOCR_VL_API_TOKEN is required for PaddleOCR-VL parsing")

    job_url = os.environ.get(
        "PADDLEOCR_VL_JOB_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    )
    model = os.environ.get("PADDLEOCR_VL_MODEL", "PaddleOCR-VL-1.6")
    timeout = float(os.environ.get("PADDLEOCR_VL_TIMEOUT", "300"))
    optional_payload = {
        "useDocOrientationClassify": env_bool("PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY", False),
        "useDocUnwarping": env_bool("PADDLEOCR_VL_USE_DOC_UNWARPING", False),
        "useChartRecognition": env_bool("PADDLEOCR_VL_USE_CHART_RECOGNITION", False),
    }

    headers = {"Authorization": f"bearer {token}"}
    with file_path.open("rb") as file_obj:
        response = requests.post(
            job_url,
            headers=headers,
            data={
                "model": model,
                "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
            },
            files={"file": file_obj},
            timeout=timeout,
        )
    if response.status_code != 200:
        raise RuntimeError(f"PaddleOCR-VL submit failed: {response.status_code} {response.text}")
    return str(response.json()["data"]["jobId"])


def poll_paddleocr_vl_job(job_id: str) -> str:
    token = os.environ.get("PADDLEOCR_VL_API_TOKEN")
    job_url = os.environ.get(
        "PADDLEOCR_VL_JOB_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    )
    timeout = float(os.environ.get("PADDLEOCR_VL_TIMEOUT", "300"))
    poll_interval = float(os.environ.get("PADDLEOCR_VL_POLL_INTERVAL", "5"))
    headers = {"Authorization": f"bearer {token}"}
    deadline = time.time() + timeout

    while time.time() < deadline:
        response = requests.get(f"{job_url}/{job_id}", headers=headers, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"PaddleOCR-VL poll failed: {response.status_code} {response.text}")
        payload = response.json()["data"]
        state = payload["state"]
        if state == "done":
            return str(payload["resultUrl"]["jsonUrl"])
        if state == "failed":
            raise RuntimeError(f"PaddleOCR-VL job failed: {payload.get('errorMsg')}")
        print(f"waiting PaddleOCR-VL job={job_id} state={state}")
        time.sleep(poll_interval)
    raise RuntimeError(f"PaddleOCR-VL job timeout: {job_id}")


def parse_with_paddleocr_vl(asset: Asset) -> list[ParsedDocument]:
    output_dir = PARSED_DIR / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = output_dir / "raw.jsonl"

    if raw_jsonl_path.exists() and raw_jsonl_path.stat().st_size > 0:
        jsonl_text = raw_jsonl_path.read_text(encoding="utf-8")
    else:
        job_id = submit_paddleocr_vl_job(asset.file_path)
        jsonl_url = poll_paddleocr_vl_job(job_id)
        response = requests.get(jsonl_url, timeout=float(os.environ.get("PADDLEOCR_VL_TIMEOUT", "300")))
        response.raise_for_status()
        jsonl_text = response.text
        raw_jsonl_path.write_text(jsonl_text, encoding="utf-8")

    docs: list[ParsedDocument] = []
    page_num = 0
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        result = json.loads(line).get("result", {})
        for parsed_page in result.get("layoutParsingResults", []):
            markdown = parsed_page.get("markdown", {})
            markdown_text = str(markdown.get("text", "")).strip()
            if not markdown_text:
                page_num += 1
                continue
            markdown_path = output_dir / f"page_{page_num}.md"
            markdown_path.write_text(markdown_text, encoding="utf-8")
            docs.append(
                ParsedDocument(
                    text=markdown_text,
                    metadata={
                        "asset_id": asset.asset_id,
                        "asset_title": asset.title,
                        "source_type": asset.source_type,
                        "source_path": asset.relative_path,
                        "source_url": asset.source_url,
                        "page": page_num,
                        "parser": "paddleocr-vl-api",
                        "markdown_path": str(markdown_path),
                        "tags": asset.tags,
                    },
                )
            )
            page_num += 1
    return docs


def parse_pdf(asset: Asset, parser: str) -> list[ParsedDocument]:
    if parser == "paddleocr_vl":
        return parse_with_paddleocr_vl(asset)
    if parser == "pymupdf":
        return parse_pdf_with_pymupdf(asset)
    if parser == "auto":
        if os.environ.get("PADDLEOCR_VL_API_TOKEN"):
            return parse_with_paddleocr_vl(asset)
        return parse_pdf_with_pymupdf(asset)
    raise ValueError(f"Unsupported PDF parser: {parser}")

