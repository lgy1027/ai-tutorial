import base64
import json
import os
import urllib.request
from pathlib import Path

import requests

from .assets import Asset
from .config import CAPTION_DIR, PARSED_DIR
from .schema import ParsedDocument


def call_ocr_http(image_path: Path) -> list[dict[str, object]]:
    url = os.environ.get("OCR_HTTP_URL", "http://127.0.0.1:8000/ocr")
    timeout = float(os.environ.get("OCR_HTTP_TIMEOUT", "60"))
    body = json.dumps(
        {
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "file_name": image_path.name,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_ocr_blocks(payload)


def normalize_ocr_blocks(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_items = payload.get("blocks") or payload.get("results") or payload.get("data") or []
    blocks = []
    for item in raw_items:
        if isinstance(item, str):
            blocks.append({"text": item, "bbox": None, "confidence": None})
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("value")
            if text:
                blocks.append(
                    {
                        "text": str(text).strip(),
                        "bbox": item.get("bbox") or item.get("box") or item.get("points"),
                        "confidence": item.get("confidence") or item.get("score"),
                    }
                )
    return [block for block in blocks if block["text"]]


def call_vlm_caption(image_path: Path) -> str:
    base_url = os.environ.get("VLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("VLM_MODEL") or os.environ.get("OPENAI_MODEL")
    if not base_url or not api_key or not model:
        return ""

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower().replace(".", "") or "png"
    mime = "jpeg" if suffix == "jpg" else suffix
    payload = {
        "model": model,
        "temperature": float(os.environ.get("VLM_TEMPERATURE", "0.1")),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请用中文描述这张图片。重点说明可检索的对象、文字、场景、用途和可能回答的问题。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{image_base64}"},
                    },
                ],
            }
        ],
    }
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(os.environ.get("VLM_TIMEOUT", "120")),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(content).strip()


def parse_image(asset: Asset, enable_ocr: bool, enable_vlm: bool) -> list[ParsedDocument]:
    output_dir = PARSED_DIR / asset.asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_path = output_dir / "ocr.json"
    caption_path = CAPTION_DIR / f"{asset.asset_id}.json"

    blocks: list[dict[str, object]] = []
    if enable_ocr:
        if ocr_path.exists() and ocr_path.stat().st_size > 0:
            blocks = json.loads(ocr_path.read_text(encoding="utf-8")).get("blocks", [])
        else:
            try:
                blocks = call_ocr_http(asset.file_path)
            except Exception as exc:
                print(f"OCR skipped for {asset.asset_id}: {exc}")
                blocks = []
            ocr_path.write_text(
                json.dumps({"engine": "ocr-http", "blocks": blocks}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    caption = ""
    if enable_vlm:
        if caption_path.exists() and caption_path.stat().st_size > 0:
            caption = str(json.loads(caption_path.read_text(encoding="utf-8")).get("caption", ""))
        else:
            try:
                caption = call_vlm_caption(asset.file_path)
            except Exception as exc:
                print(f"VLM caption skipped for {asset.asset_id}: {exc}")
                caption = ""
            caption_path.write_text(
                json.dumps({"engine": "openai-compatible-vlm", "caption": caption}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    ocr_text = "\n".join(str(block["text"]) for block in blocks if block.get("text"))
    text = (
        f"图片标题：{asset.title}\n"
        f"图片标签：{', '.join(asset.tags)}\n"
        f"VLM 描述：{caption}\n"
        f"OCR 文本：\n{ocr_text}\n"
        f"原图：{asset.relative_path}"
    ).strip()
    return [
        ParsedDocument(
            text=text,
            metadata={
                "asset_id": asset.asset_id,
                "asset_title": asset.title,
                "source_type": asset.source_type,
                "source_path": asset.relative_path,
                "source_url": asset.source_url,
                "page": None,
                "parser": "image-ocr-vlm",
                "ocr_path": str(ocr_path),
                "caption_path": str(caption_path),
                "tags": asset.tags,
            },
        )
    ]

