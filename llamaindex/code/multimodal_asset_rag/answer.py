import json
import os
import re

import requests

from .retrieval import hybrid_search
from .schema import SearchHit


def format_sources(hits: list[SearchHit]) -> list[dict[str, object]]:
    return [
        {
            "asset_id": hit.asset_id,
            "title": hit.title,
            "source_type": hit.source_type,
            "source_path": hit.source_path,
            "score": round(hit.score, 4),
            "routes": hit.metadata.get("routes", [hit.route]),
            "page": hit.metadata.get("page"),
            "parser": hit.metadata.get("parser") or hit.metadata.get("provider"),
        }
        for hit in hits
    ]


def fallback_answer(question: str, hits: list[SearchHit]) -> dict[str, object]:
    evidence = "\n\n".join(hit.evidence[:300] for hit in hits[:3] if hit.evidence)
    return {
        "question": question,
        "answer": "当前未配置 LLM，因此返回检索证据摘要。请先检查 sources 中的原始资料、页码和解析器，再决定是否接入生成式回答。\n\n" + evidence,
        "sources": format_sources(hits),
    }


def llm_answer(question: str, hits: list[SearchHit]) -> dict[str, object]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not base_url or not api_key or not model:
        return fallback_answer(question, hits)

    context = "\n\n".join(
        f"[{index}] asset_id={hit.asset_id} title={hit.title} source={hit.source_path} "
        f"page={hit.metadata.get('page')}\n{hit.evidence[:1200]}"
        for index, hit in enumerate(hits, start=1)
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": "你是多模态资料检索助手。只能基于给定证据回答；如果证据不足，要明确说不足。回答要列出关键来源。",
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{context}",
            },
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    answer = re.sub(r"<think>.*?</think>", "", str(answer), flags=re.DOTALL).strip()
    return {
        "question": question,
        "answer": answer,
        "sources": format_sources(hits),
    }


def answer_question(question: str, top_k: int = 5, backend: str = "qdrant") -> dict[str, object]:
    hits = hybrid_search(question, top_k=top_k, backend=backend)
    return llm_answer(question, hits)


def answer_json(question: str, top_k: int = 5, backend: str = "qdrant") -> str:
    return json.dumps(answer_question(question, top_k=top_k, backend=backend), ensure_ascii=False, indent=2)
