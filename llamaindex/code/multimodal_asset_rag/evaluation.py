import json
from dataclasses import asdict, dataclass

from .config import WORK_DIR
from .retrieval import hybrid_search


EVAL_CASES = [
    {
        "query": "哪份资料讲了 retrieval augmented generation？",
        "expected_asset_ids": ["pdf_rag"],
    },
    {
        "query": "找和 CLIP 图文对齐有关的资料",
        "expected_asset_ids": ["pdf_clip"],
    },
    {
        "query": "有没有包含文档版面理解或 OCR 的资料？",
        "expected_asset_ids": ["pdf_layoutlm"],
    },
]


@dataclass
class EvalResult:
    query: str
    expected_asset_ids: list[str]
    actual_asset_ids: list[str]
    hit: bool


def run_eval(top_k: int = 5, backend: str = "qdrant") -> list[EvalResult]:
    results = []
    for case in EVAL_CASES:
        hits = hybrid_search(str(case["query"]), top_k=top_k, backend=backend)
        actual = [hit.asset_id for hit in hits]
        expected = [str(item) for item in case["expected_asset_ids"]]
        results.append(
            EvalResult(
                query=str(case["query"]),
                expected_asset_ids=expected,
                actual_asset_ids=actual,
                hit=any(asset_id in actual for asset_id in expected),
            )
        )
    return results


def write_eval_report(results: list[EvalResult]) -> None:
    path = WORK_DIR / "eval_report.json"
    path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
