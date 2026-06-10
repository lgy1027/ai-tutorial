import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from common import ROOT_DIR


SAMPLE_FILES = [
    ROOT_DIR / "data" / "advanced_docs" / "parsed_doc_good.md",
    ROOT_DIR / "data" / "advanced_docs" / "parsed_doc_needs_review.md",
    ROOT_DIR / "data" / "advanced_docs" / "parsed_doc_bad.md",
]

PARSER_VERSION = "demo-parser-v1"
PARSE_OUTPUT_TYPE = "markdown"


def simple_tokenizer(text: str) -> list[str]:
    return text.split()


WARNING_WEIGHTS = {
    "包含页眉": 1,
    "包含页脚": 1,
    "疑似表格被打散": 2,
    "疑似正文缺失": 3,
    "文本过短": 2,
}


def clean_parsed_text(text: str) -> str:
    """保守清洗：只处理明显噪声，不尝试修复表格和正文缺失。"""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("页眉："):
            continue
        if re.match(r"页脚：第 \d+ 页", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def collect_warnings(text: str) -> list[str]:
    """返回解析质量问题，作为进入 ingestion 前的验收项。"""
    checks = {
        "包含页眉": "页眉：" in text,
        "包含页脚": "页脚：" in text,
        "疑似表格被打散": "| E-503\n|" in text or "| 时间\n|" in text,
        "疑似正文缺失": "未识别" in text or "无法识别" in text,
        "文本过短": len(text.strip()) < 80,
    }
    return [name for name, matched in checks.items() if matched]


def split_nodes(document: Document):
    splitter = SentenceSplitter(
        chunk_size=120,
        chunk_overlap=20,
        tokenizer=simple_tokenizer,
    )
    return splitter.get_nodes_from_documents([document])


def score_warnings(warnings: list[str]) -> int:
    return sum(WARNING_WEIGHTS.get(warning, 1) for warning in warnings)


def decide(score: int, warnings: list[str]) -> str:
    if not warnings:
        return "accepted"
    if score >= 4:
        return "rejected"
    return "needs_review"


def inspect_file(path: Path) -> dict[str, object]:
    raw_text = path.read_text(encoding="utf-8")
    raw_warnings = collect_warnings(raw_text)
    cleaned_text = clean_parsed_text(raw_text)
    cleaned_warnings = collect_warnings(cleaned_text)
    cleaned_document = Document(
        text=cleaned_text,
        metadata={
            "source": path.name,
            "stage": "cleaned",
            "parser_version": PARSER_VERSION,
            "parse_output_type": PARSE_OUTPUT_TYPE,
        },
    )
    nodes = split_nodes(cleaned_document)
    score = score_warnings(cleaned_warnings)
    decision = decide(score, cleaned_warnings)

    return {
        "source": path.name,
        "parser_version": PARSER_VERSION,
        "parse_output_type": PARSE_OUTPUT_TYPE,
        "raw_chars": len(raw_text),
        "cleaned_chars": len(cleaned_text),
        "raw_warnings": raw_warnings,
        "cleaned_warnings": cleaned_warnings,
        "quality_score": score,
        "node_count": len(nodes),
        "decision": decision,
        "accepted_for_ingestion": decision == "accepted",
        "next_action": build_next_action(decision, cleaned_warnings),
    }


def build_next_action(decision: str, warnings: list[str]) -> str:
    if decision == "accepted":
        return "进入 ingestion"
    if decision == "needs_review":
        return f"人工复核后再入库：{', '.join(warnings)}"
    return f"暂停入库，先修复解析策略：{', '.join(warnings)}"


def build_acceptance_report(paths: list[Path]) -> dict[str, object]:
    reports = [inspect_file(path) for path in paths]
    return {
        "summary": {
            "total": len(reports),
            "accepted": sum(1 for item in reports if item["decision"] == "accepted"),
            "needs_review": sum(1 for item in reports if item["decision"] == "needs_review"),
            "rejected": sum(1 for item in reports if item["decision"] == "rejected"),
        },
        "documents": reports,
        "ingestion_ready": [
            item["source"] for item in reports if item["decision"] == "accepted"
        ],
        "blocked": [
            item["source"] for item in reports if item["decision"] != "accepted"
        ],
    }


def main() -> None:
    print(json.dumps(build_acceptance_report(SAMPLE_FILES), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
