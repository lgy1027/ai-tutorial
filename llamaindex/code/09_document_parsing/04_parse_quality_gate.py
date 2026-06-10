import importlib.util
import json
from pathlib import Path


CHECK_PATH = Path(__file__).with_name("01_parse_result_check.py")
check_spec = importlib.util.spec_from_file_location("parse_check", CHECK_PATH)
parse_check = importlib.util.module_from_spec(check_spec)
assert check_spec and check_spec.loader
check_spec.loader.exec_module(parse_check)

CLEAN_PATH = Path(__file__).with_name("03_clean_parse_result.py")
clean_spec = importlib.util.spec_from_file_location("parse_clean", CLEAN_PATH)
parse_clean = importlib.util.module_from_spec(clean_spec)
assert clean_spec and clean_spec.loader
clean_spec.loader.exec_module(parse_clean)


def collect_warnings(text: str) -> list[str]:
    """返回解析质量问题列表，作为进入 ingestion 前的门禁。"""
    checks = {
        "包含页眉": "页眉：" in text,
        "包含页脚": "页脚：" in text,
        "疑似表格被打散": "| E-503\n|" in text or "| 时间\n|" in text,
        "疑似正文缺失": "未识别" in text or "无法识别" in text,
    }
    return [name for name, matched in checks.items() if matched]


def build_quality_report() -> dict[str, object]:
    """把解析检查、清洗和 Node 检查合成一份质量报告。"""
    document = parse_check.load_parsed_document()
    raw_warnings = collect_warnings(document.text)
    cleaned_text = parse_clean.clean_parsed_text(document.text)
    cleaned_document = parse_clean.build_clean_document(cleaned_text)
    nodes = parse_check.split_for_inspection(cleaned_document)
    cleaned_warnings = collect_warnings(cleaned_text)
    decision = "pass" if not cleaned_warnings else "needs_review"
    return {
        "source": document.metadata["source"],
        "raw_warnings": raw_warnings,
        "cleaned_warnings": cleaned_warnings,
        "cleaned_warning_count": len(cleaned_warnings),
        "cleaned_nodes": len(nodes),
        "decision": decision,
        "next_action": (
            "进入 ingestion"
            if decision == "pass"
            else "先人工复核表格、正文抽取或版式问题，再进入 ingestion"
        ),
    }


def main() -> None:
    print(json.dumps(build_quality_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
