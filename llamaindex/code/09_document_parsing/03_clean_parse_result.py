import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from common import ROOT_DIR


PARSED_FILE = ROOT_DIR / "data" / "advanced_docs" / "parsed_doc_bad.md"


def simple_tokenizer(text: str) -> list[str]:
    return text.split()


def clean_parsed_text(text: str) -> str:
    """演示最小清洗：去页眉页脚和空行。真实项目要更谨慎。"""
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


def build_clean_document(text: str) -> Document:
    """清洗后重新构造 Document，标记当前处理阶段。"""
    return Document(text=text, metadata={"source": PARSED_FILE.name, "stage": "cleaned"})


def main() -> None:
    raw_text = PARSED_FILE.read_text(encoding="utf-8")
    cleaned = clean_parsed_text(raw_text)
    document = build_clean_document(cleaned)
    splitter = SentenceSplitter(
        chunk_size=120,
        chunk_overlap=20,
        tokenizer=simple_tokenizer,
    )
    nodes = splitter.get_nodes_from_documents([document])

    print("=== Raw ===")
    print(raw_text[:500])
    print("\n=== Cleaned ===")
    print(cleaned[:500])
    print(f"\ncleaned_nodes={len(nodes)}")


if __name__ == "__main__":
    main()
