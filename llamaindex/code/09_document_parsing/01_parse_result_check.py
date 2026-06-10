import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from common import ROOT_DIR


PARSED_FILE = ROOT_DIR / "data" / "advanced_docs" / "parsed_doc_bad.md"


def simple_tokenizer(text: str) -> list[str]:
    return text.split()


def load_parsed_document() -> Document:
    """把解析后的文本显式包装成 Document，便于补 metadata 和检查来源。"""
    text = PARSED_FILE.read_text(encoding="utf-8")
    return Document(text=text, metadata={"source": PARSED_FILE.name, "stage": "parsed"})


def split_for_inspection(document: Document):
    """切成 Node 后再检查，因为真正进入索引的是 Node。"""
    splitter = SentenceSplitter(
        chunk_size=120,
        chunk_overlap=20,
        tokenizer=simple_tokenizer,
    )
    return splitter.get_nodes_from_documents([document])


def print_parse_warnings(text: str) -> None:
    """输出几个最小检查项，真实项目可以扩成解析质量门禁。"""
    checks = {
        "包含页眉": "页眉：" in text,
        "包含页脚": "页脚：" in text,
        "疑似表格被打散": "| E-503\n|" in text or "| 时间\n|" in text,
        "疑似正文缺失": "未识别" in text or "无法识别" in text,
    }
    print("=== Parse warnings ===")
    for name, matched in checks.items():
        print(f"{name}: {'yes' if matched else 'no'}")


def main() -> None:
    """进阶 3：先检查解析结果，再讨论切块和检索。"""
    document = load_parsed_document()
    print_parse_warnings(document.text)
    nodes = split_for_inspection(document)

    print(f"\ndocuments=1 nodes={len(nodes)}")
    for index, node in enumerate(nodes, start=1):
        text = node.get_content().replace("\n", " ")
        print(f"\n[{index}] {text}")


if __name__ == "__main__":
    main()
