import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import SimpleDirectoryReader

from common import ROOT_DIR


FILE_READER_STRATEGIES = [
    {
        "extensions": [".md", ".txt"],
        "entrypoint": "SimpleDirectoryReader",
        "why": "文本结构直接，适合先用本地 reader 读取，再做质量验收。",
    },
    {
        "extensions": [".html", ".htm"],
        "entrypoint": "SimpleDirectoryReader or custom file_extractor",
        "why": "网页要重点处理正文抽取、导航、广告和链接来源。",
    },
    {
        "extensions": [".csv"],
        "entrypoint": "SimpleDirectoryReader or pandas/custom reader",
        "why": "CSV 可以按文本读，但真实项目更常按表头、行列和业务字段组织 Document。",
    },
    {
        "extensions": [".xlsx"],
        "entrypoint": "LlamaParse/LlamaCloud or custom spreadsheet reader",
        "why": "Excel 常有多 sheet、合并单元格、单位和备注，直接转成大段文本容易丢关系。",
    },
    {
        "extensions": [".pdf", ".docx", ".pptx"],
        "entrypoint": "LlamaParse/LlamaCloud or SimpleDirectoryReader with suitable dependencies",
        "why": "这类文件更容易出现版式、标题层级、页眉页脚、表格抽取问题，入库前要做验收。",
    },
]


def load_local_markdown_samples():
    """演示 SimpleDirectoryReader 读取本地 Markdown 样例。"""
    reader = SimpleDirectoryReader(
        input_dir=str(ROOT_DIR / "data" / "advanced_docs"),
        required_exts=[".md"],
        recursive=False,
    )
    return reader.load_data()


def main() -> None:
    documents = load_local_markdown_samples()
    report = {
        "reader": "SimpleDirectoryReader",
        "loaded_documents": len(documents),
        "sample_sources": [
            doc.metadata.get("file_name") or doc.metadata.get("source")
            for doc in documents[:5]
        ],
        "file_reader_strategies": FILE_READER_STRATEGIES,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
