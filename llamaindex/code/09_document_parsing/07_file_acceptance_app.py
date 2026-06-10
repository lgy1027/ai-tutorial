import csv
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader

from common import ROOT_DIR


FILE_DIR = ROOT_DIR / "data" / "file_docs"
SUPPORTED_EXTS = [".md", ".txt", ".csv", ".html", ".pdf"]
PARSER_VERSION = "local-file-gate-v1"


@dataclass(frozen=True)
class WarningRule:
    name: str
    weight: int
    description: str


WARNING_RULES = {
    "missing_source": WarningRule("missing_source", 3, "缺少来源文件名"),
    "text_too_short": WarningRule("text_too_short", 3, "正文过短，可能没有解析出有效内容"),
    "page_noise": WarningRule("page_noise", 1, "页眉页脚混入正文"),
    "broken_table": WarningRule("broken_table", 3, "表格结构疑似断裂"),
    "body_missing": WarningRule("body_missing", 4, "正文内容疑似缺失"),
    "html_noise": WarningRule("html_noise", 2, "网页导航或页脚疑似混入正文"),
    "csv_missing_fields": WarningRule("csv_missing_fields", 3, "CSV 行缺少关键字段"),
    "metadata_missing": WarningRule("metadata_missing", 3, "缺少入库所需 metadata"),
}


class ArticleHTMLParser(HTMLParser):
    """只抽取 article 内容；没有 article 时退回到 body 文本。"""

    def __init__(self) -> None:
        super().__init__()
        self._in_article = False
        self._article_parts: list[str] = []
        self._body_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "article":
            self._in_article = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "article":
            self._in_article = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        self._body_parts.append(text)
        if self._in_article:
            self._article_parts.append(text)

    def text(self) -> str:
        parts = self._article_parts or self._body_parts
        return "\n".join(parts)


class CsvMetricReader(BaseReader):
    """把 CSV 每一行转成一个 Document，避免整张表变成一大段文本。"""

    def load_data(self, file: str | Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        path = Path(file)
        documents: list[Document] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=1):
                text = (
                    f"模块：{row.get('module', '')}\n"
                    f"指标：{row.get('metric', '')}\n"
                    f"当前值：{row.get('value', '')}{row.get('unit', '')}\n"
                    f"负责人：{row.get('owner', '')}"
                )
                metadata = {
                    **(extra_info or {}),
                    "source": path.name,
                    "file_type": ".csv",
                    "extension": ".csv",
                    "asset_type": "table_row",
                    "row_number": row_number,
                    "document_id": f"{path.name}#row-{row_number}",
                    "module": row.get("module", ""),
                    "metric": row.get("metric", ""),
                    "parser_version": PARSER_VERSION,
                }
                documents.append(Document(text=text, metadata=metadata))
        return documents


class ArticleHTMLReader(BaseReader):
    """从 HTML 中抽取正文，避免导航和页脚直接进入知识库。"""

    def load_data(self, file: str | Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        path = Path(file)
        parser = ArticleHTMLParser()
        parser.feed(path.read_text(encoding="utf-8"))
        metadata = {
            **(extra_info or {}),
            "source": path.name,
            "file_type": ".html",
            "extension": ".html",
            "asset_type": "html_article",
            "document_id": path.name,
            "parser_version": PARSER_VERSION,
        }
        return [Document(text=parser.text(), metadata=metadata)]


class LocalPdfReader(BaseReader):
    """用本地 PyMuPDF 解析 PDF，每页生成一个 Document。"""

    def load_data(self, file: str | Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("本地解析 PDF 需要安装 PyMuPDF: pip install pymupdf") from exc

        path = Path(file)
        documents: list[Document] = []
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                metadata = {
                    **(extra_info or {}),
                    "source": path.name,
                    "file_type": ".pdf",
                    "extension": ".pdf",
                    "asset_type": "pdf_page",
                    "document_id": f"{path.name}#page-{page_index}",
                    "page_number": page_index,
                    "page_count": pdf.page_count,
                    "parser_version": PARSER_VERSION,
                }
                documents.append(Document(text=text, metadata=metadata))
        return documents


def simple_tokenizer(text: str) -> list[str]:
    """避免示例依赖默认 tiktoken 下载；生产环境可替换成模型对应 tokenizer。"""
    return re.findall(r"[\w\u4e00-\u9fff]+|[^\w\s]", text, flags=re.UNICODE)


def list_manifest(input_dir: Path = FILE_DIR) -> list[dict[str, object]]:
    manifest = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        manifest.append(
            {
                "file": path.name,
                "extension": suffix,
                "supported": suffix in SUPPORTED_EXTS,
                "strategy": choose_strategy(suffix),
            }
        )
    return manifest


def choose_strategy(extension: str) -> str:
    if extension in {".md", ".txt"}:
        return "SimpleDirectoryReader"
    if extension == ".pdf":
        return "LocalPdfReader via file_extractor"
    if extension == ".csv":
        return "CsvMetricReader via file_extractor"
    if extension in {".html", ".htm"}:
        return "ArticleHTMLReader via file_extractor"
    if extension in {".docx", ".pptx", ".xlsx"}:
        return "LiteParse/LlamaParse or project-specific reader"
    return "unsupported"


def load_documents(input_dir: Path = FILE_DIR) -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=str(input_dir),
        required_exts=SUPPORTED_EXTS,
        recursive=False,
        file_extractor={
            ".csv": CsvMetricReader(),
            ".html": ArticleHTMLReader(),
            ".pdf": LocalPdfReader(),
        },
    )
    documents = reader.load_data()
    return [normalize_document(document) for document in documents]


def normalize_document(document: Document) -> Document:
    metadata = dict(document.metadata)
    source = metadata.get("file_name") or metadata.get("source") or "unknown"
    suffix = Path(str(source)).suffix.lower()
    if metadata.get("file_type") and not str(metadata["file_type"]).startswith("."):
        metadata.setdefault("mime_type", metadata["file_type"])
    metadata.setdefault("source", str(source))
    metadata["extension"] = str(metadata.get("extension") or suffix or "unknown")
    metadata["file_type"] = metadata["extension"]
    metadata.setdefault("asset_type", infer_asset_type(metadata))
    metadata.setdefault("document_id", str(source))
    metadata.setdefault("parser_version", PARSER_VERSION)
    metadata.setdefault("stage", "parsed")
    return Document(text=document.text, metadata=metadata)


def infer_asset_type(metadata: dict[str, Any]) -> str:
    file_type = str(metadata.get("extension") or metadata.get("file_type") or "").lower()
    if metadata.get("asset_type"):
        return str(metadata["asset_type"])
    if file_type == ".csv":
        return "table"
    if file_type == ".pdf":
        return "pdf_page"
    if file_type in {".html", ".htm"}:
        return "html"
    if file_type in {".md", ".txt"}:
        return "text"
    return "file"


def clean_text(text: str) -> str:
    lines: list[str] = []
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


def split_nodes(document: Document):
    splitter = SentenceSplitter(
        chunk_size=260,
        chunk_overlap=30,
        tokenizer=simple_tokenizer,
    )
    return splitter.get_nodes_from_documents([document])


def inspect_document(document: Document) -> dict[str, object]:
    cleaned_text = clean_text(document.text)
    cleaned_document = Document(text=cleaned_text, metadata={**document.metadata, "stage": "cleaned"})
    warnings = collect_warnings(document, cleaned_text)
    score = score_warnings(warnings)
    decision = decide(score, warnings)
    nodes = split_nodes(cleaned_document)
    return {
        "source": document.metadata["source"],
        "document_id": document.metadata["document_id"],
        "file_type": document.metadata["file_type"],
        "extension": document.metadata["extension"],
        "asset_type": document.metadata["asset_type"],
        "parser_version": document.metadata["parser_version"],
        "raw_chars": len(document.text),
        "cleaned_chars": len(cleaned_text),
        "node_count": len(nodes),
        "warnings": warnings,
        "quality_score": score,
        "decision": decision,
        "accepted_for_ingestion": decision == "accepted",
        "next_action": next_action(decision, warnings),
        "metadata_keys": sorted(document.metadata.keys()),
    }


def collect_warnings(document: Document, cleaned_text: str) -> list[str]:
    metadata = document.metadata
    text = document.text
    warnings: list[str] = []
    if not metadata.get("source"):
        warnings.append("missing_source")
    if metadata.get("asset_type") != "table_row" and len(cleaned_text.strip()) < 80:
        warnings.append("text_too_short")
    if "页眉：" in text or "页脚：" in text:
        warnings.append("page_noise")
    if "| E-503\n|" in text or "| 时间\n|" in text:
        warnings.append("broken_table")
    if "未识别" in text or "无法识别" in text:
        warnings.append("body_missing")
    if metadata.get("asset_type") == "html_article" and any(
        word in cleaned_text for word in ["登录", "Copyright", "下载"]
    ):
        warnings.append("html_noise")
    if metadata.get("asset_type") == "table_row" and not (
        metadata.get("module") and metadata.get("metric")
    ):
        warnings.append("csv_missing_fields")
    if metadata.get("asset_type") == "pdf_page" and not metadata.get("page_number"):
        warnings.append("metadata_missing")
    required_metadata = {
        "source",
        "document_id",
        "file_type",
        "extension",
        "asset_type",
        "parser_version",
        "stage",
    }
    if not required_metadata.issubset(metadata):
        warnings.append("metadata_missing")
    return warnings


def score_warnings(warnings: list[str]) -> int:
    return sum(WARNING_RULES[warning].weight for warning in warnings)


def decide(score: int, warnings: list[str]) -> str:
    if not warnings:
        return "accepted"
    if score >= 5 or "body_missing" in warnings:
        return "rejected"
    return "needs_review"


def next_action(decision: str, warnings: list[str]) -> str:
    if decision == "accepted":
        return "进入 ingestion"
    warning_text = ", ".join(warnings)
    if decision == "needs_review":
        return f"人工复核后再入库：{warning_text}"
    return f"暂停入库，先修复解析或清洗策略：{warning_text}"


def build_acceptance_report(input_dir: Path = FILE_DIR) -> dict[str, object]:
    manifest = list_manifest(input_dir)
    documents = load_documents(input_dir)
    document_reports = [inspect_document(document) for document in documents]
    accepted = [item for item in document_reports if item["decision"] == "accepted"]
    needs_review = [item for item in document_reports if item["decision"] == "needs_review"]
    rejected = [item for item in document_reports if item["decision"] == "rejected"]
    return {
        "input_dir": str(input_dir.relative_to(ROOT_DIR)),
        "manifest": manifest,
        "summary": {
            "files": len(manifest),
            "documents": len(document_reports),
            "accepted": len(accepted),
            "needs_review": len(needs_review),
            "rejected": len(rejected),
        },
        "ingestion_ready": [item["document_id"] for item in accepted],
        "blocked": [item["document_id"] for item in needs_review + rejected],
        "documents": document_reports,
    }


def main() -> None:
    report = build_acceptance_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
