import json

from .config import DOCUMENTS_JSONL
from .schema import ParsedDocument


def write_documents(documents: list[ParsedDocument]) -> None:
    DOCUMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with DOCUMENTS_JSONL.open("w", encoding="utf-8") as file_obj:
        for document in documents:
            file_obj.write(json.dumps(document.to_json(), ensure_ascii=False) + "\n")


def read_documents() -> list[ParsedDocument]:
    if not DOCUMENTS_JSONL.exists():
        raise RuntimeError(f"Document JSONL not found: {DOCUMENTS_JSONL}")
    documents = []
    with DOCUMENTS_JSONL.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            documents.append(
                ParsedDocument(
                    text=str(payload["text"]),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    return documents

