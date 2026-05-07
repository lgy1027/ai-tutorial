import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

from llama_index.core import StorageContext, load_index_from_storage

from common import STORAGE_DIR, build_example_index, configure_llamaindex


def main() -> None:
    """实验 5：持久化索引并重新加载。"""
    configure_llamaindex()

    persist_dir = STORAGE_DIR / "example_index"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    index = build_example_index()
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"索引已持久化到: {persist_dir}")

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("RAG 系统为什么需要评估和观测？")

    print(response)


if __name__ == "__main__":
    main()
