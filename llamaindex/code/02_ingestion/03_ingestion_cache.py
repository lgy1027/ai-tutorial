import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from common import configure_llamaindex, load_example_documents


def main() -> None:
    """示例 3：使用 IngestionCache 说明重复处理为什么要缓存。"""
    configure_llamaindex()

    documents = load_example_documents()
    cache = IngestionCache(collection="llamaindex_tutorial_cache")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=120, chunk_overlap=20),
        ],
        cache=cache,
    )

    first_nodes = pipeline.run(documents=documents)
    second_nodes = pipeline.run(documents=documents)

    print("第一次执行 nodes:", len(first_nodes))
    print("第二次执行 nodes:", len(second_nodes))
    print("说明：相同输入和 transformation 会命中缓存，真实项目可替换为 Redis/MongoDB 等远程缓存。")


if __name__ == "__main__":
    main()
