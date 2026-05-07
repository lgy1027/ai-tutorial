import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_example_index, configure_llamaindex


def main() -> None:
    """实验 1：使用默认 QueryEngine 完成一次查询。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("生产环境的 RAG 系统需要关注哪些问题？")

    print(response)
    print("\nsource_nodes:", len(response.source_nodes))


if __name__ == "__main__":
    main()
