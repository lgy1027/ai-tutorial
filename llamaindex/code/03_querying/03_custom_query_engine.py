import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from common import build_example_index, configure_llamaindex


def main() -> None:
    """实验 3：手动组装 RetrieverQueryEngine。"""
    configure_llamaindex()

    index = build_example_index()
    retriever = index.as_retriever(similarity_top_k=2)
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    response = query_engine.query("LlamaIndex 的 QueryEngine 做了什么？")
    print(response)


if __name__ == "__main__":
    main()
