import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from common import Settings, build_example_index, configure_llamaindex


def main() -> None:
    """实验 4：用 LlamaDebugHandler 查看一次查询的内部事件。"""
    configure_llamaindex()

    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("为什么 source_nodes 对 RAG 调试很重要？")
    print(response)


if __name__ == "__main__":
    main()
