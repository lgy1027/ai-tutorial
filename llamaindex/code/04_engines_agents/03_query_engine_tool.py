import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from common import build_example_index, configure_llamaindex


def main() -> None:
    """实验 3：把 QueryEngine 包装成 Agent 可以调用的 Tool。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="llamaindex_rag_notes",
            description="查询 LlamaIndex、RAG 数据链路和生产化注意事项。",
        ),
    )

    print("tool name:", tool.metadata.name)
    print("tool description:", tool.metadata.description)
    print("\n直接调用工具：")
    print(tool.call("LlamaIndex 的 Retriever 负责什么？"))


if __name__ == "__main__":
    main()
