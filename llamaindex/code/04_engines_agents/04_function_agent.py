import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from common import Settings, build_example_index, configure_llamaindex


async def main() -> None:
    """实验 4：让 Agent 在工具中选择并调用 QueryEngine。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_llamaindex_notes",
            description="查询 LlamaIndex 教程笔记，适合回答 RAG、索引、检索和生产化问题。",
        ),
    )

    agent = FunctionAgent(
        name="llamaindex_helper",
        description="回答 LlamaIndex 教程相关问题",
        tools=[rag_tool],
        llm=Settings.llm,
        system_prompt="你是 LlamaIndex 教程助手。需要查资料时先调用工具，再基于工具结果回答。",
        verbose=True,
        timeout=60,
    )

    response = await agent.run(user_msg="Retriever 和 QueryEngine 的区别是什么？")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
