import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from common import Settings, configure_llamaindex


def query_incident(code: str) -> str:
    """查询故障码处理建议。"""
    if code == "E-503":
        return "E-503 表示索引构建任务未完成，先检查 ingestion job 状态。"
    if code == "E-401":
        return "E-401 表示权限不足，先确认用户分组和文档 visibility。"
    return "没有查到这个故障码，请转人工复核。"


def create_ticket(summary: str) -> str:
    """模拟创建工单。"""
    return f"已创建待确认工单: {summary}"


async def main() -> None:
    """进阶 4：多工具 Agent 的边界要清楚。"""
    configure_llamaindex()

    tools = [
        FunctionTool.from_defaults(fn=query_incident),
        FunctionTool.from_defaults(fn=create_ticket),
    ]
    agent = FunctionAgent(
        name="support_agent",
        tools=tools,
        llm=Settings.llm,
        system_prompt=(
            "你是客服知识库助手。先查询故障码，再决定是否需要创建工单。"
            "回答要说明调用了什么能力。"
        ),
    )
    response = await agent.run(user_msg="E-503 怎么处理？需要创建一个复核工单。")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
