import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class BasicWorkflow(Workflow):
    """实验 1：最小 Workflow。"""

    @step
    async def run_step(self, ev: StartEvent) -> StopEvent:
        topic = ev.topic
        return StopEvent(result=f"收到任务：{topic}")


async def main() -> None:
    workflow = BasicWorkflow(timeout=10, verbose=False)
    result = await workflow.run(topic="理解 LlamaIndex Workflow 的 StartEvent 和 StopEvent")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
