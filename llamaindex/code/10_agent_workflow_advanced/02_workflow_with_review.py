import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class NeedReviewEvent(Event):
    question: str
    draft: str


class ReviewedEvent(Event):
    answer: str


class SupportWorkflow(Workflow):
    """进阶 4：把固定流程写进 Workflow，而不是全交给 Agent。"""

    @step
    async def draft_answer(self, ev: StartEvent) -> NeedReviewEvent:
        question = ev.question
        if "创建工单" in question:
            draft = "该请求涉及外部动作，建议先人工确认后再创建工单。"
        else:
            draft = "该问题可以直接从知识库查询后回答。"
        return NeedReviewEvent(question=question, draft=draft)

    @step
    async def review(self, ev: NeedReviewEvent) -> ReviewedEvent:
        if "人工确认" in ev.draft:
            answer = f"{ev.draft}\n当前示例自动模拟为：等待人工确认。"
        else:
            answer = ev.draft
        return ReviewedEvent(answer=answer)

    @step
    async def finish(self, ev: ReviewedEvent) -> StopEvent:
        return StopEvent(result=ev.answer)


async def main() -> None:
    workflow = SupportWorkflow(timeout=30)
    result = await workflow.run(question="请根据知识库结论创建工单")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
