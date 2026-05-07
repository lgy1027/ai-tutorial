import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from common import build_example_index, configure_llamaindex


class NeedRewriteEvent(Event):
    question: str


class ReadyToAnswerEvent(Event):
    question: str


class BranchingRagWorkflow(Workflow):
    """实验 3：用分支表达简单的查询改写逻辑。"""

    def __init__(self, query_engine, **kwargs):
        super().__init__(**kwargs)
        self.query_engine = query_engine

    @step
    async def inspect_question(self, ev: StartEvent) -> NeedRewriteEvent | ReadyToAnswerEvent:
        question = ev.question
        if len(question) < 8:
            return NeedRewriteEvent(question=question)
        return ReadyToAnswerEvent(question=question)

    @step
    async def rewrite_question(self, ev: NeedRewriteEvent) -> ReadyToAnswerEvent:
        rewritten = f"请结合 LlamaIndex 教程资料回答：{ev.question}"
        print("rewritten question:", rewritten)
        return ReadyToAnswerEvent(question=rewritten)

    @step
    async def answer(self, ev: ReadyToAnswerEvent) -> StopEvent:
        response = self.query_engine.query(ev.question)
        return StopEvent(result=str(response))


async def main() -> None:
    configure_llamaindex()

    index = build_example_index()
    workflow = BranchingRagWorkflow(
        query_engine=index.as_query_engine(similarity_top_k=2),
        timeout=60,
        verbose=False,
    )

    result = await workflow.run(question="检索")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
