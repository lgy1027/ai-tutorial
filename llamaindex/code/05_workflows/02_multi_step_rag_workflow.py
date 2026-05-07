import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from common import build_example_index, configure_llamaindex, print_nodes


class RetrievedEvent(Event):
    question: str
    context: str


class AnswerEvent(Event):
    question: str
    answer: str


class RagWorkflow(Workflow):
    """实验 2：检索和回答拆成两个 step。"""

    def __init__(self, retriever, query_engine, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.query_engine = query_engine

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievedEvent:
        nodes = self.retriever.retrieve(ev.question)
        print("\n=== Retrieved Nodes ===")
        print_nodes(nodes)
        context = "\n\n".join(node.node.get_content() for node in nodes)
        return RetrievedEvent(question=ev.question, context=context)

    @step
    async def answer(self, ev: RetrievedEvent) -> AnswerEvent:
        response = self.query_engine.query(ev.question)
        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def finish(self, ev: AnswerEvent) -> StopEvent:
        return StopEvent(result=ev.answer)


async def main() -> None:
    configure_llamaindex()

    index = build_example_index()
    workflow = RagWorkflow(
        retriever=index.as_retriever(similarity_top_k=2),
        query_engine=index.as_query_engine(similarity_top_k=2),
        timeout=60,
        verbose=False,
    )

    result = await workflow.run(question="Workflow 适合解决什么问题？")
    print("\n=== Answer ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
