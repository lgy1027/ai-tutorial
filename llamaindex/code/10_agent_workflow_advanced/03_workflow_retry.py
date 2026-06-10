import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class RetrieveEvent(Event):
    question: str
    attempt: int
    top_k: int


class JudgeEvent(Event):
    question: str
    attempt: int
    top_k: int
    contexts: list[str]


class SynthesizeEvent(Event):
    question: str
    contexts: list[str]
    trace: list[str]


class ControlledRagWorkflow(Workflow):
    """进阶 4：把检索、判断、重试、生成拆成可观察流程。"""

    @step
    async def start(self, ev: StartEvent) -> RetrieveEvent:
        return RetrieveEvent(question=ev.question, attempt=1, top_k=1)

    @step
    async def retrieve(self, ev: RetrieveEvent) -> JudgeEvent:
        contexts = retrieve_contexts(ev.question, top_k=ev.top_k)
        print(f"第 {ev.attempt} 次检索：top_k={ev.top_k}，contexts={len(contexts)}")
        return JudgeEvent(
            question=ev.question,
            attempt=ev.attempt,
            top_k=ev.top_k,
            contexts=contexts,
        )

    @step
    async def judge(self, ev: JudgeEvent) -> RetrieveEvent | SynthesizeEvent:
        enough = any("E-503" in item and "索引构建" in item for item in ev.contexts)
        if not enough and ev.attempt < 2:
            print("上下文不足：扩大 top_k 后重试。")
            return RetrieveEvent(
                question=ev.question,
                attempt=ev.attempt + 1,
                top_k=3,
            )

        trace = [
            f"attempt={ev.attempt}",
            f"top_k={ev.top_k}",
            f"contexts={len(ev.contexts)}",
            f"enough={enough}",
        ]
        return SynthesizeEvent(question=ev.question, contexts=ev.contexts, trace=trace)

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        context_text = "\n".join(ev.contexts)
        answer = (
            "E-503 通常表示索引构建任务未完成，查询服务可能还在使用旧索引。"
            "我会先检查 ingestion job 状态，再确认查询服务加载的是不是最新索引。"
        )
        return StopEvent(
            result={
                "answer": answer,
                "trace": ev.trace,
                "used_context": context_text,
            }
        )


def retrieve_contexts(question: str, top_k: int) -> list[str]:
    """用固定文本模拟 Retriever，便于观察 Workflow 的控制逻辑。"""
    knowledge = [
        "E-401 表示权限不足，先确认用户分组和文档 visibility。",
        "E-503 表示索引构建任务未完成，查询服务可能仍在使用旧索引。",
        "索引构建任务完成后，查询服务需要加载最新持久化索引。",
    ]
    if top_k == 1:
        return knowledge[:1]
    return knowledge[:top_k]


async def main() -> None:
    workflow = ControlledRagWorkflow(timeout=30)
    result = await workflow.run(question="E-503 怎么处理？")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
