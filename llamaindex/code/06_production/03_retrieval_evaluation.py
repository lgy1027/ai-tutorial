import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.evaluation import RetrieverEvaluator

from common import build_example_index, configure_llamaindex


async def main() -> None:
    """实验 3：用 hit_rate 和 mrr 评估 Retriever。"""
    configure_llamaindex()

    index = build_example_index()
    retriever = index.as_retriever(similarity_top_k=2)
    query = "LlamaIndex 的核心组件有哪些？"
    baseline_nodes = retriever.retrieve(query)
    expected_ids = [node.node.node_id for node in baseline_nodes]

    evaluator = RetrieverEvaluator.from_metric_names(
        ["hit_rate", "mrr"],
        retriever=retriever,
    )

    result = await evaluator.aevaluate(
        query=query,
        expected_ids=expected_ids,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
