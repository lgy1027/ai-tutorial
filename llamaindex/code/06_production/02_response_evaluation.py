import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

from common import Settings, build_example_index, configure_llamaindex


async def main() -> None:
    """实验 2：用 Faithfulness 和 Relevancy 检查回答质量。"""
    configure_llamaindex()

    index = build_example_index()
    query_engine = index.as_query_engine(similarity_top_k=2)

    question = "LlamaIndex 的 Retriever 负责什么？"
    response = query_engine.query(question)
    contexts = [node.node.get_content() for node in response.source_nodes]

    faithfulness = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy = RelevancyEvaluator(llm=Settings.llm)

    faithfulness_result = await faithfulness.aevaluate_response(response=response)
    relevancy_result = await relevancy.aevaluate(
        query=question,
        response=str(response),
        contexts=contexts,
    )

    print("=== Response ===")
    print(response)
    print("\n=== Faithfulness ===")
    print(faithfulness_result)
    print("\n=== Relevancy ===")
    print(relevancy_result)


if __name__ == "__main__":
    asyncio.run(main())
