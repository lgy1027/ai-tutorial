import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document, VectorStoreIndex

from common import configure_llamaindex


MANUAL_PATH = Path(__file__).with_name("03_manual_relationship_baseline.py")
QA_PATH = Path(__file__).with_name("04_relationship_qa_app.py")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


manual = load_module(MANUAL_PATH, "manual_relationship")
qa_app = load_module(QA_PATH, "relationship_qa_app")


def build_evidence_documents() -> list[Document]:
    """把关系证据和服务说明组织成可向量检索的文本。"""
    service_notes = [
        Document(
            text=(
                "Agent 工作台负责面向用户的知识库问答入口。它依赖检索服务查询产品文档、"
                "故障复盘和运维策略。如果检索服务不可用，Agent 工作台仍可创建人工工单，"
                "但不能稳定回答知识库问题。"
            ),
            metadata={"source": "service_notes", "entity": "Agent 工作台"},
        ),
        Document(
            text=(
                "检索服务负责 Retriever、rerank 和向量索引加载。它依赖文档解析服务提供"
                "清洗后的 Markdown 文本。索引构建任务未完成时，查询服务可能只能使用旧索引。"
            ),
            metadata={"source": "service_notes", "entity": "检索服务"},
        ),
        Document(
            text=(
                "文档解析服务负责 PDF 解析、表格抽取和扫描件 OCR。文档解析服务异常时，"
                "新文档无法进入索引，后续会影响检索服务和 Agent 工作台的问答质量。"
            ),
            metadata={"source": "service_notes", "entity": "文档解析服务"},
        ),
    ]

    relation_docs = [
        Document(
            text=relation.evidence,
            metadata={
                "source": "graph_relation_evidence",
                "subject": relation.subject,
                "relation": relation.relation,
                "target": relation.target,
            },
        )
        for relation in manual.RELATIONS
    ]
    return service_notes + relation_docs


def build_vector_retriever(documents: list[Document]):
    """构建向量索引，只负责召回文本证据。"""
    configure_llamaindex()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=5)


def retrieve_text_evidence(question: str) -> list[dict[str, object]]:
    """用向量检索找和问题相关的原文证据。"""
    documents = build_evidence_documents()
    retriever = build_vector_retriever(documents)
    nodes = retriever.retrieve(question)
    evidence = []
    for node in nodes:
        text = node.node.get_content()
        evidence.append(
            {
                "vector_score": node.score,
                "text": text,
                "metadata": node.node.metadata,
            }
        )
    return evidence


def route_question(question: str) -> str:
    """根据问题类型选择检索路径：vector、graph 或 graph_vector。"""
    relation_terms = ["依赖", "影响", "负责", "负责人", "谁", "哪些服务", "哪些模块"]
    evidence_terms = ["为什么", "依据", "证据", "原文", "说明", "解释", "会影响它吗"]

    has_relation_intent = any(term in question for term in relation_terms)
    needs_evidence = any(term in question for term in evidence_terms)

    if has_relation_intent and needs_evidence:
        return "graph_vector"
    if has_relation_intent:
        return "graph"
    return "vector"


def answer_with_vector(question: str) -> dict[str, object]:
    """只需要查说明或原文时，直接走向量检索。"""
    text_evidence = retrieve_text_evidence(question)
    top_text = text_evidence[0]["text"] if text_evidence else "没有找到相关证据。"
    return {
        "question": question,
        "route": "vector",
        "vector_evidence": text_evidence,
        "final_answer": top_text,
    }


def answer_with_graph(question: str) -> dict[str, object]:
    """只需要查结构关系时，直接走图谱。"""
    dependency = qa_app.answer_dependency_question("Agent 工作台")
    impact = qa_app.answer_impact_question("文档解析服务")
    service_profile = qa_app.answer_service_profile("检索服务")
    return {
        "question": question,
        "route": "graph",
        "graph_answer": {
            "dependency_chain": dependency["answer"],
            "impact_chain": impact["answer"],
            "service_owner": service_profile["owners"],
            "service_capabilities": service_profile["capabilities"],
        },
        "final_answer": (
            f"{dependency['answer']} {impact['answer']} "
            "检索服务负责人和能力可以在 service_owner、service_capabilities 中查看。"
        ),
    }


def answer_with_graph_and_vector(question: str) -> dict[str, object]:
    """图谱负责结构，向量检索负责证据，最后合成一个可审计答案。"""
    dependency = qa_app.answer_dependency_question("Agent 工作台")
    impact = qa_app.answer_impact_question("文档解析服务")
    service_profile = qa_app.answer_service_profile("检索服务")
    text_evidence = retrieve_text_evidence(question)

    return {
        "question": question,
        "route": "graph_vector",
        "graph_answer": {
            "dependency_chain": dependency["answer"],
            "impact_chain": impact["answer"],
            "service_owner": service_profile["owners"],
            "service_capabilities": service_profile["capabilities"],
        },
        "vector_evidence": text_evidence,
        "final_answer": (
            "Agent 工作台依赖检索服务，检索服务继续依赖文档解析服务。"
            "因此文档解析服务异常时，新文档无法进入索引，会先影响检索服务，"
            "再影响 Agent 工作台的知识库问答。检索服务负责人是张明，"
            "主要维护 Retriever、rerank 和向量索引加载。"
        ),
    }


def answer_question(question: str) -> dict[str, object]:
    """统一入口：先路由，再执行对应检索链路。"""
    route = route_question(question)
    if route == "graph_vector":
        return answer_with_graph_and_vector(question)
    if route == "graph":
        return answer_with_graph(question)
    return answer_with_vector(question)


def main() -> None:
    question = "Agent 工作台依赖哪些服务？文档解析服务异常会影响它吗？"
    try:
        result = answer_question(question)
    except Exception as exc:
        print("运行失败：请先检查 llamaindex/.env 中的模型和 embedding 接口配置。")
        print("本示例会真实调用 OpenAI 兼容 embedding 接口来构建 VectorStoreIndex。")
        print(f"错误类型: {type(exc).__name__}")
        print(f"错误信息: {exc}")
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
