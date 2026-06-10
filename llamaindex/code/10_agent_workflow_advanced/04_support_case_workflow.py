import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path


WORKFLOW_PATH = Path(__file__).with_name("03_workflow_retry.py")
spec = importlib.util.spec_from_file_location("workflow_retry", WORKFLOW_PATH)
workflow_retry = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(workflow_retry)


def evaluate_next_action(question: str, answer: str) -> dict[str, object]:
    """根据问题和回答决定下一步动作，并保留可审计原因。"""
    reasons: list[str] = []
    if "工单" in question or "复核" in question:
        reasons.append("user_requested_review")
    if "旧索引" in answer:
        reasons.append("answer_mentions_stale_index")

    if reasons:
        return {
            "next_action": "create_review_ticket",
            "risk_level": "medium",
            "reasons": reasons,
        }
    return {
        "next_action": "reply_only",
        "risk_level": "low",
        "reasons": ["answer_is_read_only"],
    }


def build_review_ticket(question: str, answer: str, reasons: list[str]) -> dict[str, object]:
    """构造外部动作载荷；示例只返回 payload，不真正调用工单系统。"""
    digest = hashlib.sha1(f"{question}|{answer}".encode("utf-8")).hexdigest()[:10]
    return {
        "ticket_id": f"review-{digest}",
        "idempotency_key": digest,
        "title": "知识库回答需要复核",
        "summary": answer,
        "reasons": reasons,
    }


async def run_support_case(question: str) -> dict[str, object]:
    """把受控 RAG 和外部动作判断串成一个客服处理闭环。"""
    workflow = workflow_retry.ControlledRagWorkflow(timeout=30)
    rag_result = await workflow.run(question=question)
    answer = rag_result["answer"]
    action = evaluate_next_action(question, answer)
    reasons = list(action["reasons"])
    ticket_payload = (
        build_review_ticket(question, answer, reasons)
        if action["next_action"] == "create_review_ticket"
        else None
    )
    return {
        "question": question,
        "answer": answer,
        "trace": rag_result["trace"],
        "used_context": rag_result["used_context"],
        "next_action": action["next_action"],
        "risk_level": action["risk_level"],
        "action_reasons": reasons,
        "action_payload": ticket_payload,
        "audit_log": {
            "workflow": "ControlledRagWorkflow",
            "trace": rag_result["trace"],
            "action_decider": "evaluate_next_action:v1",
        },
    }


def main() -> None:
    result = asyncio.run(run_support_case("E-503 怎么处理？需要复核工单。"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
