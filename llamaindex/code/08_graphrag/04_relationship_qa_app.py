import importlib.util
import json
from collections import deque
from pathlib import Path


MANUAL_PATH = Path(__file__).with_name("03_manual_relationship_baseline.py")
spec = importlib.util.spec_from_file_location("manual_relationship", MANUAL_PATH)
manual = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(manual)


def find_owner(entity: str) -> list[dict[str, str]]:
    """反向查询谁负责某个实体。"""
    owners = []
    for item in manual.find_reverse_related(entity, "负责"):
        owner = manual.get_entity(item.subject)
        owners.append(
            {
                "name": item.subject,
                "role": owner.properties.get("role", "未记录") if owner else "未记录",
                "evidence": item.evidence,
            }
        )
    return owners


def trace_relation_path(
    start_entity: str,
    relation_type: str,
    max_depth: int = 2,
) -> list[dict[str, object]]:
    """沿着某类关系做广度优先遍历，返回可解释的路径。"""
    result = []
    queue = deque([(start_entity, 0)])
    visited = set()

    while queue:
        current, depth = queue.popleft()
        if (current, depth) in visited or depth >= max_depth:
            continue
        visited.add((current, depth))

        for relation in manual.find_related(current, relation_type):
            owners = find_owner(relation.target)
            result.append(
                {
                    "depth": depth + 1,
                    "source": relation.subject,
                    "relation": relation.relation,
                    "target": relation.target,
                    "target_type": (
                        manual.get_entity(relation.target).entity_type
                        if manual.get_entity(relation.target)
                        else "unknown"
                    ),
                    "target_owners": owners,
                    "evidence": relation.evidence,
                }
            )
            queue.append((relation.target, depth + 1))
    return result


def list_capabilities(service: str) -> list[dict[str, str]]:
    """查询某个服务提供哪些能力。"""
    capabilities = []
    for relation in manual.find_related(service, "提供能力"):
        entity = manual.get_entity(relation.target)
        capabilities.append(
            {
                "capability": relation.target,
                "domain": entity.properties.get("domain", "未记录") if entity else "未记录",
                "evidence": relation.evidence,
            }
        )
    return capabilities


def answer_dependency_question(entity: str) -> dict[str, object]:
    """回答“某个系统依赖谁”。"""
    relations = trace_relation_path(entity, "依赖", max_depth=3)
    chain = " -> ".join([entity] + [item["target"] for item in relations])
    return {
        "question": f"{entity} 依赖哪些服务？",
        "answer": f"{entity} 的依赖链路是：{chain}。" if relations else "没有找到依赖关系。",
        "relations": relations,
    }


def answer_impact_question(entity: str) -> dict[str, object]:
    """回答“某个系统异常会影响谁”。"""
    relations = trace_relation_path(entity, "异常影响", max_depth=3)
    impacted = [item["target"] for item in relations]
    return {
        "question": f"{entity} 异常会影响哪些模块？",
        "answer": f"{entity} 异常会影响：{', '.join(impacted)}。" if impacted else "没有找到影响链路。",
        "relations": relations,
    }


def answer_service_profile(service: str) -> dict[str, object]:
    """把负责人、能力、依赖合在一起，形成一个最小服务画像。"""
    return {
        "service": service,
        "owners": find_owner(service),
        "capabilities": list_capabilities(service),
        "dependencies": trace_relation_path(service, "依赖", max_depth=2),
        "impacts": trace_relation_path(service, "异常影响", max_depth=2),
    }


def main() -> None:
    outputs = {
        "dependency": answer_dependency_question("Agent 工作台"),
        "impact": answer_impact_question("文档解析服务"),
        "service_profile": answer_service_profile("检索服务"),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
