from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    """业务实体。真实项目里通常来自系统清单、人员表、CMDB 或文档抽取。"""

    name: str
    entity_type: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    """一条图关系，使用 subject -> relation -> target 表达。"""

    subject: str
    relation: str
    target: str
    evidence: str


ENTITIES = [
    Entity("张明", "person", {"team": "LinAI", "role": "检索负责人"}),
    Entity("李雪", "person", {"team": "LinAI", "role": "文档解析负责人"}),
    Entity("王磊", "person", {"team": "LinAI", "role": "Agent 工作台负责人"}),
    Entity("文档解析服务", "service", {"stage": "ingestion"}),
    Entity("检索服务", "service", {"stage": "retrieval"}),
    Entity("Agent 工作台", "service", {"stage": "application"}),
    Entity("PDF 解析", "capability", {"domain": "document"}),
    Entity("表格抽取", "capability", {"domain": "document"}),
    Entity("OCR", "capability", {"domain": "document"}),
    Entity("Retriever", "capability", {"domain": "rag"}),
    Entity("rerank", "capability", {"domain": "rag"}),
    Entity("向量索引加载", "capability", {"domain": "rag"}),
]


RELATIONS = [
    Relation("张明", "负责", "检索服务", "张明负责检索服务。"),
    Relation("李雪", "负责", "文档解析服务", "李雪负责文档解析服务。"),
    Relation("王磊", "负责", "Agent 工作台", "王磊负责 Agent 工作台。"),
    Relation("文档解析服务", "提供能力", "PDF 解析", "文档解析服务负责 PDF 解析。"),
    Relation("文档解析服务", "提供能力", "表格抽取", "文档解析服务负责表格抽取。"),
    Relation("文档解析服务", "提供能力", "OCR", "文档解析服务负责扫描件 OCR。"),
    Relation("检索服务", "提供能力", "Retriever", "检索服务维护 Retriever。"),
    Relation("检索服务", "提供能力", "rerank", "检索服务维护 rerank。"),
    Relation("检索服务", "提供能力", "向量索引加载", "检索服务负责向量索引加载。"),
    Relation("检索服务", "依赖", "文档解析服务", "检索服务依赖文档解析服务提供 Markdown 文本。"),
    Relation("Agent 工作台", "依赖", "检索服务", "Agent 工作台依赖检索服务查询产品文档。"),
    Relation("文档解析服务", "异常影响", "检索服务", "文档解析异常会导致新文档无法进入索引。"),
    Relation("检索服务", "异常影响", "Agent 工作台", "检索异常会影响 Agent 工作台的知识库问答。"),
]


def get_entity(name: str) -> Entity | None:
    """根据实体名称拿到实体信息。"""
    return next((item for item in ENTITIES if item.name == name), None)


def find_related(entity: str, relation_type: str) -> list[Relation]:
    """查询某个实体的一跳关系。"""
    return [
        item
        for item in RELATIONS
        if item.subject == entity and item.relation == relation_type
    ]


def find_reverse_related(entity: str, relation_type: str) -> list[Relation]:
    """反向查询指向某个实体的一跳关系。"""
    return [
        item
        for item in RELATIONS
        if item.target == entity and item.relation == relation_type
    ]


def main() -> None:
    """先用手写关系理解知识图谱到底怎么查。"""
    entity = "Agent 工作台"
    relation_type = "依赖"
    print(f"查询: {entity} {relation_type} 哪些服务")
    for relation in find_related(entity, relation_type):
        print(f"{relation.subject} --{relation.relation}--> {relation.target}")
        print(f"证据: {relation.evidence}")


if __name__ == "__main__":
    main()
