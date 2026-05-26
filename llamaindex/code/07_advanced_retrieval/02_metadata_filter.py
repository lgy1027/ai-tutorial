import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from common import configure_llamaindex, print_nodes


def build_permission_documents() -> list[Document]:
    """准备一组带 visibility 的文档，用来模拟权限边界。"""
    return [
        Document(
            text="E-401 表示普通用户没有访问当前文档分组的权限。",
            metadata={"team": "support", "visibility": "internal"},
        ),
        Document(
            text="E-401 对外只需要解释为权限不足，请联系管理员确认账号分组。",
            metadata={"team": "support", "visibility": "public"},
        ),
        Document(
            text="E-503 表示索引构建任务未完成，查询服务可能仍在使用旧索引。",
            metadata={"team": "platform", "visibility": "internal"},
        ),
    ]


def build_public_filter() -> MetadataFilters:
    """只允许公开文档进入检索结果。"""
    return MetadataFilters(filters=[ExactMatchFilter(key="visibility", value="public")])


def main() -> None:
    """进阶 1：用 metadata filter 先排除不该看的文档。"""
    configure_llamaindex()

    documents = build_permission_documents()
    index = VectorStoreIndex.from_documents(documents)

    print("\n=== Without metadata filter ===")
    open_retriever = index.as_retriever(similarity_top_k=3)
    print_nodes(open_retriever.retrieve("E-401 是什么意思？"))

    print("\n=== With visibility=public filter ===")
    filtered_retriever = index.as_retriever(
        similarity_top_k=3,
        filters=build_public_filter(),
    )
    print_nodes(filtered_retriever.retrieve("E-401 是什么意思？"))


if __name__ == "__main__":
    main()
