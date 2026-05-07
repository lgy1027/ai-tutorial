import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)

from common import Settings, configure_llamaindex, load_example_documents


def show_nodes(name: str, nodes) -> None:
    print(f"\n=== {name} ===")
    print("nodes:", len(nodes))
    for index, node in enumerate(nodes[:3], start=1):
        print(f"\n--- Node {index} ---")
        print("metadata:", node.metadata)
        print(node.get_content()[:260].replace("\n", " "))


def main() -> None:
    """示例 5：对比不同文档切块策略。"""
    configure_llamaindex()

    documents = load_example_documents()

    splitters = {
        "SentenceSplitter": SentenceSplitter(chunk_size=160, chunk_overlap=30),
        "TokenTextSplitter": TokenTextSplitter(chunk_size=160, chunk_overlap=30),
        "SentenceWindowNodeParser": SentenceWindowNodeParser.from_defaults(
            window_size=2,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        ),
        "SemanticSplitterNodeParser": SemanticSplitterNodeParser.from_defaults(
            embed_model=Settings.embed_model,
            breakpoint_percentile_threshold=95,
        ),
    }

    for name, splitter in splitters.items():
        nodes = splitter.get_nodes_from_documents(documents)
        show_nodes(name, nodes)


if __name__ == "__main__":
    main()
