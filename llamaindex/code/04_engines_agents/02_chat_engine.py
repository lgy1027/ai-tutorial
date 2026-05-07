import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_example_index, configure_llamaindex


def main() -> None:
    """实验 2：ChatEngine 适合围绕同一批资料连续追问。"""
    configure_llamaindex()

    index = build_example_index()
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=2,
    )

    print(chat_engine.chat("RAG 系统生产化要先看什么？"))
    print("\n--- follow up ---")
    print(chat_engine.chat("把刚才的回答整理成三点。"))


if __name__ == "__main__":
    main()
