import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT_DIR


def main() -> None:
    """进阶 3：LlamaParse/LlamaCloud 可选示例，需要额外安装和配置。"""
    try:
        from llama_cloud import LlamaCloud
    except ImportError:
        print("当前环境没有安装 llama-cloud。")
        print("如需运行本示例，可以执行: pip install llama-cloud")
        return

    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("当前环境没有配置 LLAMA_CLOUD_API_KEY。")
        print("LlamaParse/LlamaCloud 需要官方 API Key，配置后再运行本示例。")
        return

    file_path = ROOT_DIR / "data" / "advanced_docs" / "sample.pdf"
    if not file_path.exists():
        print(f"没有找到示例文件: {file_path}")
        print("可以把要测试的 PDF、Word、PPT 或其他受支持文件放到这个路径，或修改脚本里的 file_path。")
        return

    client = LlamaCloud()
    uploaded_file = client.files.create(file=str(file_path), purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded_file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    first_page = result.markdown.pages[0].markdown
    print(first_page[:1000])


if __name__ == "__main__":
    main()
