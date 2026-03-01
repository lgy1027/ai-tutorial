# # pip install semantic-router python-dotenv
# # 向量语义路由
import os
from typing import Any, List, Optional
from dotenv import load_dotenv
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import DenseEncoder
from openai import OpenAI
import tiktoken

load_dotenv()


class CustomOpenAIEncoder(DenseEncoder):
    """
    自定义 OpenAI Encoder，支持任意远程 embedding API。
    绕过 tiktoken 检查，可以使用任意模型名称。
    """
    
    type: str = "custom_openai"
    
    client: Any = None
    
    def __init__(
        self,
        name: str = "BAAI/bge-m3",
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=openai_api_key,
        )
        # 使用 cl100k_base 编码器（通用的）绕过 tiktoken 模型名检查
        self._token_encoder = tiktoken.get_encoding("cl100k_base")
    
    def __call__(self, docs: List[str]) -> List[List[float]]:
        """将文本列表转换为嵌入向量列表"""
        response = self.client.embeddings.create(
            model=self.name,
            input=docs,
        )
        return [item.embedding for item in response.data]


# 定义意图路由及其典型话术
politics_route = Route(
    name="politics",
    utterances=[
        "你对现在的政治局势怎么看？", 
        "给我讲讲最近的国际新闻吧", 
        "关于那个政策你有什么想法？"
    ],
)

chitchat_route = Route(
    name="chitchat",
    utterances=[
        "今天天气真不错啊", 
        "你吃饭了吗？", 
        "周末打算去哪里玩？",
        "最近有什么好看的电影推荐吗？"
    ],
)

# 初始化编码器，将文本转为向量
encoder = CustomOpenAIEncoder(
    name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    openai_base_url=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# 构建路由层 (默认使用本地向量检索)
router = SemanticRouter(
    encoder=encoder, 
    routes=[politics_route, chitchat_route],
    auto_sync="local"  # 自动同步本地索引
)

# 执行意图识别
user_query = "伊朗和美国的局势"
route_choice = router(user_query)

if route_choice:
    print(f"识别到的意图: {route_choice.name}") 
    # 输出: 识别到的意图: chitchat
else:
    print("未匹配到任何意图")