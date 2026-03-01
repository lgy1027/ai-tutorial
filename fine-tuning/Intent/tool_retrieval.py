# pip install faiss-cpu openai python-dotenv
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化远程 Embedding 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# 模拟工具库
tools = [
    {"name": "order_cancel", "desc": "取消未发货的订单"},
    {"name": "addr_modify", "desc": "修改收货地址和联系电话"},
    {"name": "tax_query", "desc": "查询个人所得税缴纳记录"}
]

# 构建简单向量索引
descriptions = [t['desc'] for t in tools]
encoded_data = np.array(get_embeddings(descriptions), dtype=np.float32)
index = faiss.IndexFlatL2(encoded_data.shape[1])
index.add(encoded_data)


def retrieve_tools(query, top_k=1):
    query_vector = np.array(get_embeddings([query]), dtype=np.float32)
    D, I = index.search(query_vector, top_k)
    return [tools[i] for i in I[0]]


# 测试
print(retrieve_tools("我想换个收货地方"))  # 召回 addr_modify