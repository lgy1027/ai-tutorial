from langchain_core.documents import Document

my_document = Document(
    page_content="LangChain是一个用于开发大语言模型应用的框架。",
    metadata={"source": "tutorial_intro", "date": "2025-06-02"}
)
print(my_document)
print(my_document.page_content)
print(my_document.metadata)

# 1. 准备一个文本文件 (先手动创建一个 example.txt)
# 文件内容：
# LangChain是一个开源框架。
# 它的主要目标是帮助开发者构建大语言模型应用。
# RAG是LangChain的重要应用之一。
with open("docs/example.txt", "w", encoding="utf-8") as f:
    f.write("LangChain是一个开源框架。\n")
    f.write("它的主要目标是帮助开发者构建大语言模型应用。\n")
    f.write("RAG是LangChain的重要应用之一。")

from langchain_community.document_loaders import TextLoader

loader = TextLoader("docs/example.txt", encoding="utf-8")
documents = loader.load() # load() 返回一个 Document 列表
print("\n--- TextLoader 示例 ---")
print(f"加载的文档数量: {len(documents)}")
print(f"第一个文档内容:\n{documents[0].page_content}")
print(f"第一个文档元数据:\n{documents[0].metadata}")

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com/langsmith") # LangChain官方博客页

web_documents = loader.load()
print("\n--- WebBaseLoader 示例 ---")
print(f"加载的文档数量: {len(web_documents)}")
if web_documents:
    print(f"第一个网页文档内容 (部分):\n{web_documents[0].page_content[:200]}...")
    print(f"第一个网页文档元数据:\n{web_documents[0].metadata}")

from langchain_community.document_loaders import PyPDFLoader

try:
    pdf_loader = PyPDFLoader("docs/example.pdf") # 替换为你的PDF文件路径
    pdf_documents = pdf_loader.load()
    print("\n--- PyPDFLoader 示例 ---")
    print(f"加载的PDF文档数量 (按页分): {len(pdf_documents)}")
    if pdf_documents:
        print(f"第一个PDF页面内容 (部分):\n{pdf_documents[0].page_content[:200]}...")
        print(f"第一个PDF页面元数据:\n{pdf_documents[0].metadata}")
except FileNotFoundError:
    print("\n--- PyPDFLoader 示例 (跳过): 请放置一个 'sample.pdf' 文件在当前目录 ---")
except Exception as e:
    print(f"\n--- PyPDFLoader 示例 (错误): {e} ---")

from langchain_community.document_loaders import TextLoader

# 假设 example.txt 很大
loader = TextLoader("example.txt", encoding="utf-8")
print("\n--- lazy_load() 示例 ---")
for i, doc in enumerate(loader.lazy_load()):
    print(f"正在处理第 {i+1} 个文档 (内容部分: {doc.page_content[:50]}...)")
    if i >= 1: # 仅处理前2个作为示例
        break
print("懒加载完成。\n")


from langchain.text_splitter import RecursiveCharacterTextSplitter

long_text = """
LangChain 是一个强大的框架，用于开发由大型语言模型驱动的应用程序。
作为一名LangChain教程架构师，我负责设计一套全面、深入且易于理解的LangChain系列教程，
旨在帮助读者从入门到精通，掌握LangChain的核心技术和应用。

RAG（检索增强生成）是LangChain中的一个关键应用场景。
通过RAG，我们可以将LLM与外部知识库相结合，从而让LLM能够回答其训练数据之外的问题。
这大大扩展了LLM的应用范围，解决了幻觉和知识过时的问题。
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # 每个块的最大字符数
    chunk_overlap=20, # 相邻块之间的重叠字符数
    length_function=len, # 使用 Python 的 len() 函数来计算长度
    is_separator_regex=False, # 分隔符不是正则表达式
)

# 可以直接切分字符串
chunks_from_str = text_splitter.split_text(long_text)
print("\n--- RecursiveCharacterTextSplitter 示例 (切分字符串) ---")
for i, chunk in enumerate(chunks_from_str):
    print(f"块 {i+1} (长度 {len(chunk)}):\n'{chunk}'\n")

# 也可以切分 Document 对象列表 (这是更常见的用法)
# 首先创建一个 Document
doc_to_split = Document(page_content=long_text, metadata={"source": "example_doc"})
chunks_from_doc = text_splitter.split_documents([doc_to_split])
print("\n--- RecursiveCharacterTextSplitter 示例 (切分 Document) ---")
for i, chunk_doc in enumerate(chunks_from_doc):
    print(f"块 {i+1} 内容 (长度 {len(chunk_doc.page_content)}):\n'{chunk_doc.page_content}'")
    print(f"块 {i+1} 元数据:\n{chunk_doc.metadata}\n")

from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 将单个文本转换为向量
text1 = "苹果是一种水果"
embedding1 = embeddings_model.embed_query(text1) # embed_query 用于单个文本
print("\n--- OpenAIEmbeddings 示例 ---")
print(f"'{text1}' 的向量长度: {len(embedding1)}")
# print(f"向量:\n{embedding1[:10]}...") # 打印部分向量值

text2 = "香蕉是一种水果"
text3 = "苹果公司生产手机"

embedding2 = embeddings_model.embed_query(text2)
embedding3 = embeddings_model.embed_query(text3)

# 简单计算相似度 (这里只是示意，实际会用向量数据库的相似度计算)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 将列表转换为 numpy 数组进行计算
sim1_2 = cosine_similarity(np.array(embedding1).reshape(1, -1), np.array(embedding2).reshape(1, -1))[0][0]
sim1_3 = cosine_similarity(np.array(embedding1).reshape(1, -1), np.array(embedding3).reshape(1, -1))[0][0]

print(f"'{text1}' 和 '{text2}' 的相似度: {sim1_2:.4f} (语义相似)")
print(f"'{text1}' 和 '{text3}' 的相似度: {sim1_3:.4f} (语义不相似，但词语重合)\n")
# 你会发现 sim1_2 远高于 sim1_3，因为“苹果”和“香蕉”都是水果，语义上更近。

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda

load_dotenv()

# --- 1. 定义数据加载器 ---
loader = TextLoader("example.txt", encoding="utf-8") # 使用我们之前创建的 example.txt

# --- 2. 定义文本切分器 ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

# --- 3. 定义Embedding模型 ---
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# --- 4. 构建LCEL数据准备流水线 ---
# 步骤 A: 加载文档 (loader.load() 返回 List[Document])
# 步骤 B: 切分文档 (text_splitter.split_documents() 接受 List[Document]，返回 List[Document])
# 步骤 C: 提取每个 Document 的 page_content，形成 List[str]
# 步骤 D: 将 List[str] 转换为 List[List[float]] (Embedding 向量)

data_preparation_pipeline = (
    RunnableLambda(lambda x: loader.load()) # A: 加载文档
    | RunnableLambda(lambda docs: text_splitter.split_documents(docs)) # B: 切分文档
    | RunnableLambda(lambda chunks: [chunk.page_content for chunk in chunks]) # C: 提取文本内容
    | RunnableLambda(lambda texts: embeddings_model.embed_documents(texts)) # D: 生成向量
)

print("\n--- LCEL 数据准备流水线示例 ---")
# 运行流水线
# 注意：这里 invoke() 的输入可以为空字典 {}，因为 loader.load() 不依赖外部输入
all_embeddings = data_preparation_pipeline.invoke({})

print(f"生成的块数量: {len(all_embeddings)}")
if all_embeddings:
    print(f"第一个块的向量长度: {len(all_embeddings[0])}")
    # print(f"第一个块的向量 (部分):\n{all_embeddings[0][:10]}...")