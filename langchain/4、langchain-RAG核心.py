from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL"),
    api_key=os.environ.get("EMBEDDING_API_KEY"),
    base_url=os.environ.get("EMBEDDING_BASE_URL"),
)

# 准备数据 (与第三期类似)
with open("docs/example.txt", "w", encoding="utf-8") as f:
    f.write("LangChain 是一个强大的框架，用于开发由大型语言模型驱动的应用程序。\n")
    f.write("作为一名LangChain教程架构师，我负责设计一套全面、深入且易于理解的LangChain系列教程。\n")
    f.write("旨在帮助读者从入门到精通，掌握LangChain的核心技术和应用。\n")
    f.write("RAG（检索增强生成）是LangChain中的一个关键应用场景。\n")
    f.write("通过RAG，我们可以将LLM与外部知识库相结合。\n")
    f.write("从而让LLM能够回答其训练数据之外的问题。\n")
    f.write("这大大扩展了LLM的应用范围，解决了幻觉和知识过时的问题。\n")
    f.write("LangSmith 是 LangChain 的一个强大工具，用于调试和评估 LLM 应用程序。\n")
    f.write("LCEL 是 LangChain Expression Language 的简称，是构建链条的首选方式。\n")
    f.write("LangGraph 则用于构建具有循环和复杂状态的 Agent。\n")


loader = TextLoader("docs/example.txt", encoding="utf-8")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

# 步骤 1: 加载和切分文档，得到 Document 块列表
raw_documents = loader.load()
split_documents = text_splitter.split_documents(raw_documents)

print(f"原始文档切分后得到 {len(split_documents)} 个块。")
# print(f"第一个块内容: {split_documents[0].page_content}")

# --- 2. 创建并持久化 Chroma 向量数据库 ---
# from_documents 方法会同时进行 embedding 和存储
# persist_directory 参数用于指定存储路径，这样数据就会被保存到磁盘上，下次可以直接加载
persist_directory = "./chroma_db"
# 如果目录已存在，可以先清理 (仅用于测试)
import shutil
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

print(f"正在创建或加载 Chroma 数据库到 '{persist_directory}'...")
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings_model,
    persist_directory=persist_directory
)
print("Chroma 数据库创建/加载完成并已持久化。")

# --- 3. 进行相似度检索 (直接使用向量数据库的相似度搜索方法) ---
query = "LangChain是用来做什么的？"
# similarity_search 方法会向量化查询，并在数据库中搜索最相似的文档
found_docs = vectorstore.similarity_search(query, k=2) # k=2 表示返回最相似的2个文档

print(f"\n--- 对查询 '{query}' 的检索结果 (Chroma.similarity_search) ---")
for i, doc in enumerate(found_docs):
    print(f"文档 {i+1} (来源: {doc.metadata.get('source', 'N/A')}, 页码: {doc.metadata.get('page', 'N/A')}):")
    print(doc.page_content)
    print("-" * 30)

# --- 4. 从持久化路径加载数据库 (下次运行时可以直接加载，无需重新创建) ---
print(f"\n--- 从持久化路径重新加载 Chroma 数据库 ---")
loaded_vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_model # 注意：加载时也要指定embedding_function
)
reloaded_found_docs = loaded_vectorstore.similarity_search(query, k=1)
print(f"重新加载后检索结果 (部分): {reloaded_found_docs[0].page_content[:100]}...")


# 假设我们给文档添加了更多元数据，这里修改 example.txt 并重新加载
# 为了简化，我们直接修改 split_documents，为其中一些添加 category
if len(split_documents) > 2:
    split_documents[0].metadata["category"] = "LangChain Core"
    split_documents[1].metadata["category"] = "LCEL"
    split_documents[2].metadata["category"] = "RAG"
    split_documents[3].metadata["category"] = "RAG"

persist_directory="./chroma_db_with_meta"

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# 重新创建向量库 (或加载后，如果需要更新则需要重新添加)
vectorstore_with_meta = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings_model,
    persist_directory=persist_directory
)

# 使用元数据过滤进行检索
query_filtered = "LangChain的关键概念"
# filter 参数接受一个字典，定义过滤条件
# "$eq" 表示等于 (equal)，"$in" 表示包含在列表中 (in)
found_docs_filtered = vectorstore_with_meta.similarity_search(
    query_filtered, 
    k=3, 
    filter={"category": "LangChain Core"} # 仅搜索 category 为 "LangChain Core" 的文档
)

print(f"\n--- 对查询 '{query_filtered}' 的元数据过滤检索结果 (category = 'LangChain Core') ---")
for i, doc in enumerate(found_docs_filtered):
    print(f"文档 {i+1} (分类: {doc.metadata.get('category', 'N/A')}):")
    print(doc.page_content)
    print("-" * 30)

# 复杂过滤条件
found_docs_complex_filter = vectorstore_with_meta.similarity_search(
    query_filtered,
    k=3,
    filter={"$or": [{"category": "LangChain Core"}, {"category": "LCEL"}]} # 或关系
)

print(f"\n--- 对查询 '{query_filtered}' 的元数据过滤检索结果 (category = 'LangChain Core' or ''LCEL) ---")
for i, doc in enumerate(found_docs_complex_filter):
    print(f"文档 {i+1} (分类: {doc.metadata.get('category', 'N/A')}):")
    print(doc.page_content)
    print("-" * 30)
# 更多过滤条件请参考 Chroma 文档

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 从我们上面创建的 vectorstore (或 loaded_vectorstore) 获取一个 retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # search_kwargs 可以传递给底层的 similarity_search

# 构建一个基础的 RAG 链 (使用 LCEL)
# 输入 {"question": "...", "context": "...(retrieved docs)..."}
rag_prompt = ChatPromptTemplate.from_template("""
请根据提供的上下文回答以下问题。
如果上下文中没有足够的信息，请说明你不知道。

问题: {question}

上下文:
{context}
""")

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.9,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)
# 核心 LCEL RAG 链
# 1. 接收 {question} 作为输入
# 2. RunnableParallel 会并行处理 "context" (通过 retriever 获取) 和 "question" (直接透传)
# 3. 结果合并为 {"context": List[Document], "question": str}
# 4. prompt 接收这个字典，格式化
# 5. LLM 生成答案
# 6. OutputParser 解析
basic_rag_chain = (
    {"context": base_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n--- 基础 RAG 链示例 (使用 VectorstoreRetriever) ---")
query_basic = "LangChain是什么？"
response_basic = basic_rag_chain.invoke(query_basic)
print(f"问题: {query_basic}")
print(f"回答: {response_basic}")


from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # 内存存储，也可以用 Redis, MongoDB 等
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 定义大块和小块切分器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) # 大块
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20) # 小块 (用于检索)

persist_directory="./chroma_db_child_chunks"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# 2. 定义向量存储 (用于小块) 和文档存储 (用于大块，通过ID映射)
vectorstore_for_child = Chroma(
    embedding_function=embeddings_model,
    persist_directory=persist_directory
)

doc_store = InMemoryStore() # 存储大块原始文档

# 3. 创建 ParentDocumentRetriever
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore_for_child,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 4. 添加文档
# 注意：这里直接 add_documents，ParentDocumentRetriever 会自动处理切分和存储
print("\n--- ParentDocumentRetriever 示例 (添加文档) ---")
parent_document_retriever.add_documents(raw_documents) # raw_documents 是未切分的原始文档
print("文档已添加到 ParentDocumentRetriever。")

# 5. 进行检索
query_parent = "什么是LangGraph，它和LCEL有什么区别？"
retrieved_parent_docs = parent_document_retriever.invoke(query_parent)

print(f"\n--- 对查询 '{query_parent}' 的 ParentDocumentRetriever 检索结果 ---")
for i, doc in enumerate(retrieved_parent_docs):
    print(f"文档 {i+1} (长度: {len(doc.page_content)}):")
    print(doc.page_content) # 打印部分内容，通常会比小块大
    print("-" * 30)

# 你会发现这里的文档块长度更长，因为它们是从大块中取出的，包含了更多上下文。


from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 定义我们的文档元数据信息
# 这是关键！让LLM知道有哪些元数据字段以及它们的含义
document_content_description = "关于LangChain框架、RAG、LangSmith、LCEL和LangGraph的文档。"
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="文档内容所属的类别，例如 'LangChain Core', 'RAG', 'LangSmith', 'LCEL', 'LangGraph'。",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="文档的来源文件名称，例如 'example.txt'。",
        type="string",
    ),
]

# 从我们之前创建的 vectorstore_with_meta 获取 SelfQueryRetriever
# 注意：SelfQueryRetriever 需要一个 LLM 来解析查询
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore_with_meta, # 带有元数据的向量库
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True # 开启 verbose 可以看到LLM生成的查询
)

print("\n--- SelfQueryRetriever 示例 ---")
query_self = "关于RAG的关键应用，只从RAG相关的文档中搜索。"
retrieved_self_docs = self_query_retriever.invoke(query_self)

print(f"对查询 '{query_self}' 的 SelfQueryRetriever 检索结果:")
for i, doc in enumerate(retrieved_self_docs):
    print(f"文档 {i+1} (分类: {doc.metadata.get('category', 'N/A')}):")
    print(doc.page_content)
    print("-" * 30)

query_self_2 = "LangGraph是什么？它的来源文件是哪个？" # 假设我们知道是 example.txt
retrieved_self_docs_2 = self_query_retriever.invoke(query_self_2)
print(f"\n对查询 '{query_self_2}' 的 SelfQueryRetriever 检索结果:")
for i, doc in enumerate(retrieved_self_docs_2):
    print(f"文档 {i+1} (分类: {doc.metadata.get('category', 'N/A')}, 来源: {doc.metadata.get('source', 'N/A')}):")
    print(doc.page_content)
    print("-" * 30)