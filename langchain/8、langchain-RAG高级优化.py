from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv; load_dotenv()
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

embeddings_model = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL"),
        base_url=os.environ.get("EMBEDDING_BASE_URL"),
        openai_api_key=os.environ.get("EMBEDDING_API_KEY"),
    )
# --- 2. 准备数据和向量数据库 (复用并简化之前的代码) ---
# 确保 example.txt 存在
with open("example.txt", "w", encoding="utf-8") as f:
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

loader = TextLoader("example.txt", encoding="utf-8")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
raw_documents = loader.load()
split_documents = text_splitter.split_documents(raw_documents)

persist_directory = "./chroma_db_rag_basic"
# 创建并持久化 Chroma 向量数据库 (如果已存在则加载)
persist_directory = "./chroma_db_rag_basic"
if not os.path.exists(persist_directory) or not Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)._collection.count():
    print(f"正在创建并持久化 Chroma 数据库到 '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    print("Chroma 数据库创建/加载完成。")
else:
    print(f"从 '{persist_directory}' 加载现有 Chroma 数据库...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_model
    )
    print("Chroma 数据库加载完成。")

# 2. 创建 MultiQueryRetriever
# 它需要一个 LLM 来生成多个查询
# multiquery_retriever = MultiQueryRetriever.from_llm(
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), # 基础检索器
#     llm=llm, # 用于生成新查询的LLM
#     # prompt=ChatPromptTemplate.from_template("Generate 3 different ways to ask the question: {question}"), # 也可以自定义生成查询的Prompt
#     # parser_key="text", # LLM输出中包含查询的键
# )

# print("\n--- MultiQueryRetriever 示例 ---")
# query_mq = "LangChain的优势是什么？"
# retrieved_docs_mq = multiquery_retriever.invoke(query_mq)

# print(f"对查询 '{query_mq}' 的 MultiQueryRetriever 检索结果 ({len(retrieved_docs_mq)} 个文档):")
# for i, doc in enumerate(retrieved_docs_mq):
#     print(f"文档 {i+1} (来源: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content}")
#     print("-" * 30)

# # 1. 初始化关键词检索器
# bm25_retriever = BM25Retriever.from_documents(split_documents)
# bm25_retriever.k = 3

# # 2. 初始化向量检索器
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # 3. 初始化 EnsembleRetriever
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5] # 可以给不同检索器设置不同权重
# )

# # 4. 使用
# query = "LangChain中的LCEL是什么?"
# retrieved_docs = ensemble_retriever.invoke(query)
# print(f"混合搜索召回了 {len(retrieved_docs)} 个文档。")


# 示例: 使用Flashrank 进行重排序
# from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain.retrievers import ContextualCompressionRetriever

# # 1. 定义一个重排序压缩器
# # top_n 是重排序后返回多少个文档
# rerank_compressor = FlashrankRerank(
#     model="miniReranker_arabic_v1",
#     top_n=3
# )
# # 2. 创建一个 ContextualCompressionRetriever, 使用 Flashrank 作为压缩器
# rerank_retriever = ContextualCompressionRetriever(
#     base_compressor=rerank_compressor,
#     base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}) # 先检索5个再重排
# )

# # 3. 使用
# print("\n--- Reranking (Flashrank) 示例 ---")
# query_rerank = "LangChain的最新功能是什么?"
# retrieved_reranked_docs = rerank_retriever.invoke(query_rerank)
# print(f"对查询 '{query_rerank}' 的重排序检索结果({len(retrieved_reranked_docs)} 个文档):")
# for i, doc in enumerate(retrieved_reranked_docs):
#     print(f"文档 {i+1} (分数: {doc.metadata.get('relevance_score', 'N/A')}):\n{doc.page_content[:100]}...")
# print("-" * 30)

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain_openai import ChatOpenAI

# # 1. 定义一个基础检索器(先多检索一些,再压缩)
# base_retriever_for_comp = vectorstore.as_retriever(search_kwargs={"k": 5})

# # 2. 定义一个 LLMChainExtractor (压缩器)
# compressor = LLMChainExtractor.from_llm(llm=llm)

# # 3. 创建 ContextualCompressionRetriever
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=base_retriever_for_comp
# )

# # 4. 使用
# print("\n--- ContextualCompressionRetriever 示例 ---")
# query_comp = "LangChain的调试工具叫什么?它的主要作用是什么?"
# retrieved_compressed_docs = compression_retriever.invoke(query_comp)
# print(f"对查询 '{query_comp}' 的ContextualCompressionRetriever 检索结果:")
# for i, doc in enumerate(retrieved_compressed_docs):
#     # LLMChainExtractor 在压缩过程中会将原始文档内容存储在 metadata 的 'original_content' 字段中
#     original_len = len(doc.metadata.get('original_content', doc.page_content))
#     compressed_len = len(doc.page_content)
#     print(f"文档 {i+1}(原始长度: {original_len}, 压缩后长度: {compressed_len}):")
#     print(doc.page_content)
#     print("-" * 30)


# def find_elbow_point(scores: np.ndarray) -> int:
#     """
#     使用点到直线最大距离的纯几何方法。
#     返回的是拐点在原始列表中的索引。
#     """
#     n_points = len(scores)
#     if n_points < 3:
#         return n_points -1 # 返回最后一个点的索引

#     # 创建点坐标 (x, y)，x是索引，y是分数
#     points = np.column_stack((np.arange(n_points), scores))

#     # 获取第一个点和最后一个点
#     first_point = points[0]
#     last_point = points[-1]

#     # 计算每个点到首末点连线的垂直距离
#     # 使用向量射影的方法
#     line_vec = last_point - first_point
#     line_vec_normalized = line_vec / np.linalg.norm(line_vec)
    
#     vec_from_first = points - first_point
    
#     # scalar_product 是每个点向量在直线方向上的投影长度
#     scalar_product = np.dot(vec_from_first, line_vec_normalized)
    
#     # vec_parallel 是投影向量
#     vec_parallel = np.outer(scalar_product, line_vec_normalized)
    
#     # vec_perpendicular 是垂直向量，它的模长就是距离
#     vec_perpendicular = vec_from_first - vec_parallel
    
#     dist_to_line = np.linalg.norm(vec_perpendicular, axis=1)

#     # 找到距离最大的点的索引
#     elbow_index = np.argmax(dist_to_line)
#     return elbow_index

# def truncate_with_elbow_method_final(
#     reranked_docs: List[Tuple[float, Document]]
# ) -> List[Document]:
#     if not reranked_docs or len(reranked_docs) < 3:
#         print("文档数量不足3个，无法进行拐点检测，返回所有文档。")
#         return [doc for _, doc in reranked_docs]

#     scores = np.array([score for score, _ in reranked_docs])
#     docs = [doc for _, doc in reranked_docs]
    
#     # 调用我们验证过有效的拐点检测函数
#     elbow_index = find_elbow_point(scores)
    
#     # 我们需要包含拐点本身，所以截取到 elbow_index + 1
#     num_docs_to_keep = elbow_index + 1
#     final_docs = docs[:num_docs_to_keep]
    
#     print(f"检测到分数拐点在第 {elbow_index + 1} 位。截断后返回 {len(final_docs)} 个文档。")
#     return final_docs

# print("\n--- 拐点检测示例 ---")
# # 假设 reranked_docs 是你的输入数据
# reranked_docs = [
#     (0.98, "文档1"),
#     (0.95, "文档2"),
#     (0.92, "文档3"),
#     (0.75, "文档4"),
#     (0.5, "文档5"),
#     (0.48, "文档6")
# ]
# final_documents = truncate_with_elbow_method_final(reranked_docs)
# print(final_documents)


# 方法一：将QA对作为知识库直接索引
# --- 1. 数据准备：高质量的QA对 ---
# qa_pairs = [
#     {
#         "question": "LangChain的LCEL是什么，它有什么用？",
#         "answer": "LCEL，全称LangChain Expression Language，是一种用于声明式地链式组合AI组件的语言。它简化了复杂链的构建，并原生支持流式处理、异步和并行执行等高级功能。"
#     },
#     {
#         "question": "什么是RAG系统中的“幻觉”问题？",
#         "answer": "RAG系统中的“幻觉”指的是，大型语言模型在生成答案时，捏造了事实，或者生成了与提供的上下文不符的、看似有理有据的错误信息。"
#     },
#     {
#         "question": "如何提升RAG系统的检索精度？",
#         "answer": "提升RAG检索精度的方法有很多，包括使用更先进的嵌入模型、对文档进行智能分块、采用混合搜索、以及在检索后进行重排序（Re-ranking）等。"
#     }
# ]

# # --- 2. 索引构建：只对问题进行嵌入 ---
# # 创建一个文档列表，每个文档的内容是“问题”，元数据包含“答案”
# question_documents = []
# for pair in qa_pairs:
#     # 将答案作为元数据存储
#     metadata = {"answer": pair["answer"]}
#     doc = Document(page_content=pair["question"], metadata=metadata)
#     question_documents.append(doc)

# # 使用OpenAI的嵌入模型
# embeddings_model = OpenAIEmbeddings(
#         model=os.environ.get("EMBEDDING_MODEL"),
#         base_url=os.environ.get("EMBEDDING_BASE_URL"),
#         openai_api_key=os.environ.get("EMBEDDING_API_KEY")
#         )

# # 创建一个临时的Chroma向量数据库来存储问题向量
# vectorstore_qa = Chroma.from_documents(
#     documents=question_documents,
#     embedding=embeddings_model
# )

# # --- 3. 检索与问答 ---
# def answer_from_qa_pairs(user_question: str):
#     """
#     通过在QA对知识库中搜索最相似的问题来回答。
#     """
#     print(f"\n用户问题: '{user_question}'")

#     # 在向量数据库中搜索与用户问题最相似的“已索引问题”
#     similar_question_docs = vectorstore_qa.similarity_search(user_question,k=1)

#     if similar_question_docs:
#         # 提取最相似问题的预设答案
#         most_similar_question = similar_question_docs[0].page_content
#         retrieved_answer = similar_question_docs[0].metadata['answer']
        
#         print(f"匹配到的最相似问题: '{most_similar_question}'")
#         print(f"系统回答 (来自预设答案): {retrieved_answer}")
#     else:
#         print("抱歉，在知识库中没有找到相关问题。")

# # --- 测试 ---
# answer_from_qa_pairs("langchain的LCEL是做什么的？")
# answer_from_qa_pairs("大模型幻视是什么意思？")



# 方法二：从文档中生成QA对，进行“多路召回”
# --- 1. 原始文档准备 ---
# original_docs = [
#     Document(page_content="LCEL，全称LangChain Expression Language，它通过操作符重载（如`|`符号）提供了一种声明式的、流畅的方式来构建AI链。它的关键优势包括：开箱即用的流式处理、异步和并行执行能力，以及对整个链的生命周期管理（如日志、调试）提供了极大的便利。", metadata={"doc_id": "lcel_intro"}),
#     Document(page_content="混合搜索（Hybrid Search）结合了传统关键词搜索（如BM25）和现代向量搜索的优点。关键词搜索能精确匹配术语和缩写，而向量搜索擅长理解语义和意图。二者结合能显著提升检索的鲁棒性和准确性。", metadata={"doc_id": "hybrid_search_intro"}),
# ]

# # --- 2. 从文档生成“代理问题”的链 ---
# question_gen_prompt_str = (
#     "你是一位AI专家。请根据以下文档内容，生成3个用户可能会提出的、高度相关的问题。\n"
#     "只返回问题列表，每个问题占一行，不要有其他前缀或编号。\n\n"
#     "文档内容:\n"
#     "----------\n"
#     "{content}\n"
#     "----------\n"
# )
# question_gen_prompt = ChatPromptTemplate.from_template(question_gen_prompt_str)
# question_generator_chain = question_gen_prompt | llm | StrOutputParser()

# # --- 3. MultiVectorRetriever 设置 ---
# # a. 向量数据库
# vectorstore_mv = Chroma(collection_name="multivector_retriever", embedding_function=embeddings_model)
# # b. 文档存储器：用于根据ID存储和查找原始文档
# doc_store = InMemoryStore()
# # c. 生成的子文档（问题）列表
# sub_docs = []
# # d. 原始文档ID列表
# doc_ids = [doc.metadata["doc_id"] for doc in original_docs]

# # 遍历每个原始文档，生成问题并存储
# for i, doc in enumerate(original_docs):
#     doc_id = doc_ids[i]
#     # 生成问题
#     generated_questions = question_generator_chain.invoke({"content": doc.page_content}).split("\n")
#     # 清理可能存在的空字符串
#     generated_questions = [q for q in generated_questions if q.strip()]
    
#     # 将每个问题包装成一个Document，并链接到原始文档的ID
#     for q in generated_questions:
#         sub_docs.append(Document(page_content=q, metadata={"doc_id": doc_id}))

# # 将原始文档和生成的子文档（问题）都添加到存储中
# doc_store.mset(list(zip(doc_ids, original_docs))) # 存储原始文档
# vectorstore_mv.add_documents(sub_docs) # 只索引问题

# # 初始化MultiVectorRetriever
# # - search_type="similarity": 表示用向量搜索来查找问题
# # - a. 它会在vectorstore_mv中搜索最相似的问题
# # - b. 然后根据问题的metadata['doc_id']，去doc_store中取回原始文档
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore_mv,
#     docstore=doc_store,
#     id_key="doc_id",
#     search_type="similarity"
# )

# # --- 4. 检索测试 ---
# user_query = "混合检索的好处是什么？"
# retrieved_results = retriever.invoke(user_query)

# print(f"\n用户问题: '{user_query}'")
# print("\n--- 检索到的原始文档 ---")
# print(retrieved_results[0].page_content)

# 方法三：利用QA对微调嵌入模型

import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

# --- 1. 准备微调数据集 (示例) ---
# 实际应用中，你需要成千上万这样的样本
# 每个样本是一个“查询”和“一个相关的正面段落”
train_examples = [
    InputExample(texts=["LCEL的优点有哪些？", "LCEL，全称LangChain Expression Language...提供了极大的便利。"]),
    InputExample(texts=["什么是混合搜索？", "混合搜索（Hybrid Search）结合了传统关键词搜索和现代向量搜索的优点..."]),
    InputExample(texts=["如何解决RAG幻觉", "RAG系统中的“幻觉”指的是...可以通过多种方式缓解，例如提高检索质量..."]),
    # ... 更多样本
]

# --- 2. 定义模型和数据加载器 ---
# 选择一个强大的预训练模型作为基础
model_name = 'moka-ai/m3e-base' # 这是一个优秀的中英双语模型
model = SentenceTransformer(model_name)

# 准备数据加载器
# MultipleNegativesRankingLoss 是一种非常适合此类任务的损失函数
# 它会智能地将一个batch内的其他“正向段落”作为当前查询的“负向样本”
batch_size = 4 # 实际应用中可以更大，如32或64
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# --- 3. 定义损失函数 ---
# 这个损失函数的目标是：让一个batch内，每个查询与其正向段落的相似度得分尽可能高，
# 同时与其他所有段落的相似度得分尽可能低。
train_loss = losses.MultipleNegativesRankingLoss(model)

# --- 4. 模型微调 ---
num_epochs = 3 # 实际应用中可能需要更多轮次
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10%的预热步数

print("开始微调嵌入模型...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="./finetuned_embedding_model", # 微调后模型的保存路径
    show_progress_bar=True
)

print("\n微调完成！模型已保存至 './finetuned_embedding_model'")

# --- 5. 如何使用微调后的模型 ---
# 你可以像这样加载和使用它
# finetuned_model = SentenceTransformer("./finetuned_embedding_model")
# user_embedding = finetuned_model.encode("LCEL有什么好处？")




# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# class AnswerFormat(BaseModel):
#     answer: str = Field(description="针对问题的回答")
#     references: list[str] = Field(description="答案引用的文档块ID或来源信息列表")
#     confidence_score: float = Field(description="对答案的自信度评分,0.0到1.0之间")

# # 绑定 LLM, 强制其输出 AnswerFormat 结构
# llm_structured_output = llm.with_structured_output(AnswerFormat)

# # 定义一个 Prompt,鼓励LLM输出结构化数据
# structured_rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个知识问答机器人。请根据提供的上下文,以JSON格式回答问题。如果信息不足,请将answer字段设置为'不知道',confidence_score为0.0。"),
#     ("user", "问题:{question}\n\n上下文:\n{context}")
# ])

# # 创建一个只负责生成结构化答案的链
# structured_answer_chain = (
#     {"question": RunnablePassthrough(), "context": vectorstore.as_retriever()}
#     | structured_rag_prompt
#     | llm_structured_output
# )

# print("\n--- 结构化输出 RAG 示例 ---")
# query_structured = "LangChain的优势是什么？?"
# response_structured = structured_answer_chain.invoke(query_structured)
# print(f"问题: {query_structured}")
# # pydantic模型可以方便地转为JSON
# print(f"回答:\n{response_structured}")

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain.retrievers import ContextualCompressionRetriever

# # --- 1. 构建多阶段优化检索器 ---
# # 基础检索器: 混合搜索
# bm25_retriever = BM25Retriever.from_documents(split_documents)
# bm25_retriever.k = 5
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5]
# )

# # 查询扩展: 将混合搜索作为基础
# multi_query_hybrid_retriever = MultiQueryRetriever.from_llm(
#     retriever=ensemble_retriever,
#     llm=llm
# )

# # 重排序: 将查询扩展和混合搜索的结果进行精排
# rerank_compressor = FlashrankRerank(
#     model="miniReranker_arabic_v1",
#     top_n=3
# )
# # 创建一个 ContextualCompressionRetriever, 使用 Flashrank 作为压缩器
# final_optimized_retriever = ContextualCompressionRetriever(
#     base_compressor=rerank_compressor,
#     base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}) # 先检索5个再重排
# )

# # --- 2. 构建最终的RAG链 ---

# # a. 定义RAG的Prompt
# optimized_rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一名专业的知识库助手。请根据提供的上下文**简洁明了**地回答以下问题。\n**如果上下文没有足够信息，请明确说明你不知道，不要凭空捏造。**\n\n上下文:\n{context}"),
#     ("user", "{input}")
# ])

# # 创建文档处理链
# optimized_document_chain = create_stuff_documents_chain(llm, optimized_rag_prompt)

# # 创建完整的检索链
# optimized_retrieval_chain = create_retrieval_chain(
#     final_optimized_retriever,
#     optimized_document_chain
# )

# # --- 3. 调用测试 ---
# print("\n--- 完整优化后的RAG链示例 ---")
# query_final = "LangChain的优势是什么？"
# response = optimized_retrieval_chain.invoke({"input": query_final})

# print(f"用户: {query_final}")
# print(f"AI: {response['answer']}")

# # 查看检索到的上下文，验证其高质量
# print("\n--- 检索到的上下文(Context) ---")
# for i, doc in enumerate(response['context']):
#     print(f"文档 {i+1} (来源: {doc.metadata.get('source', 'N/A')}, 分数: {doc.metadata.get('relevance_score', 'N/A')}):")
#     print(doc.page_content)
#     print("-" * 20)