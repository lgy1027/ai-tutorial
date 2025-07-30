from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever # 历史感知检索器
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

# --- 1. 初始化模型和Embedding ---
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


# --- 3. 创建 Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 检索最相关的2个文档块

# --- 4. 定义 RAG 提示模板 ---
# 提示中需要包含 {context} 和 {input} 两个变量
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "请根据提供的上下文回答以下问题。\n如果上下文没有明确信息，请说明你不知道。\n\n上下文:\n{context}"),
    ("user", "{input}")
])

# --- 5. 构建文档合并链 (stuffing) ---
# 这个链负责将检索到的文档 {context} 和用户问题 {input} 传入 Prompt 并交给 LLM 生成答案
document_chain = create_stuff_documents_chain(llm, rag_prompt)

# --- 6. 构建最终的 RAG 链 ---
# 这个链将 retriever 和 document_chain 组合起来
# 它接收用户问题，先通过 retriever 获取文档，再将文档和问题传给 document_chain
retrieval_rag_chain = create_retrieval_chain(retriever, document_chain)

# --- 7. 调用 RAG 链 ---
print("\n--- 基础 RAG 问答系统示例 ---")
query1 = "LangChain是做什么的？"
response1 = retrieval_rag_chain.invoke({"input": query1})
print(f"问题: {query1}")
print(f"回答: {response1['answer']}") # 注意 create_retrieval_chain 的输出是字典，答案在 'answer' 键
print(f"检索到的文档 (部分):\n")
for doc in response1["context"]:
    print(f"- {doc.page_content[:50]}...") # 打印检索到的文档内容
print("-" * 30)


query2 = "RAG解决了什么问题？"
response2 = retrieval_rag_chain.invoke({"input": query2})
print(f"问题: {query2}")
print(f"回答: {response2['answer']}")
print(f"检索到的文档 (部分):\n")
for doc in response2["context"]:
    print(f"- {doc.page_content[:50]}...")
print("-" * 30)

query3 = "2023年诺贝尔物理学奖得主是谁？" # 知识库中没有的信息
response3 = retrieval_rag_chain.invoke({"input": query3})
print(f"问题: {query3}")
print(f"回答: {response3['answer']}") # 应该回答不知道
print("-" * 30)


# --- 准备文档和 Prompt
long_doc_content_1 = "这是一个关于人工智能发展史的长篇介绍，从最初的逻辑推理，到专家系统，再到机器学习，直至深度学习和大型语言模型的崛起。它详细描述了各个阶段的关键里程碑和技术突破，以及面临的挑战和伦理考量。AI已经深刻改变了多个行业，未来潜力无限。"
long_doc_content_2 = "本节深入探讨了生成式AI的最新进展，特别是扩散模型和Transformer架构。生成式AI能够创造出逼真的图像、文本和音频，在艺术、设计、内容创作等领域展现出巨大潜力。同时，也讨论了其在偏见、版权和滥用方面的问题。"
map_docs = [
    Document(page_content=long_doc_content_1, metadata={"source": "AIHistory"}),
    Document(page_content=long_doc_content_2, metadata={"source": "GenerativeAI"})
]

# # 定义 map 阶段的 Prompt
# # 这个 prompt 接收一个名为 "context" 的变量，它将是单个文档的内容
map_prompt = ChatPromptTemplate.from_template("请总结以下文档的关键信息：\n{context}")

# # 定义 reduce 阶段的 Prompt
# 这个 prompt 接收两个变量: "context" (所有总结的合并) 和 "question"
reduce_prompt = ChatPromptTemplate.from_template("根据以下总结，生成最终答案：\n{context}\n\n问题: {question}")


# --- 使用 LCEL 构建 Map-Reduce 链 ---

# 1. 定义 Map 链
# 这个链负责处理单个文档
# 它接收一个 Document 对象，提取其 page_content，然后传递给 prompt 和 LLM
map_chain = (
    {"context": lambda doc: doc.page_content}
    | map_prompt
    | llm
    | StrOutputParser()
)

# 2. 定义 Reduce 步骤中合并总结的函数
def combine_summaries(summaries):
    """将总结列表合并成一个字符串"""
    return "\n\n".join(summaries)

# 3. 构建完整的 Map-Reduce 链
# 使用 RunnablePassthrough.assign 来并行处理，并将结果赋给新的键
# 链的输入是 {"question": str, "context": List[Document]}
map_reduce_chain = (
    RunnablePassthrough.assign(
        # "summaries" 键的值通过对输入的 "context" 应用 map_chain.map() 来获得
        summaries= (lambda x: x["context"]) | map_chain.map()
    )
    | {
        # 为 reduce_prompt 准备输入
        "context": lambda x: combine_summaries(x["summaries"]), # 合并总结
        "question": lambda x: x["question"], # 传递原始问题
    }
    | reduce_prompt
    | llm
    | StrOutputParser()
)


# --- 调用链并打印结果 ---
print("\n--- Map-Reduce 文档合并策略示例 ---")
question = "生成式AI的最新进展是什么？"
result_map_reduce = map_reduce_chain.invoke({"question": question, "context": map_docs})

print(f"问题: {question}")
print(f"回答 (Map-Reduce): {result_map_reduce}")



# 1. 定义初始处理链的 Prompt
initial_prompt_template = "根据以下内容，简洁地回答问题：\n\n内容: {context}\n\n问题: {question}"
initial_prompt = ChatPromptTemplate.from_template(initial_prompt_template)

# 2. 定义 Refine 处理链的 Prompt
refine_prompt_template = (
    "原始问题: {question}\n"
    "我们已经有了一个初步的答案: {existing_answer}\n"
    "现在有额外的上下文信息: {context}\n"
    "请根据新的上下文信息，精炼或扩展之前的答案。如果新信息与问题无关，请返回原答案。"
)
refine_prompt = ChatPromptTemplate.from_template(refine_prompt_template)

def run_refine_chain(docs, question):
    # 创建处理第一个文档的链
    initial_chain = (
        {
            "context": lambda x: x["context"][0].page_content, # 提取第一个文档的内容
            "question": lambda x: x["question"]
        }
        | initial_prompt
        | llm
        | StrOutputParser()
    )
    # 运行初始链，得到初步答案
    initial_answer = initial_chain.invoke({"context": docs, "question": question})

    # 创建精炼链
    refine_chain = (
        {
            "question": lambda x: x["question"],
            "existing_answer": lambda x: x["existing_answer"],
            "context": lambda x: x["context"].page_content # 提取当前文档的内容
        }
        | refine_prompt
        | llm
        | StrOutputParser()
    )

    # d. 循环处理剩余的文档
    refined_answer = initial_answer
    for i, doc in enumerate(docs[1:]):
        print(f"Refining with doc {i+1}...")
        refined_answer = refine_chain.invoke({
            "question": question,
            "existing_answer": refined_answer,
            "context": doc
        })

    return refined_answer

question = "生成式AI的最新进展是什么？"
print("\n--- Refine 文档合并策略示例---")
result_refine = run_refine_chain(map_docs, question)
print(f"问题: {question}")
print(f"回答 (Refine): {result_refine}")



# --- 1. 准备数据和向量数据库 (同上) ---
persist_directory_history_rag = "./chroma_db_history_rag"
if not os.path.exists(persist_directory_history_rag) or not Chroma(persist_directory=persist_directory_history_rag, embedding_function=embeddings_model)._collection.count():
    print(f"正在创建并持久化 Chroma 数据库到 '{persist_directory_history_rag}'...")
    vectorstore_history_rag = Chroma.from_documents(
        documents=split_documents, # 使用之前切分好的文档
        embedding=embeddings_model,
        persist_directory=persist_directory_history_rag
    )
    print("Chroma 数据库创建/加载完成。")
else:
    print(f"从 '{persist_directory_history_rag}' 加载现有 Chroma 数据库...")
    vectorstore_history_rag = Chroma(
        persist_directory=persist_directory_history_rag,
        embedding_function=embeddings_model
    )
    print("Chroma 数据库加载完成。")

retriever_history_rag = vectorstore_history_rag.as_retriever(search_kwargs={"k": 2})

# --- 2. 创建历史感知检索器 (History-Aware Retriever) ---
# 这个 LLM Chain 会根据对话历史和用户最新问题，生成一个独立的查询字符串
history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"), # 历史对话
    ("user", "{input}"), # 用户当前问题
    ("system", "根据上面的对话历史和用户最新问题，生成一个独立的、用于检索相关文档的问题。只返回新的查询，不要添加其他内容。")
])

# 将 llm 和 history_aware_prompt 组合，用于生成新的查询，然后将查询传给 retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever_history_rag, # 基础检索器
    history_aware_prompt # 用于生成新的查询的Prompt
)

# --- 3. 定义 RAG 提示模板 (同上) ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "请根据提供的上下文回答以下问题。\n如果上下文没有足够的信息，请说明你不知道。\n\n上下文:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"), # 确保Prompt也能看到历史
    ("user", "{input}")
])

# --- 4. 构建文档合并链 (同上) ---
document_chain_history_rag = create_stuff_documents_chain(llm, rag_prompt)

# --- 5. 构建最终的 RAG 链 (使用历史感知检索器) ---
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, # 这里使用历史感知检索器
    document_chain_history_rag
)

# --- 6. 整合记忆管理 (使用 RunnableWithMessageHistory) ---
# 存储会话历史的字典 (模拟持久化存储)
store = {}
def get_session_history_for_rag(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ConversationBufferWindowMemory(
            k=5, # 保留更多轮次的记忆
            return_messages=True,
            input_key="input", # 指定输入键
            output_key="answer" # 指定输出键，与 create_retrieval_chain 的输出匹配
        ).chat_memory
    return store[session_id]

# 包装 RAG 链，使其具有记忆功能
with_message_history_rag_chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history_for_rag,
    input_messages_key="input",
    history_messages_key="chat_history", # 与 Prompt 中的 placeholder 对应
    output_messages_key="answer" # 与 create_retrieval_chain 的输出字典中的键对应
)

# --- 7. 进行多轮对话测试 ---
print("\n--- 带记忆的多轮 RAG 聊天机器人示例 ---")
session_id = "rag_user_test_001"

print("\n--- 第一轮：介绍 ---")
response_m1 = with_message_history_rag_chain.invoke(
    {"input": "你好，我想了解LangChain。"},
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 你好，我想了解LangChain。")
print(f"AI: {response_m1['answer']}")

print("\n--- 第二轮：关于它的调试工具 ---")
response_m2 = with_message_history_rag_chain.invoke(
    {"input": "它的调试工具叫什么？"}, # 注意这里没有明确指明是LangChain
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 它的调试工具叫什么？")
print(f"AI: {response_m2['answer']}") # 应该能正确回答LangSmith

print("\n--- 第三轮：它解决了什么问题？ ---")
response_m3 = with_message_history_rag_chain.invoke(
    {"input": "它解决了什么问题？"}, # 再次隐晦指代
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 它解决了什么问题？")
print(f"AI: {response_m3['answer']}") # 应该能正确回答RAG解决的问题

print("\n--- 第四轮：问一个知识库没有的问题 ---")
response_m4 = with_message_history_rag_chain.invoke(
    {"input": "谁是美国第一位总统？"}, # 知识库没有的常识问题
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 谁是美国第一位总统？")
print(f"AI: {response_m4['answer']}") # 应该回答不知道或基于LLM自身知识回答