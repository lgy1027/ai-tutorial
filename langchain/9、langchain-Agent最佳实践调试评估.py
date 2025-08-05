import logging
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks import CallbackManager # 导入 CallbackManager

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory   
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv() # 加载 LangSmith 相关的环境变量

# --- 1. RAG 链的构建 (复用第八期优化后的链) ---
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

# 加载或创建向量数据库
persist_directory_langsmith_rag = "./chroma_db_rag_basic" # 使用第八期的数据库
vectorstore_langsmith_rag = Chroma(
    persist_directory=persist_directory_langsmith_rag,
    embedding_function=embeddings_model
)
base_retriever_for_ls = vectorstore_langsmith_rag.as_retriever(search_kwargs={"k": 3})

# 历史感知检索器 (简化，不包含MultiQuery和Rerank，以聚焦LangSmith追踪)
history_aware_prompt_ls = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "根据上面的对话历史和用户最新问题，生成一个独立的、用于检索相关文档的问题。只返回新的查询。")
])
history_aware_retriever_ls = create_history_aware_retriever(llm, base_retriever_for_ls, history_aware_prompt_ls)

# RAG 提示模板
rag_prompt_ls = ChatPromptTemplate.from_messages([
    ("system", "请根据提供的上下文回答以下问题。\n如果上下文没有足够信息，请说明你不知道。\n\n上下文:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain_ls = create_stuff_documents_chain(llm, rag_prompt_ls)
conversational_rag_chain_ls = create_retrieval_chain(history_aware_retriever_ls, document_chain_ls)

# 记忆管理
store_ls = {}
def get_session_history_ls(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_ls:
        store_ls[session_id] = ConversationBufferWindowMemory(
            k=5, return_messages=True, input_key="input", output_key="answer"
        ).chat_memory
    return store_ls[session_id]

final_rag_chain_for_ls = RunnableWithMessageHistory(
    conversational_rag_chain_ls,
    get_session_history_ls,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- 2. 运行链条，观察 LangSmith 追踪 ---
print("\n--- 运行RAG链，数据将追踪到LangSmith ---")
session_id_ls = "langsmith_test_user_001"

response_ls_1 = final_rag_chain_for_ls.invoke(
    {"input": "你好，我想了解LangChain。"},
    config={"configurable": {"session_id": session_id_ls}}
)
print(f"用户: 你好，我想了解LangChain。")
print(f"AI: {response_ls_1['answer']}")

response_ls_2 = final_rag_chain_for_ls.invoke(
    {"input": "它的调试工具叫什么？"},
    config={"configurable": {"session_id": session_id_ls}}
)
print(f"用户: 它的调试工具叫什么？")
print(f"AI: {response_ls_2['answer']}")

print("\n请访问 LangSmith 平台 (smith.langchain.com) 查看本次运行的详细追踪信息。")
print("在 'Projects' 页面找到你的项目名称，点击进入即可看到每次 'invoke' 对应的 Trace。")


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 自定义一个 CallbackHandler 来记录 LLM 调用
class CustomLLMLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        logging.info(f"LLM Call Started. Prompts: {prompts[0][:50]}...")

    def on_llm_end(self, response, **kwargs):
        logging.info(f"LLM Call Ended. Generated (partial): {response.generations[0][0].text[:50]}...")

# 将 CustomLLMLogger 添加到 LLM 的 callback_manager 中
llm_with_logging = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        callbacks=CallbackManager([CustomLLMLogger()]) # 在这里添加你的自定义Logger
    )

# 使用 llm_with_logging 构建RAG链，就会在控制台看到 LLM 的日志
res = llm_with_logging.invoke("你好")
print(res)