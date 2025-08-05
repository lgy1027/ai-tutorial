import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0",
    description="A RAG chatbot API powered by LangChain and LangServe.",
)

# --- 1. 初始化模型和向量数据库 ---
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

# 加载向量数据库
persist_directory_for_serve = "./chroma_db_rag_basic"
vectorstore_for_serve = Chroma(
    persist_directory=persist_directory_for_serve,
    embedding_function=embeddings_model
)
retriever_for_serve = vectorstore_for_serve.as_retriever(search_kwargs={"k": 3})


# --- 2. 构建 RAG 链条 ---

# 历史感知提示模板
history_aware_prompt_serve = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "根据上面的对话历史和用户最新问题，生成一个独立的、用于检索相关文档的问题。")
])

# RAG 提示模板
rag_prompt_serve = ChatPromptTemplate.from_messages([
    ("system", "请根据提供的上下文回答问题。\n上下文:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# 文档处理链
document_chain_serve = create_stuff_documents_chain(llm, rag_prompt_serve)


# --- 核心改动：重构对话式 RAG 链 ---

retriever_chain_with_parser = history_aware_prompt_serve | llm | StrOutputParser()

# 创建检索链
# 第一个参数是一个新的链条：它接收输入，通过带解析器的模型生成查询字符串，然后用该字符串进行检索
# create_retrieval_chain 会智能地将检索到的文档（作为 'context'）和原始输入一起传递给 document_chain_serve
conversational_rag_chain_serve = create_retrieval_chain(
    retriever_chain_with_parser | retriever_for_serve,
    document_chain_serve
)

# --- 3. 添加记忆功能 ---

store_serve = {}
def get_session_history_serve(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_serve:
        store_serve[session_id] = ConversationBufferWindowMemory(
            k=5, return_messages=True, input_key="input", output_key="answer"
        ).chat_memory
    return store_serve[session_id]

# 最终的带记忆 RAG 链
final_rag_chain_serve = RunnableWithMessageHistory(
    conversational_rag_chain_serve,
    get_session_history_serve,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


# --- 4. 使用 add_routes 将链条注册为 API 端点 ---
add_routes(
    app,
    final_rag_chain_serve,
    path="/rag-chatbot", # API 的根路径
    playground_type="default" # 提供一个默认的 playground UI
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5432)