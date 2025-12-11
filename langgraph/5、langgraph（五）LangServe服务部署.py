from pydantic import BaseModel
import uvicorn
from typing import Annotated, List, TypedDict, Literal, Optional
from langchain_core.runnables import ConfigurableField, RunnableConfig

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from fastapi import FastAPI
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from langchain_core.runnables import ConfigurableField

load_dotenv()

llm = ChatOpenAI(
    model_name="Qwen3-235B-A22B",
    temperature=0.9,
    openai_api_base="http://10.1.18.99:8089/v1",
    openai_api_key="sk-dIl9oEE1SCJHXkzkdTmivPJgtxMGHNgvvNx5e17T4XYHBBOG",
)

search_tool = TavilySearch(
    max_results=2,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key="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",
    )
)

# 定义消息列表合并函数，用于状态管理
def add_messages(left: list, right: list) -> list:
    """合并两个消息列表，用于LangGraph的状态更新"""
    return left + right

# 定义共享状态
class TeamState(TypedDict):
    """定义Agent团队共享的状态结构"""
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    research_data: Optional[str]
    article: Optional[str]
    next_agent: str

# 定义"下属" Agent
async def research_node(state: TeamState, config: RunnableConfig | None = None):
    """研究节点：负责搜索主题相关信息"""
    print("--- [共享状态] 专家 [Researcher] 已激活 ---")
    
    # 从配置中获取thread_id
    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")
        print(f" > 研究节点获取到的thread_id: {thread_id}")
    
    search_results = await search_tool.ainvoke(state['topic'])
    # 返回更新后的研究数据
    result = {"research_data": str(search_results)}
    
    # 如果有thread_id，也添加到返回结果中
    if thread_id:
        result["current_thread_id"] = thread_id
        
    return result

# 构建研究专家的子图
research_graph_builder = StateGraph(TeamState)
research_graph_builder.add_node("researcher", research_node)
research_graph_builder.add_edge(START, "researcher")
research_graph_builder.add_edge("researcher", END)
researcher_app = research_graph_builder.compile()

async def writer_node(state: TeamState, config: RunnableConfig | None = None):
    """写作节点：根据研究数据生成报告"""
    print("--- [共享状态] 专家 [Writer] 已激活 ---")
    
    # 从配置中获取thread_id
    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")
        print(f" > 写作节点获取到的thread_id: {thread_id}")
    
    # 构建提示词，包含主题和研究资料
    prompt = f"根据以下研究资料，撰写一篇关于主题 '{state['topic']}' 的简短报告。\n\n研究资料:\n{state['research_data']}"
    # 使用LLM生成报告
    response = await llm.ainvoke(prompt)
    
    # 返回生成的文章和消息，以及thread_id
    result = {
        "article": response.content,
        "messages": [AIMessage(content=response.content)]
    }
    
    # 将thread_id添加到返回结果中
    if thread_id:
        result["current_thread_id"] = thread_id
        # 也可以将thread_id添加到响应消息中，让用户在回复中看到
        enhanced_content = f"{response.content}\n\n(当前会话ID: {thread_id})"
        result["messages"] = [AIMessage(content=enhanced_content)]
    
    return result

# 构建写作专家的子图
writer_graph_builder = StateGraph(TeamState)
writer_graph_builder.add_node("writer", writer_node)
writer_graph_builder.add_edge(START, "writer")
writer_graph_builder.add_edge("writer", END)
writer_app = writer_graph_builder.compile()

# 定义"主管" Agent
class InputMessage(BaseModel):
    """API输入消息模型，用于验证和类型转换"""
    type: Literal["human", "ai", "system"]
    content: str

async def entry_node_normalize_messages(state: TeamState, config: RunnableConfig | None = None):
    """入口节点：负责规范化输入消息和确保状态完整性"""
    print("--- 节点 [entry_node_normalize_messages] 运行 ---")
    
    # 从配置中获取thread_id（演示如何获取当前thread_id）
    if config:
        # 通过config获取thread_id的方式
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            print(f" > 当前会话的thread_id: {thread_id}")
        else:
            print(" > 未找到thread_id，这是新的会话")
    else:
        print(" > 配置信息未提供")
    
    # 创建状态副本，避免直接修改原始状态
    new_state = dict(state)
    
    # 检查并转换消息类型，确保消息格式正确
    if new_state.get('messages'):
        if not isinstance(new_state['messages'], list):
            print(" > 警告: messages 不是列表类型，将其转换为空列表")
            new_state['messages'] = []
        elif new_state['messages'] and isinstance(new_state['messages'][0], InputMessage):
            print(" > 检测到 'messages' 为 List[InputMessage]，正在手动转换为 List[BaseMessage]...")
            
            converted_messages = []
            # 将InputMessage转换为LangChain的消息类型
            for msg in new_state['messages']:
                if msg.type == "human":
                    converted_messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    converted_messages.append(AIMessage(content=msg.content))
                elif msg.type == "system":
                    converted_messages.append(SystemMessage(content=msg.content))
            
            new_state['messages'] = converted_messages
    else:
        # 确保messages字段始终存在且为列表
        new_state['messages'] = []
    
    # 确保其他必要字段存在并有默认值
    if 'topic' not in new_state:
        new_state['topic'] = ""
    if 'research_data' not in new_state:
        new_state['research_data'] = None
    if 'article' not in new_state:
        new_state['article'] = None
    if 'next_agent' not in new_state:
        new_state['next_agent'] = ""
    
    return new_state


async def supervisor_router(state: TeamState, config: RunnableConfig | None = None):
    """主管路由器：基于当前状态决定下一步操作"""
    try:
        print("--- 主管正在决策... ---")
        
        # 从配置中获取thread_id
        thread_id = None
        if config:
            thread_id = config.get("configurable", {}).get("thread_id")
            print(f" > 主管节点获取到的thread_id: {thread_id}")
        
        # 获取当前状态信息
        current_topic = state.get('topic', '未知课题')
        has_research = bool(state.get('research_data'))
        has_article = bool(state.get('article'))

        # 基于状态进行决策：先研究，再写作，完成后结束
        if not has_research:
            next_agent = "researcher"
        elif not has_article:
            next_agent = "writer"
        else:
            next_agent = "END"
        
        print(f"--- 主管决策：-> {next_agent} ---")
        
        # 返回next_agent和当前thread_id
        result = {"next_agent": next_agent}
        if thread_id:
            result["thread_id"] = thread_id
            
        return result
    except Exception as e:
        # 添加异常捕获，确保出错时不会导致服务崩溃
        print(f"--- 主管决策过程中发生错误：{str(e)} ---")
        # 出错时返回END以避免无限循环
        return {"next_agent": "END"}

# --- 6. 构建并编译 Agent 团队 --- 
team_graph_builder = StateGraph(TeamState)

# 添加各个节点
team_graph_builder.add_node("entry_normalizer", entry_node_normalize_messages)
team_graph_builder.add_node("supervisor_router", supervisor_router)
team_graph_builder.add_node("researcher", researcher_app)
team_graph_builder.add_node("writer", writer_app)

team_graph_builder.set_entry_point("entry_normalizer")

team_graph_builder.add_edge("entry_normalizer", "supervisor_router")

team_graph_builder.add_conditional_edges(
    "supervisor_router",
    lambda state: state['next_agent'],
    {"researcher": "researcher", "writer": "writer", "END": END}
)

team_graph_builder.add_edge("researcher", "supervisor_router")
team_graph_builder.add_edge("writer", "supervisor_router")

checkpointer = MemorySaver()
team_app_with_memory = team_graph_builder.compile(checkpointer=checkpointer)

# 定义API输入模型
class AgentInput(BaseModel):
    """API输入模型，定义客户端请求的数据结构"""
    topic: str
    messages: List[InputMessage]

# 配置应用的输入输出类型，并添加线程ID配置
final_app = team_app_with_memory.with_types(
    input_type=AgentInput,
    output_type=TeamState
).with_config(
    configurable={"thread_id": ConfigurableField(
        id="thread_id", 
        name="Thread ID", 
        description="会话ID，用于启用记忆功能，确保多轮对话的连续性"
    )}
)

# 创建FastAPI应用实例
app = FastAPI(
    title="LangGraph Multi-Agent Team API",
    version="1.0",
    description="一个用于研究和写作的 Agent 团队 API 服务",
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加LangServe路由，将Agent应用注册到FastAPI
add_routes(
    app, 
    final_app, 
    path="/agent",
    config_keys=["thread_id"]
)

# 主函数，启动服务
if __name__ == "__main__":
    print("--- LangGraph Agent 服务启动 ---")
    print("API 文档地址: http://localhost:8002/docs")
    print("Playground 地址: http://localhost:8002/agent/playground/")
    # 启动Uvicorn服务器，监听所有网络接口的8002端口
    uvicorn.run(app, host="0.0.0.0", port=8002)