from typing import Annotated, List, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from langchain_tavily._utilities import TavilySearchAPIWrapper
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
llm = ChatOpenAI(
        model_name=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

search_tool = TavilySearch(
    max_results=2,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
    )
)


def add_messages(left: list, right: list) -> list:return left + right

# 下属 Agent 的 State
# class ResearchState(TypedDict):
#     topic: str
#     research_data: str

# class WriterState(TypedDict):
#     research_data: str
#     article: str

# # 2. 主管 Agent 的 State    
# class SupervisorState(TypedDict):
#     topic: str
#     research_data: str | None
#     article: str | None
#     messages: Annotated[List[BaseMessage], add_messages]

# def research_node(state: ResearchState):
#     print("--- [黑盒内部] Researcher 正在执行 ---")
#     search_results = search_tool.invoke(state['topic'])
#     return {"research_data": str(search_results)}

# research_graph_builder = StateGraph(ResearchState)
# research_graph_builder.add_node("researcher", research_node)
# research_graph_builder.set_entry_point("researcher")
# research_graph_builder.add_edge("researcher", END)
# researcher_app_mode1 = research_graph_builder.compile()

# def writer_node(state: WriterState):
#     print("--- [黑盒内部] Writer 正在执行 ---")
#     prompt = f"根据以下研究资料，撰写一篇关于此主题的简短报告。\n\n研究资料:\n{state['research_data']}"
#     response = llm.invoke(prompt)
#     return {"article": response.content}

# writer_graph_builder = StateGraph(WriterState)
# writer_graph_builder.add_node("writer", writer_node)
# writer_graph_builder.set_entry_point("writer")
# writer_graph_builder.add_edge("writer", END)
# writer_app_mode1 = writer_graph_builder.compile()


# # --- Supervisor Nodes ---
# def supervisor_research_node(state: SupervisorState):
#     print("--- 主管 [模式一] 正在委托 Researcher ---")
#     # 1. 数据转换：SupervisorState -> ResearchState
#     research_input = {"topic": state['topic']}
#     # 2. 手动调用 invoke
#     result = researcher_app_mode1.invoke(research_input)
#     # 3. 数据转换：ResearchState -> SupervisorState
#     return {"research_data": result['research_data']}

# def supervisor_write_node(state: SupervisorState):
#     print("--- 主管 [模式一] 正在委托 Writer ---")
#     # 1. 数据转换：SupervisorState -> WriterState
#     write_input = {"research_data": state['research_data']}
#     # 2. 手动调用 invoke
#     result = writer_app_mode1.invoke(write_input)
#     # 3. 数据转换：WriterState -> SupervisorState
#     return {"article": result['article']}

# # --- 构建主管图 ---
# supervisor_graph_builder = StateGraph(SupervisorState)
# supervisor_graph_builder.add_node("research_node", supervisor_research_node)
# supervisor_graph_builder.add_node("write_node", supervisor_write_node)

# # 流程：先研究，再写作 (简单的线性流程)
# supervisor_graph_builder.set_entry_point("research_node")
# supervisor_graph_builder.add_edge("research_node", "write_node")
# supervisor_graph_builder.add_edge("write_node", END)

# team_app_mode1 = supervisor_graph_builder.compile()

# # --- 3.4 运行团队！ ---
# print("--- 运行 [模式一] ---")
# inputs = {"topic": "医疗保健中的人工智能", "messages": []}
# result = team_app_mode1.invoke(inputs)

# print("\n--- [模式一] 最终文章 ---")
# print(result['article'])


# --- 模式二：State 共享 ---
class TeamState(TypedDict):
    """一个共享的状态，所有 Agent 都可以读写"""
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    research_data: str | None
    article: str | None
    # "next_agent" 字段用于主管路由
    next_agent: str

# --- 1. Researcher App ---
def research_node_mode2(state: TeamState):
    print("--- [共享状态] 专家 [Researcher] 已激活 ---")
    search_results = search_tool.invoke(state['topic'])
    return {"research_data": str(search_results)}

research_graph_builder_m2 = StateGraph(TeamState)
research_graph_builder_m2.add_node("researcher", research_node_mode2)
research_graph_builder_m2.add_edge(START, "researcher")
research_graph_builder_m2.add_edge("researcher", END) 
researcher_app_mode2 = research_graph_builder_m2.compile()

# --- 2. Writer App ---
def writer_node_mode2(state: TeamState):
    print("--- [共享状态] 专家 [Writer] 已激活 ---")
    prompt = f"根据以下研究资料，撰写一篇关于主题 '{state['topic']}' 的简短报告。\n\n研究资料:\n{state['research_data']}"
    response = llm.invoke(prompt)
    return {
        "article": response.content,
        "messages": [AIMessage(content=response.content)]
    }

writer_graph_builder_m2 = StateGraph(TeamState)
writer_graph_builder_m2.add_node("writer", writer_node_mode2)
writer_graph_builder_m2.add_edge(START, "writer")
writer_graph_builder_m2.add_edge("writer", END)
writer_app_mode2 = writer_graph_builder_m2.compile()

# --- 主管的路由决策逻辑 ---
# # 1. 定义主管的路由工具
class Route(BaseModel):
    """选择下一步要调用的专家，或者结束"""
    next: Literal["researcher", "writer", "END"]

# 2. 绑定工具
supervisor_llm = llm.bind_tools([Route])

# 3. 主管路由节点
def supervisor_router(state: TeamState):
    print("--- 主管 [模式二] 正在决策... ---")
    
    current_topic = state['topic']
    has_research = bool(state.get('research_data'))
    has_article = bool(state.get('article'))
    system_prompt = f"""你是一个多智能体团队的主管。
    你需要根据“当前状态”，遵循“决策规则”，决定下一步行动。

    [当前状态]
    - 课题: {current_topic}
    - 是否已有研究资料: {has_research}
    - 是否已有最终报告: {has_article}

    [决策规则]
    1. 如果 "是否已有研究资料" 为 False, 必须调用 'researcher'。
    2. 如果 "是否已有研究资料" 为 True, 且 "是否已有最终报告" 为 False, 必须调用 'writer'。
    3. 如果 "是否已有最终报告" 为 True, 工作完成, 必须返回 'END'。
    """

    messages = [HumanMessage(content=system_prompt)] + state['messages']
    
    response = supervisor_llm.invoke(messages)
    
    if not response.tool_calls:
        print("--- 主管决策：END (无工具调用) ---")
        return {"next_agent": "END"}
        
    route_decision = response.tool_calls[0]['args']
    next_agent = route_decision['next']
    print(f"--- 主管决策：-> {next_agent} ---")
    return {"next_agent": next_agent}

# --- 4.4 构建主管图 ---
team_graph_builder = StateGraph(TeamState)

# 1. 添加主管的“路由”节点
team_graph_builder.add_node("supervisor_router", supervisor_router)

# 2. 关键：将“下属 App” 作为节点添加
team_graph_builder.add_node("researcher", researcher_app_mode2)
team_graph_builder.add_node("writer", writer_app_mode2)

# 3. 设置入口
team_graph_builder.set_entry_point("supervisor_router")

# 4. 设置动态路由
team_graph_builder.add_conditional_edges(
    "supervisor_router",
    # 读取 state['next_agent'] 的值来决定去向
    lambda state: state['next_agent'], 
    {
        "researcher": "researcher",
        "writer": "writer",
        "END": END
    }
)

# 5. 专家完成工作后，必须返回给主管决策
team_graph_builder.add_edge("researcher", "supervisor_router")
team_graph_builder.add_edge("writer", "supervisor_router")

# 编译最终的团队应用
# team_app_mode2 = team_graph_builder.compile()

# print("\n--- 运行 [模式二] ---")
# topic = "人工智能在医疗保健领域的应用"
# inputs = {
#     "messages": [HumanMessage(content=f"请帮我研究并撰写一篇关于 '{topic}' 的报告")],
#     "topic": topic
# } 
# result = team_app_mode2.invoke(inputs)

# print("\n--- [模式二] 最终文章 ---")
# print(result['article'])



from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
# 编译模式二的图，并传入 checkpointer
team_app_with_memory = team_graph_builder.compile(
    checkpointer=checkpointer
)

async def run_with_full_observability():
    topic = "人工智能在医疗保健领域的应用"
    inputs = {
        "messages": [HumanMessage(content=f"请帮我研究并撰写一篇关于 '{topic}' 的报告")],
        "topic": topic
    }
    
    # 监听 "updates" 流，并开启 subgraphs 追踪
    async for chunk in team_app_with_memory.astream(
        inputs, 
        stream_mode="updates", 
        subgraphs=True,
    ):
        print(chunk)
        print("-" * 25)

asyncio.run(run_with_full_observability())