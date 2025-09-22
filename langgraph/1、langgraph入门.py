from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

load_dotenv()
llm = ChatOpenAI(
        model_name=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )


# # 定义状态
# class ReportState(TypedDict):
#     topic: str
#     draft: str
#     report: str

# # 定义节点函数
# def draft_node(state: ReportState) -> dict:
#     """起草节点"""
#     print(">>> 正在执行节点: draft_node")
#     prompt = f"以技术报告的格式，为主题 '{state['topic']}' 写一份500字左右的草稿。"
#     draft = llm.invoke(prompt).content
#     return {"draft": draft}

# def review_node(state: ReportState) -> dict:
#     """审阅节点"""
#     print(">>> 正在执行节点: review_node")
#     prompt = f"你是一位资深编辑。请审阅以下报告草稿，修正语法、改进措辞。草稿：\n\n{state['draft']}"
#     report = llm.invoke(prompt).content
#     return {"report": report}

# # 初始化状态图
# workflow = StateGraph(ReportState)

# # 添加节点
# workflow.add_node("drafter", draft_node)
# workflow.add_node("reviewer", review_node)

# # 设置入口点并添加边，构建线性流程
# workflow.set_entry_point("drafter")
# workflow.add_edge("drafter", "reviewer")
# workflow.add_edge("reviewer", END)

# # 编译图
# linear_app = workflow.compile()

# # 可视化图结构
# linear_app.get_graph().print_ascii()

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """一个简单的列表拼接函数"""
    return left + right

class AgentState(TypedDict):
    # 使用 Annotated 指示 messages 字段应通过 add_messages 函数进行更新
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper

# 实例化工具
search_tool = TavilySearch(
    max_results=2,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=os.environ.get("TAVILY_API_KEY")
    )
)
tools = [search_tool]

def router(state: AgentState) -> str:
    """根据 Agent 的最新消息中是否包含工具调用请求，来决定下一步的走向。"""
    print("--- 正在执行路由判断 ---")
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # 如果模型决定调用工具
        print(">>> 路由决策：调用工具")
        return "tool_node"
    else:
        # 如果模型决定直接回复
        print(">>> 路由决策：直接结束")
        return "END"
    
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# 定义 Agent 节点，负责调用模型进行“思考”
def agent_node(state: AgentState):
    print("--- 正在执行节点: agent_node (思考) ---")
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state['messages'])
    return {"messages": [response]}

# 定义工具节点，这是一个预构建的节点，专门用于执行工具
tool_node = ToolNode(tools=[search_tool])

from langgraph.graph import StateGraph, END

# 初始化图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)

# 设置入口点
graph.set_entry_point("agent_node")

# 添加条件边
graph.add_conditional_edges(
    "agent_node", # 决策的起点
    router,       # 决策的判断逻辑
    {
         # 路径映射
        "tool_node": "tool_node",
        "END": END
    }
)

# 添加从工具节点到 Agent 节点的常规边，形成循环
graph.add_edge("tool_node", "agent_node")

# 编译图
agent_app = graph.compile()

# 可视化 Agent 的图结构
agent_app.get_graph().print_ascii()

from langchain_core.messages import HumanMessage

# 测试 1: 需要调用工具的问题
inputs = {"messages": [HumanMessage(content="今天北京的天气怎么样？")]}
result = agent_app.invoke(inputs)
print("\n--- 最终结果 1 ---")
print(result['messages'][-1].content)

# 测试 2: 不需要调用工具的问题
inputs_2 = {"messages": [HumanMessage(content="1加1等于多少")]}
result_2 = agent_app.invoke(inputs_2)
print("\n--- 最终结果 2 ---")
print(result_2['messages'][-1].content)

from langgraph.prebuilt import create_react_agent

# 实例化工具
search_tool = TavilySearch(
    max_results=2,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=os.environ.get("TAVILY_API_KEY")
    )
)
tools = [search_tool]

# 2. 调用工厂函数，一键生成 Agent
# 它会自动处理工具绑定
app = create_react_agent(llm, tools)

# 3. 直接使用
inputs = {"messages": [HumanMessage(content="今天北京的天气怎么样？")]}
result = app.invoke(inputs)

print("\n--- 最终结果 ---")
print(result['messages'][-1].content)