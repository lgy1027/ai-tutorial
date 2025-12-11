import os
import asyncio
import time
from typing import Annotated, List, Generator
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.config import get_stream_writer  
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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
        tavily_api_key="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"
    )
)

tools = [search_tool]

# def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
#     """将新消息列表追加到旧消息列表中"""
#     return left + right

# class AgentState(TypedDict):
#     messages: Annotated[List[BaseMessage], add_messages]

# def agent_node(state: AgentState):
#     """思考节点：调用 LLM 决定下一步行动"""
#     print("--- Executing node: agent_node ---")
#     response = llm.bind_tools(tools).invoke(state['messages'])
#     return {"messages": [response]}

# tool_node = ToolNode(tools=tools)

# def router(state: AgentState) -> str:
#     """路由：判断是否需要调用工具"""
#     print("--- Executing router ---")
#     last_message = state['messages'][-1]
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         return "tool_node"
#     else:
#         return "END"

# graph_builder = StateGraph(AgentState)
# graph_builder.add_node("agent_node", agent_node)
# graph_builder.add_node("tool_node", tool_node)
# graph_builder.set_entry_point("agent_node")
# graph_builder.add_conditional_edges(
#     "agent_node", 
#     router,
#     {
#         "tool_node": "tool_node",
#         "END": END
#     }
# )
# graph_builder.add_edge("tool_node", "agent_node")

# app = graph_builder.compile()


# async def run_values_mode():
#     print("\n--- 模式: values ---")
#     inputs = {"messages": [HumanMessage(content="上海天气怎么样?")]}
#     async for chunk in app.astream(inputs, stream_mode="values"):
#         print("--- 状态快照 ---")
#         print(chunk)
#         print("-" * 25)

# asyncio.run(run_values_mode())

# async def run_updates_mode():
#     print("\n--- 模式: updates ---")
#     inputs = {"messages": [HumanMessage(content="上海天气怎么样?")]}
#     async for chunk in app.astream(inputs, stream_mode="updates"):
#         print(chunk)
#         print("-" * 25)

# asyncio.run(run_updates_mode())

# async def run_messages_mode():
#     print("\n--- 模式: messages ---")
#     inputs = {"messages": [HumanMessage(content="上海天气怎么样?")]}
#     async for chunk in app.astream(inputs, stream_mode="messages"):
#         if chunk:
#             print(chunk[0].content, end="", flush=True)
#             print("-" * 25)

# asyncio.run(run_messages_mode())

# def agent_node_with_custom_event(state: AgentState):
#     print("--- Executing node: agent_node_with_custom_event ---")
#     writer = get_stream_writer()  
#     writer({"data": "Retrieved 0/100 records", "type": "progress"})  
#     # 执行查询  
#     writer({"data": "Retrieved 100/100 records", "type": "progress"})  

#     time.sleep(1) 
#     response = llm.bind_tools(tools).invoke(state['messages'])
#     return {"messages": [response]}

# # 2. 构建并编译新图
# graph_custom_builder = StateGraph(AgentState)
# graph_custom_builder.add_node("agent_node", agent_node_with_custom_event)
# graph_custom_builder.add_node("tool_node", tool_node)
# graph_custom_builder.set_entry_point("agent_node")
# graph_custom_builder.add_conditional_edges(
#     "agent_node", 
#     router,
#     {
#         "tool_node": "tool_node",
#         "END": END
#     }
# )
# graph_custom_builder.add_edge("tool_node", "agent_node")
# app_custom = graph_custom_builder.compile()

# # 3. 运行并监听
# async def run_custom_mode():
#     print("\n--- 模式: custom ---")
#     inputs = {"messages": [HumanMessage(content="上海天气怎么样?")]}
#     async for chunk in app_custom.astream(inputs, stream_mode="custom"):
#         print(chunk)

# asyncio.run(run_custom_mode())

# # 1. 编译一个在 agent_node 之后会中断的 app

# checkpointer = MemorySaver()
# app_interrupt = graph_builder.compile(checkpointer=checkpointer,interrupt_after=["agent_node"])

# async def run_interrupt_mode_correctly():
#     print("\n--- Correctly Handling Interrupts ---")
#     # thread_id 就像一个“存档ID”，让我们可以恢复图的状态
#     config = {"configurable": {"thread_id": "interrupt-thread-1"}}
#     inputs = {"messages": [HumanMessage(content="上海天气怎么样?")]}
#     print("--- First run, streaming with 'values', expecting interrupt ---")
#     # 使用 "values" 模式运行，流会在中断点自动结束
#     async for chunk in app_interrupt.astream(inputs, config=config, stream_mode="updates"):
#         print("--- Stream Chunk ---")
#         print(chunk)
    
#     print("\n--- [Graph Interrupted] ---")

#     # 检查当前状态，确认我们正处于中断状态
#     current_state = await app_interrupt.aget_state(config)
#     print("\nLast message before interrupt:", current_state.values['messages'][-1])
#     # `next` 属性告诉我们下一步将执行哪个节点，这证明图已暂停
#     print("Next step would be:", current_state.next)

#     if current_state.next: # 如果 next 有值，说明图被中断了
#         print("\n--- Resuming execution ---")
#         # 传入 None 并使用相同的 config 来继续执行
#         async for chunk in app_interrupt.astream(None, config=config, stream_mode="updates"):
#             print("--- Resumed Chunk ---")
#             print(chunk)
#             print("-" * 25)

# asyncio.run(run_interrupt_mode_correctly())


# 定义一个包含生成内容的状态
# class GenerationState(TypedDict):
#     messages: list
#     generation: str

# def generator_node(state: GenerationState):
#     print(">>> 正在执行: generator_node")
#     return {"generation": "这是一个初步生成的答案。"}

# def validator_node(state: GenerationState):
#     print(">>> 正在执行: validator_node")
#     quality_score = random.random()
#     print(f"答案质量分: {quality_score:.2f}")
#     if quality_score > 0.7:
#         print("--- 决策: 质量合格，直接结束 ---")
#         return Command(goto=END)
#     else:
#         print("--- 决策: 质量不合格，强制跳转到人工审核 ---")
#         return Command(goto="human_review")

# def human_review_node(state: GenerationState):
#     print(">>> 正在执行: human_review_node (需要人工介入！)")
#     return Command(update={"generation": state['generation'] + " [经过人工优化]"})

# # 构建图：注意，我们故意不画 validator 到 human_review 的边
# graph_dynamic = StateGraph(GenerationState)
# graph_dynamic.add_node("generator", generator_node)
# graph_dynamic.add_node("validator", validator_node)
# graph_dynamic.add_node("human_review", human_review_node)
# graph_dynamic.set_entry_point("generator")
# graph_dynamic.add_edge("generator", "validator")
# graph_dynamic.add_edge("human_review", "validator") # 审核后再次验证

# app_dynamic = graph_dynamic.compile()

# # 多次运行，你会看到它有时直接结束，有时会进入 human_review
# print("\n--- 运行动态跳转图 ---")
# result = app_dynamic.invoke({"messages":[]})
# print(result['generation'])

tool_node = ToolNode(tools=tools)
from functools import partial

def add_messages(left: list[BaseMessage], right: list[BaseMessage],k: int = 10) -> list[BaseMessage]:
    """将新消息列表追加到旧消息列表中"""
    full_list = left + right
    return full_list[-k:]



class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], partial(add_messages, k=10)]
    search_results: list | None
    draft_report: str | None


async def generate_draft_node(state: ResearchState):
    """根据搜索结果生成报告初稿"""
    print("--- Executing node: generate_draft_node (AI正在撰写初稿...) ---")
    prompt = f"根据以下搜索结果，为用户的最后一个问题生成一份详细的报告初稿: {state['search_results']}"
    messages = state["messages"] + [("user", prompt)]
    
    response = await llm.ainvoke(messages)
    
    return {"draft_report": response.content}

def human_review_node(state: ResearchState):
    """人类审核节点 - 这个节点本身不执行逻辑，仅作为中断点"""
    print("--- Reached node: human_review_node (等待人类审核...) ---")
    return {}

def finalize_report_node(state: ResearchState):
    """根据（可能被修改过的）初稿生成最终消息"""
    print("--- Executing node: finalize_report_node (生成最终报告...) ---")
    reviewed_report = state["draft_report"]
    final_message = AIMessage(content=f"这是根据您的审核生成的最终报告：\n\n{reviewed_report}")
    return {"messages": [final_message]}

def quality_check_node(state: ResearchState):
    """
    检查搜索结果的质量，并使用 goto 动态决定下一步。
    """
    print("--- Executing node: quality_check_node ---")
    writer = get_stream_writer()
    writer({"status": "正在评估搜索结果质量...","type":"quality_check_node"})

    search_results = state.get("search_results")
    if not search_results or len(search_results) < 2:
        print("--- 决策: 搜索结果不足，中断并请求用户澄清 ---")
        return Command(goto="clarify_with_user_node")
    else:
        print("--- 决策: 搜索结果充足，跳转至报告生成 ---")
        return Command(goto="generate_draft_node")


def clarify_with_user_node(state: ResearchState):
    """生成需要用户澄清的提示消息"""
    clarify_msg = AIMessage(content="搜索结果不足，请提供更具体的问题或补充信息。")
    return {"messages": [clarify_msg]}

def agent_node(state: ResearchState):
    """思考节点：调用 LLM 决定下一步行动"""
    response = llm.bind_tools(tools).invoke(state['messages'])
    return {"messages": [response]}

def tool_executor(state: ResearchState):
    """执行工具并提取结果到 search_results"""
    tool_node = ToolNode(tools=tools)
    tool_output = tool_node.invoke(state)
    tool_results = []
    for msg in tool_output["messages"]:
        if isinstance(msg, ToolMessage):
            import json
            tool_results.append(json.loads(msg.content))
    return {
        # "messages": tool_output["messages"],
        "search_results": tool_results
    }
    
def router(state: ResearchState) -> str:
    """路由：判断下一步是调用工具、直接结束还是进入质量检查"""
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    else:
        return "END"

final_graph_builder = StateGraph(ResearchState)

final_graph_builder.add_node("agent_node", agent_node)
final_graph_builder.add_node("tool_node", tool_executor)
final_graph_builder.add_node("quality_check_node", quality_check_node)
final_graph_builder.add_node("clarify_with_user_node", clarify_with_user_node)

final_graph_builder.add_node("generate_draft_node", generate_draft_node)
final_graph_builder.add_node("human_review_node", human_review_node)
final_graph_builder.add_node("finalize_report_node", finalize_report_node)

final_graph_builder.add_edge("generate_draft_node", "human_review_node")
final_graph_builder.add_edge("human_review_node", "finalize_report_node")
final_graph_builder.add_edge("finalize_report_node", END)

final_graph_builder.set_entry_point("agent_node")

final_graph_builder.add_conditional_edges(
    "agent_node", 
    router,
    {
        "tool_node": "tool_node",
        "END": END
    }
)


final_graph_builder.add_edge("tool_node", "quality_check_node")
final_graph_builder.add_edge("clarify_with_user_node", END)
final_checkpointer = MemorySaver()
app = final_graph_builder.compile(
    checkpointer=final_checkpointer,
    interrupt_before=["human_review_node", "clarify_with_user_node"],
)

async def run_collaborative_session():
    config = {"configurable": {"thread_id": "collab-thread-2"}}
    inputs = {"messages": [HumanMessage(content="对比一下LangGraph和传统的LangChain Agent在实现复杂工作流时的优劣势")]}

    print("--- [Session Start] ---")
    # 1. 启动图，它将运行直到第一个中断点
    async for output in app.astream(inputs, config=config):
        for key, value in output.items():
            print(f"Node '{key}' output: {value}")
    
    # 2. 检查中断状态
    current_state = await app.aget_state(config)
    
    # 检查是否在 human_review_node 中断
    if "human_review_node" in current_state.next:
        print("\n🚦 --- [Graph Interrupted for Human Review] --- 🚦")
        
        # 3. 从状态中提取AI生成的初稿
        draft_report = current_state.values.get("draft_report")
        
        # --- [API Boundary] Backend 将初稿发送给 Frontend ---
        print("\nAI 生成的报告初稿：")
        print("--------------------")
        print(draft_report)
        print("--------------------")
        
        # 4. 模拟用户在前端页面进行修改
        print("\n请在下方确认或修改报告内容。如果无需修改，直接按回车。")
        user_feedback = input("您的修改版本: ")
        
        # 如果用户没有输入，则使用原始初稿
        if not user_feedback.strip():
            final_draft = draft_report
            print("--- 用户已确认，使用原始初稿继续 ---")
        else:
            final_draft = user_feedback
            print("--- 用户已提交修改，使用新版本继续 ---")
            
        resume_inputs = {"draft_report": final_draft}
        
        print("\n--- [Session Resumed] ---")
        async for output in app.astream(resume_inputs, config=config):
            for key, value in output.items():
                print(f"Node '{key}' output: {value}")

    # 6. 获取并打印最终结果
    final_state = await app.aget_state(config)
    if not final_state.next:
        final_message = final_state.values["messages"][-1]
        print(final_message.content)

if __name__ == "__main__":
    asyncio.run(run_collaborative_session())
