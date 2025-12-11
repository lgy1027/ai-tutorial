import time
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.types import Command, interrupt       
class AgentState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: AgentState):
    """
    一个需要人工批准的节点。
    
    注意：这个节点在“恢复”时会从头重新运行。
    """
    print("--- 节点 [approval_node] 开始执行 ---")
    print(f"  > 待办操作: {state['action_details']}")
    
    # 仅在 'pending' 状态时才触发中断
    if state['status'] == 'pending':
        # 3. 核心组件：调用 interrupt() 来暂停
        # 'payload' 将会返回给调用者
        payload = {
            "question": "您是否批准此操作？",
            "details": state["action_details"]
        }
        
        print("  > 暂停，等待人工批准...")
        # 第一次运行：Graph 在此暂停。
        # 恢复运行时：decision 将被赋予 Command(resume=...) 中的值。
        decision = interrupt(payload) 
        
        print(f"  > 收到人工决策: {decision}")
        
        # 根据决策更新状态
        if decision:
            return {"status": "approved"}
        else:
            return {"status": "rejected"}
    
    # 如果状态不是 'pending' (例如在重跑时)，则跳过
    print(f"  > 状态为 {state['status']}, 跳过中断。")
    return {}

def proceed_node(state: AgentState):
    print("--- 节点 [proceed_node] 执行 ---")
    print(f"  > 已批准，正在执行操作: {state['action_details']}")
    return {} 

def cancel_node(state: AgentState):
    print("--- 节点 [cancel_node] 执行 ---")
    print(f"  > 已拒绝，取消操作: {state['action_details']}")
    return {}

builder = StateGraph(AgentState)

# 添加节点
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)

# 设置入口
builder.set_entry_point("approval")

# 定义条件路由
def route_decision(state: AgentState):
    if state["status"] == "approved":
        return "proceed"
    else:
        return "cancel"

# 'approval' 节点完成后，根据 'status' 决定去向
builder.add_conditional_edges(
    "approval",
    route_decision,
    {
        "proceed": "proceed",
        "cancel": "cancel"
    }
)

# 最终节点
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# 1. 核心组件：实例化 Checkpointer
checkpointer = MemorySaver()

# 编译 Graph，必须传入 checkpointer
graph = builder.compile(checkpointer=checkpointer)

# 2. 核心组件：定义一个唯一的 thread_id
config = {"configurable": {"thread_id": "tx-12345"}}
initial_input = {"action_details": "向用户 'A' 转账 500RMB", "status": "pending"}

# --- 步骤 1: 第一次调用，触发暂停 ---
print("--- 第一次运行 (将触发暂停) ---")
# Graph 会运行到 approval_node，调用 interrupt()，然后暂停
result = graph.invoke(initial_input, config=config)

print("\n--- Graph 已暂停 ---")
print("  > Graph 的当前状态 (已保存):")
print(f"  > {graph.get_state(config).values}")
print("\n  > 'interrupt' 返回的数据 (用于展示给用户):")
# 注意这个特殊的 `__interrupt__` 字段
print(f"  > {result['__interrupt__']}") 

# ... # ... 此时，你的应用程序（如 Web UI）会向用户展示 'result['__interrupt__']' 的内容
# ... 用户审查后，决定 "批准" (True)# ...
time.sleep(1) # 模拟人类思考时间
human_decision = True 
print(f"\n--- 用户已做出决策: {human_decision} ---")

# --- 步骤 2: 第二次调用，使用 Command 恢复 ---
print("--- 恢复 Graph 运行 ---")

# 4. 核心组件：使用 Command(resume=...) 和 *相同的 config* 来恢复
# # 传入的 `resume=True` 将成为 `interrupt()` 的返回值
resume_result = graph.invoke(Command(resume=human_decision), config=config)

print("\n--- Graph 运行完毕 ---")
print("  > Graph 的最终状态:")
print(f"  > {graph.get_state(config).values}")
print("\n  > 最后一步的输出:")
print(f"  > {resume_result}")


def approval_node(state: AgentState):
    print("--- 节点 [approval_node] 开始执行 ---")
    
    # 正确：使用状态来控制中断
    if state['status'] == 'pending':
        # 第一次运行：'pending'，触发中断
        decision = interrupt(...) 
        if decision:
            return {"status": "approved"}
        else:
            return {"status": "rejected"}
    
    # 第二次运行 (恢复后)：
    # 状态已是 'approved' 或 'rejected'# 'if' 条件为 False，中断被跳过
    print(f"  > 状态为 {state['status']}, 跳过中断。")
    return {}