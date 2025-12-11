import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询实时天气"""
    return f"{city} 天气晴朗，气温 25°C"

# 2. 定义模型 (支持静态配置)
model = ChatOpenAI(model="Qwen3-235B-A22B", temperature=0, api_key="sk-dIl9oEE1SCJHXkzkdTmivPJgtxMGHNgvvNx5e17T4XYHBBOG", base_url="http://10.1.18.99:8089/v1")


# 定义上下文结构
@dataclass
class UserContext:
    user_id: str
    is_vip: bool

@tool
def check_balance(runtime: ToolRuntime[UserContext]) -> str:
    """
    查询余额。
    注意：runtime 参数对 LLM 隐身，LLM 认为此工具不需要参数！
    """
    # 直接从运行时获取上下文，无需模型传参
    ctx = runtime.context
    
    # 模拟逻辑
    base_balance = 100
    if ctx.is_vip:
        return f"尊贵的 VIP 用户 {ctx.user_id}，您的余额是 {base_balance * 10} 元"
    return f"用户 {ctx.user_id}，您的余额是 {base_balance} 元"

# --- 调用环节 ---
# 假设这是从 API 网关获取的当前用户信息
current_user = UserContext(user_id="alice_888", is_vip=True)


# 3. 创建 Agent (底层自动构建 Graph)
agent = create_agent(model, tools=[get_weather, check_balance])

# 4. 运行
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我还有多少钱？"}]},
    context=current_user  # <--- 关键注入点
)
print(result["messages"][-1].content)


from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

# 定义期望的数据结构
class SentimentReport(BaseModel):
    score: int = Field(description="情感评分 1-10")
    tags: list[str] = Field(description="情感关键词，如：愤怒、开心")

agent = create_agent(
    model=model,
    tools=[], 
    # 核心：绑定结构 + 开启错误处理
    response_format=ToolStrategy(
        schema=SentimentReport,
        handle_errors=True  # <--- 开启自动纠错
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "这个产品太烂了，物流慢得要死！"}]
})

# 直接获取强类型对象
report = result["structured_response"]
print(f"评分: {report.score}, 标签: {report.tags}")
# 输出: 评分: 2, 标签: ['愤怒', '失望']


from langgraph.checkpoint.memory import InMemorySaver 
# 生产环境推荐使用: from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, check_balance],
    checkpointer=checkpointer # 挂载存档器
)

# 第一次对话，指定 thread_id
config = {"configurable": {"thread_id": "session_1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "我叫云枢"}]}, config)
print(result["messages"][-1].content)

# 第二次对话，它依然记得你
result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config)
print(result["messages"][-1].content)
# AI: "你叫 云枢"

from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def dynamic_system_message(request: ModelRequest) -> str:# 从 context 中获取用户等级
    level = request.runtime.context.get("level", "junior")
    
    base_prompt = "你是一个 Python 专家。" 
    if level == "senior":
        return base_prompt + "请直接给出极简的高级代码，不要废话。"
    return base_prompt + "请像老师一样详细解释每一行代码。"# 挂载中间件

agent = create_agent(
    model=model, 
    middleware=[dynamic_system_message],
)

result = agent.invoke({"messages": [{"role": "user", "content": "请写一个斐波那契数列的函数"}]}, context={"level": "senior"})
print(result["messages"][-1].content)

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    return f"{city} 今日天气晴朗，气温 25°C，适合出游。"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

agent = create_agent(model, tools=[get_weather, calculator])

input_msg = {
    "messages": [
        HumanMessage(content="帮我查一下北京的天气，然后根据气温计算一下：如果每度气温需要喝 10ml 水，我今天需要喝多少水？")
    ]
}

print(f"--- 🚀 开始处理请求: {input_msg['messages'][0].content} ---")

# stream_mode="updates" 意味着每当 Graph 中的一个节点完成工作，就推送一次更新
for chunk in agent.stream(input_msg, stream_mode="updates"):
    for node, update in chunk.items():
        
        # -------------------------------------------------
        # 场景 A: 捕获 Agent 节点的动作 (模型思考 & 决定)
        # -------------------------------------------------
        if node == "model":
            if "messages" in update:
                ai_msg = update["messages"][-1]
                
                # 1. 模型决定调用工具
                if ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        print(f"\n🤖 [Agent 思考] 决定调用工具: {tool_call['name']}")
                        print(f"    └─ 参数: {tool_call['args']}")
                
                # 2. 模型直接回复 (或思考过程)
                elif ai_msg.content:
                    # 注意：有些模型在调用工具前也会输出一段 content 文本
                    print(f"\n💬 [Agent 回复]: {ai_msg.content}")

        # -------------------------------------------------
        # 场景 B: 捕获 Tools 节点的动作 (工具实际执行结果)
        # -------------------------------------------------
        elif node == "tools":
            if "messages" in update:
                tool_msg = update["messages"][-1]
                
                print(f"\n🛠️ [Tools 执行] 工具运行完毕")
                # tool_msg.content 就是工具函数的 return 值
                print(f"    └─ 结果: {tool_msg.content}")

print("\n--- ✅ 流程结束 ---")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langgraph.errors import GraphInterrupt

@tool
def request_refund(amount: int) -> str:
    """发起退款申请。"""
    print(f"\n[Tool] 1. 收到退款请求: {amount}元")
    print("[Tool] 2. 正在请求人工审核，程序将在此暂停...")

    approval_result = interrupt(f"申请退款 {amount} 元，请管理员批示")

    # --- 恢复执行后，从这里继续跑 ---
    print(f"[Tool] 3. 收到审核结果: {approval_result}")

    if approval_result == "审核通过":
        # 逻辑没有断！我们可以继续调用其他函数
        result = execute_refund_transaction(amount)
        return f"审核通过，{result}"
    else:
        return f"审核拒绝。原因: {approval_result}"

def execute_refund_transaction(amount: int) -> str:
    """模拟银行转账逻辑"""
    print(f"\n⚡ [Bank API] 正在执行转账: {amount}元...")
    return f"转账成功：{amount}元已退回用户账户。"


# 初始化内存存储（生产环境通常使用 PostgresSaver）
# 如果没有 checkpointer，中断后状态就会丢失，无法 resume
checkpointer = InMemorySaver()

agent = create_agent(
    model, 
    tools=[request_refund], 
    checkpointer=checkpointer
)

thread_config = {"configurable": {"thread_id": "tx_123"}}

print("--- 阶段 1: 用户发起请求 ---")

try:
    # Agent 会思考 -> 调用 request_refund -> 触发 Interrupt -> 抛出 GraphInterrupt 异常
    agent.invoke(
        {"messages": [{"role": "user", "content": "我要退款 100 元"}]}, 
        thread_config
    )
except GraphInterrupt as e:
    # 捕获中断异常
    print(f"⏸️  任务已暂停! 收到中断信号: {e}")
    print("    (当前状态已保存到内存中)")


print("\n--- 阶段 2: 人工审核 ---")
# 这里模拟管理员在控制台输入，实际场景可能是前端的一个按钮
user_approval = input("管理员：批准退款吗？(输入 y 批准，其他拒绝): ")

# 决定恢复执行时的返回值
if user_approval.lower() == "y":
    resume_value = "审核通过"
    print("    -> 管理员已批准。")
else:
    resume_value = "审核拒绝：金额过大"
    print("    -> 管理员已拒绝。")

print("\n--- 🔄 阶段 3: 恢复执行 ---")

# 使用 Command(resume=...) 恢复执行
# 这里的 resume_value 会直接作为 request_refund 工具的“返回值”给到 LLM
# 此时 LLM 看到的历史是：
# User: 退款 100 -> AI: 调用 request_refund -> Tool Output: "审核通过" (我们注入的值)
result = agent.invoke(
    Command(resume=resume_value), 
    thread_config
)


print("\n--- ✅ 最终结果 ---")
# 打印最后一条消息内容
last_message = result["messages"][-1]
print(f"AI 回复: {last_message.content}")

from typing import List, Annotated
from typing_extensions import TypedDict

from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

# 1. 定义状态 Schema

class AnalystState(AgentState):
    scratchpad: List[str]  # 扩展字段：草稿本

# 2. 定义工具：写 & 读

@tool
def batch_save_notes(
    notes: List[str], 
    # 使用标准注入方式获取 ID 和 状态
    tool_call_id: Annotated[str, InjectedToolCallId], 
    state: Annotated[AnalystState, InjectedState]
) -> Command:
    """
    【批量写入工具】将多个关键发现一次性记录到草稿本。
    参数 notes: 一个字符串列表，例如 ["项目A预算增加", "项目B取消"]
    """
    # 1. 获取当前草稿本
    current_pad = state.get("scratchpad", [])
    
    # 2. 批量追加
    new_pad = current_pad + notes
    
    print(f"📝 [Notepad] 正在批量记录 {len(notes)} 条数据...")
    for n in notes:
        print(f"   - {n}")

    # 3. 更新状态
    return Command(
        update={
            "scratchpad": new_pad,
            "messages": [
                ToolMessage(
                    content=f"成功批量记录了 {len(notes)} 条笔记。",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def read_notes(
    # 读取也需要获取 State
    state: Annotated[AnalystState, InjectedState]
) -> str:
    """
    【读取工具】读取草稿本中的所有内容。
    """
    current_pad = state.get("scratchpad", [])
    
    if not current_pad:
        return "草稿本是空的。"
    
    print("📖 [Notepad] Agent 正在回顾笔记...")
    
    formatted_notes = "\n".join([f"{i+1}. {note}" for i, note in enumerate(current_pad)])
    return f"--- 草稿本内容 ---\n{formatted_notes}\n----------------"

system_prompt = """
你是一名从不遗漏细节的数据录入员。
你的任务是提取用户输入中的**所有事实数据**并存入草稿本。

【执行规则】
1. 分析输入，将其拆解为独立的“事实点”。
2. **必须**调用 `batch_save_notes` 工具，将这些事实点作为一个**列表**一次性存入。
3. 如果输入包含多个项目（如项目A、B、C），你的列表里必须包含对应的多条数据。

【❌ 严禁事项】
1. 严禁记录“用户让我分析...”或“处理复杂信息...”这类指令。只记录**业务数据**。
2. 严禁凭空捏造数据。
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[batch_save_notes, read_notes],    
    state_schema=AnalystState,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "analysis_session_1"}}

# 初始化状态
initial_state = {
    "messages": [],
    "scratchpad": [] # 初始化为空
}


# 模拟第一步：让 Agent 分析一段复杂的文本并记录
# 这里的 invoke 会触发 Agent 思考 -> 调用 save_note -> 更新 State
user_input_1 = """
请分析以下会议记录并记录关键点：
项目A的预算从50万增加到了80万，截止日期推迟到了12月31日。
项目B已被取消，资源转移到了项目C。
项目C现在的负责人变成了Alice，预算为20万。
"""

agent.invoke(
    {"messages": [{"role": "user", "content": user_input_1}], **initial_state},
    config
)

# 模拟第二步：追加信息
# 注意：我们不需要把上一轮的 user_input_1 再传一遍，State 已经在 Graph 里了
print("\n--- 🔄 追加信息 ---")
user_input_2 = "补充一点：项目A的负责人还是Bob，但他下个月要离职。"
agent.invoke(
    {"messages": [{"role": "user", "content": user_input_2}]},
    config
)

# 模拟第三步：最终汇总
# Agent 应该会先调用 read_notes，然后再回答
print("\n--- 📊 请求汇总 ---")
final_response = agent.invoke(
    {"messages": [{"role": "user", "content": "好了，现在根据你草稿本里的内容，给我生成一份最终的项目状态报告。"}]},
    config
)

print("\n--- ✅ 最终报告 ---")
print(final_response["messages"][-1].content)

# 验证：我们可以直接从 State 中查看草稿本，看看它存了什么
print("\n--- 🕵️‍♂️ (后台数据检查) 草稿本内容 ---")
final_state = agent.get_state(config)
print(json.dumps(final_state.values.get("scratchpad"), indent=2, ensure_ascii=False))


# [
#   "项目A预算增加",
#   "项目B取消",
#   "项目C延期至下季度",
#   "项目A负责人是Bob",
#   "Bob下个月将离职"
# ]