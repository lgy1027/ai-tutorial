from tavily import TavilyClient
from typing import Literal
from langchain_openai import ChatOpenAI  
import os
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.types import Command

model = ChatOpenAI(model="Qwen3-235B-A22B", temperature=0, api_key="sk-dIl9oEE1SCJHXkzkdTmivPJgtxMGHNgvvNx5e17T4XYHBBOG", base_url="http://10.1.18.99:8089/v1")

os.environ["TAVILY_API_KEY"] = "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"

tavily_client = TavilyClient()

def internet_search(
    query: str,
    max_results: int = 2, 
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = True, 
):
    """
    执行互联网搜索。
    """
    print(f"\n[Tool Call] 正在搜索: {query}...")
    response = tavily_client.search(
        query, 
        max_results=max_results, 
        include_raw_content=include_raw_content, 
        topic=topic
    )
    
    if "results" in response:
        for res in response["results"]:
            raw_content = res.get("raw_content") or ""
            
            content_len = len(raw_content)
            # 如果内容不足进行填充以触发拦截
            if content_len < 50000:
                res["raw_content"] = raw_content + (" [PADDING_DATA] " * 5000)
                
    return response

# 初始化全局 Store (生产环境可替换为 PostgresStore)
global_store = InMemoryStore()

def hybrid_backend_factory(runtime):
    """
    工厂函数：DeepAgents 运行时会自动调用此函数创建后端实例。
    """
    return CompositeBackend(
        # 默认路由：临时文件存入 State
        default=StateBackend(runtime),
        
        # 路由规则：以 /memories/ 开头的路径存入 Store
        routes={
            "/memories/": StoreBackend(runtime)
        }
    )

# 子智能体配置字典
research_subagent_config = {
    "name": "deep_researcher",
    "description": "专门用于执行复杂的互联网信息检索和分析任务。",
    "system_prompt": """你是一个严谨的研究员。
    你的任务是：
    1. 使用 internet_search 工具搜索信息。
    2. 如果搜索结果被存入文件（Output saved to file...），请务必使用 read_file 读取关键部分。
    3. 将分析结果整理为摘要返回。""",
    "tools": [internet_search], # 子智能体独占搜索工具
    "model": model 
}

# 定义 Checkpointer (人机协同必须配置 Checkpointer 以保存断点状态)
checkpointer = MemorySaver()

# 3. 创建 Deep Agent
agent = create_deep_agent(
    model=model,
    tools=[], # 主 Agent 不直接持有搜索工具
    store=global_store, # 注入持久化存储
    
    # 注入混合后端工厂
    backend=hybrid_backend_factory,
    
    # 注册子智能体
    subagents=[research_subagent_config],
    
    # 配置 Human-in-the-loop 中断策略
    interrupt_on={
        # 对 task 工具（唤起子智能体）进行审核
        "task": {"allowed_decisions": ["approve", "reject"]},
        
        # 对写文件进行强审核，允许修改写入内容 ("edit")
        "write_file": {"allowed_decisions": ["approve", "reject", "edit"]} 
    },
    
    checkpointer=checkpointer,
    
    system_prompt="""你是项目经理。
    1. 遇到调研任务，必须使用 task 工具委派给 deep_researcher。
    2. 将调研的草稿文件保存在根目录（如 /draft.md）。
    3. 将最终的重要结论，必须写入 /memories/ 目录（如 /memories/report.txt）。
    """
)

# 生成唯一的线程 ID
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print(f"--- 启动 Deep Agents 会话: {thread_id} ---")

# 1. 第一次调用：发送任务
# 预期行为：Agent 会思考 -> 决定调用 task 唤起子智能体 -> 触发中断 -> 暂停
print("\n>>> 用户发送指令...")
result = agent.invoke(
    {"messages": [("user", "请调研 LangGraph 的核心架构优势，整理成简报，并保存到我的长期记忆库中。")]},
    config=config
)

# 2. 检查是否有中断 (HITL)
if result.get("__interrupt__"):
    # 获取中断详情
    interrupt_info = result["__interrupt__"][0].value
    action_requests = interrupt_info["action_requests"]
    tool_name = action_requests[0]["name"]
    
    print(f"\n[人机协同] ✋ Agent 请求执行操作: {tool_name}")
    print(f"参数详情: {action_requests[0]['args']}")
    print("系统已暂停，等待您的批准...")
    
    # 3. 模拟人类批准操作
    # 在真实应用中，这里会是一个前端 UI 按钮
    user_decision = input(f"是否批准执行 {tool_name}? (y/n): ")
    
    if user_decision.lower() == "y":
        print(f"\n>>> 用户批准，恢复执行...")
        
        # 发送 Resume 指令，允许 Agent 继续
        # 注意：这里我们使用 Command 对象来恢复执行
        final_result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        
        # 4. 打印最终结果（包含大文件拦截日志）
        print("\n--- 最终执行结果 ---")
        for msg in final_result["messages"]:
            if msg.type == "tool" and "Output saved to" in str(msg.content):
                 print(f"[系统拦截] 检测到大文件，已自动转存: {msg.content}")
            elif msg.type == "ai":
                print(f"[AI]: {msg.content}")

# 验证持久化存储
print("\n--- 验证长期记忆存储 ---")
namespaces = (thread_id, "memories") 
memories = global_store.search(namespaces)

if memories:
    print(f"成功在数据库中发现 {len(memories)} 个持久化文件。")
    for mem in memories:
        print(f" - 文件路径: {mem.key}")
else:
    print("未发现持久化文件。")