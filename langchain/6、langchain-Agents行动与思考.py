from langchain_core.tools import tool,Tool
from pydantic import BaseModel, Field
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor # 核心的Agent执行器
from langchain.agents.format_scratchpad import format_to_openai_function_messages # 格式化中间步骤
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser # 解析LLM输出
from langchain_core.runnables import RunnablePassthrough
import os
# 1. 定义一个简单的计算器工具
@tool
def add(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b

print("--- @tool 装饰器示例 ---")
print(f"add(2, 3) 结果: {add.invoke({'a': 2, 'b': 3})}") # 工具也可以直接invoke
print(f"multiply(4, 5) 结果: {multiply.invoke({'a': 4, 'b': 5})}")
print(f"工具名称: {add.name}")
print(f"工具描述: {add.description}\n")

# 定义一个Pydantic模型来描述输入参数
class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

def _run_search(query: str) -> str:
    """这是一个模拟的网页搜索函数"""
    # 实际生产中这里会调用搜索引擎API
    if "天气" in query:
        return "北京明天晴，气温20-30度。"
    elif "LangChain" in query:
        return "LangChain是一个用于LLM应用开发的框架。"
    else:
        return "没有找到相关信息。"

search_tool = Tool(
    name="web_search",
    description="Search the web for information. Use this tool for general knowledge questions or current events.",
    func=_run_search,
    args_schema=SearchInput # 使用Pydantic模型定义输入
)

print("--- Tool 类示例 ---")
print(f"search_tool.invoke('北京明天天气') 结果: {search_tool.invoke('北京明天天气')}\n")

# 初始化维基百科工具
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh",top_k_results=1))

print("--- 预构建工具示例 (Wikipedia) ---")
try:
    wiki_result = wiki_tool.invoke("马斯克")
    print(f"马斯克维基百科 (部分): {wiki_result}...\n")
except Exception as e:
    print(f"Wikipedia工具调用失败，可能需要安装 'wikipedia' 库或网络问题: {e}\n")
    # pip install wikipedia

load_dotenv()
llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
tools = [add, multiply] # 将工具列表传入

# 1. 定义 Agent 的提示模板
# MessagesPlaceholder("agent_scratchpad") 是关键，它会插入LLM的思考过程和工具执行结果
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学助手，可以使用工具进行加法和乘法运算。"),
    MessagesPlaceholder(variable_name="chat_history"), # 可以选择性加入聊天历史
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # LLM的思考过程和工具输出会填充这里
])

# 2. 将LLM绑定到工具 (这是 OpenAI Function Calling 的核心)
# LLM 会自动知道如何使用这些工具，并在需要时生成工具调用请求
llm_with_tools = llm.bind_tools(tools)

# 3. 构建 Agent 核心逻辑 (LCEL 链)
# 这一步是关键！它定义了 Agent 的思考循环
agent_runnable = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
) | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()


# 4. 创建 AgentExecutor
# AgentExecutor 是实际运行 Agent 的组件，它会处理 LLM 的响应 (是文本还是工具调用)，
# 如果是工具调用，它会执行工具并把结果反馈给 LLM，直到任务完成。
agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True) # verbose=True 会打印详细的思考过程

test_input = "200加300是多少？"
response = agent_executor.invoke({"input": test_input, "chat_history": []})
print(response)  # 应返回 500

print("--- 简单的计算器 Agent 示例 ---")
print("Agent 正在运行中...")
# 调用 Agent
response_calc1 = agent_executor.invoke({"input": "256乘以48是多少？", "chat_history": []})
print(f"问题: 256乘以48是多少？")
print(f"Agent 回答: {response_calc1['output']}\n")

response_calc2 = agent_executor.invoke({"input": "200加300再乘以2是多少？", "chat_history": []})
print(f"问题: 200加300再乘以2是多少？")
print(f"Agent 回答: {response_calc2['output']}\n")

response_no_tool = agent_executor.invoke({"input": "你好，能帮我做点什么？", "chat_history": []})
print(f"问题: 你好，能帮我做点什么？")
print(f"Agent 回答: {response_no_tool['output']}\n") # 不需要工具，直接回答