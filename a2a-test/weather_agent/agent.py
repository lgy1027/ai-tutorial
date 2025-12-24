import os
from collections.abc import AsyncIterable
from typing import Any, Literal, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

# 使用内存存储会话状态
memory = MemorySaver()

# --- 1. 定义工具 (Mock) ---
@tool
def get_weather(city: str) -> Dict[str, Any]:
    """Get the current weather for a specific city.
    
    Args:
        city: The name of the city (e.g., "Beijing", "New York").
    """
    # 模拟数据，实际项目中这里会调用 API
    mock_data = {
        "Beijing": {"temp": 25, "condition": "Sunny"},
        "Shanghai": {"temp": 22, "condition": "Cloudy"},
        "New York": {"temp": 15, "condition": "Rainy"}
    }
    result = mock_data.get(city, {"temp": 20, "condition": "Unknown"})
    return {"city": city, **result}

# --- 2. 定义响应格式 ---
class ResponseFormat(BaseModel):
    """用于控制 Agent 返回给 A2A 协议的状态"""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

# --- 3. Agent 类 ---
class WeatherAgent:
    SYSTEM_INSTRUCTION = (
        "You are a helpful weather assistant. "
        "Use the 'get_weather' tool to find weather information. "
        "If you don't know the city, ask the user for it. "
    )
    
    FORMAT_INSTRUCTION = (
        "Set status to 'input_required' if you need more info from user. "
        "Set status to 'completed' if you have the answer. "
        "Set status to 'error' if something fails."
    )

    def __init__(self):
        # 请确保环境变量中设置了 OPENAI_API_KEY
        # 或者替换为你自己的模型配置
        self.model = ChatOpenAI(
            model="Qwen3-235B-A22B",
            base_url="http://10.1.18.99:8089/v1",
            temperature=0,
            openai_api_key="sk-"
        )
        self.tools = [get_weather]

        # 创建 LangGraph ReAct Agent

        self.graph = create_agent(
            self.model,
            tools=[get_weather],
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=ToolStrategy(
                schema=ResponseFormat,
                handle_errors=True  # <--- 开启自动纠错
            )
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """流式处理逻辑，适配 A2A Executor"""
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        # 遍历 LangGraph 的事件流
        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            
            # 如果 Agent 正在调用工具
            if isinstance(message, AIMessage) and message.tool_calls:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '正在查询天气数据...',
                }
            # 如果工具返回了结果
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '正在处理天气数据...',
                }

        # 获取最终的结构化响应
        yield self.get_agent_final_response(config)

    def get_agent_final_response(self, config):
        """解析 ResponseFormat 并转换为 executor 需要的字典"""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            return {
                'is_task_complete': structured_response.status == 'completed',
                'require_user_input': structured_response.status == 'input_required',
                'content': structured_response.message,
            }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': "抱歉，无法处理您的请求，请重试。"
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']