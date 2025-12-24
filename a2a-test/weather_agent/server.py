import logging
import uvicorn
import httpx
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent import WeatherAgent
from executor import WeatherAgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():
    host = "localhost"
    port = 8008

    # 1. 定义能力
    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    
    skill = AgentSkill(
        id="check_weather",
        name="Check Weather",
        description="Check weather for a specific city",
        examples=["What is the weather in Beijing?"],
        tags=["weather", "info"],
    )

    # 2. 创建名片 (Card)
    agent_card = AgentCard(
        name="Weather Agent",
        description="A demo agent that checks weather",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=WeatherAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=WeatherAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    # 3. 初始化处理链
    httpx_client = httpx.AsyncClient()
    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx_client, 
        config_store=push_config_store
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(), # 注入我们的 Executor
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=push_sender
    )

    # 4. 构建 App
    server = A2AStarletteApplication(
        agent_card=agent_card, 
        http_handler=request_handler
    )

    print(f"Starting Weather Agent on http://{host}:{port}")
    uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    main()