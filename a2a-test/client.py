import asyncio
import logging
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    base_url = "http://localhost:8008"

    async with httpx.AsyncClient() as http_client:
        # 1. 发现 Agent
        resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
        card = await resolver.get_agent_card()
        logger.info(f"Found Agent: {card.name}")

        # 2. 初始化客户端
        client = A2AClient(httpx_client=http_client, agent_card=card)

        # 3. 发送消息
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What is the weather in Beijing?"}],
                "message_id": uuid4().hex,
            }
        }
        
        request = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**payload)
        )

        logger.info("Sending request...")
        # 这里的 send_message 会等待任务直到处于 'completed' 或 'input_required' 状态
        response = await client.send_message(request)
        
        result = response.root.result
        print("\n--- Final Result ---")
        print(f"Status: {result.status}")
        
        # 打印最后一条消息的内容
        if result.items:
            last_msg = result.items[-1]
            if last_msg.parts:
                print(f"Response: {last_msg.parts[0].root.text}")

if __name__ == "__main__":
    asyncio.run(main())