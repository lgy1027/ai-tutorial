# A2A 协议教程

A2A（Agent-to-Agent）协议是用于智能体间通信的标准协议，实现多 Agent 协作。

## 目录结构

```
a2a-test/
├── client.py           # 客户端示例
├── 基础/               # 基础通信示例
│   ├── client_agent.py
│   └── server_agent.py
├── 进阶/               # 高级交互模式
│   ├── client_advanced.py
│   └── server_advanced.py
└── weather_agent/      # 完整天气查询 Agent
    ├── agent.py
    ├── executor.py
    └── server.py
```

## 核心流程

### 1. 发现 Agent
客户端通过 A2ACardResolver 发现并获取 Agent 信息。

```python
from a2a.client import A2ACardResolver

resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
card = await resolver.get_agent_card()
```

### 2. 初始化客户端
根据 Agent Card 初始化通信客户端。

```python
from a2a.client import A2AClient

client = A2AClient(httpx_client=http_client, agent_card=card)
```

### 3. 发送消息
构建消息请求并发送给 Agent。

```python
from a2a.types import MessageSendParams, SendMessageRequest

payload = {
    "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is the weather?"}],
        "message_id": uuid4().hex,
    }
}
request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
response = await client.send_message(request)
```

## 示例说明

### 基础示例
最简单的 Client-Server 通信演示。

### 进阶示例
展示更复杂的交互模式，包括流式响应和多轮对话。

### Weather Agent
完整的天气查询 Agent 实现：
- `agent.py` - Agent 核心逻辑
- `executor.py` - A2A 执行器适配
- `server.py` - 服务启动入口

## 快速开始

```bash
# 启动服务端
python weather_agent/server.py

# 运行客户端
python client.py
```

## 依赖

- a2a
- httpx
- langchain-openai
- langgraph
