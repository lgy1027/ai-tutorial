import requests
import json
import uuid

# 目标 Agent 的地址
TARGET_AGENT_URL = "http://127.0.0.1:8001"

def discover_agent():
    """阶段一：发现 (Discovery)"""
    try:
        # A2A 标准：先查户口
        response = requests.get(f"{TARGET_AGENT_URL}/.well-known/agent.json")
        card = response.json()
        print(f"发现 Agent: {card['metadata']['name']}")
        print(f"能力: {[cap['name'] for cap in card['capabilities']]}")
        return True
    except Exception as e:
        print(f"无法连接 Agent: {e}")
        return False

def call_agent_capability(a, b):
    """阶段二：交互 (Interaction via JSON-RPC)"""
    
    # 构造标准的 JSON-RPC 请求包
    payload = {
        "jsonrpc": "2.0",
        "method": "add",  # 对应 Agent Card 里的能力名称
        "params": {"a": a, "b": b},
        "id": str(uuid.uuid4())
    }
    
    print(f"\n📤 发送任务: 计算 {a} + {b} ...")
    
    response = requests.post(f"{TARGET_AGENT_URL}/agent/rpc", json=payload)
    response_data = response.json()
    
    if "error" in response_data:
        print(f"任务失败: {response_data['error']}")
    else:
        print(f"任务完成，结果: {response_data['result']}")

if __name__ == "__main__":
    # 模拟流程
    if discover_agent():
        call_agent_capability(10, 55)