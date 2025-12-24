from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# 定义 Agent Card, A2A 协议规范要求暴露 capabilities 和 version
AGENT_CARD = {
    "version": "1.0.0",
    "metadata": {
        "name": "MathExpert",
        "description": "我是一个擅长处理基础数学运算的 Agent。",
        "id": "agent-math-001"
    },
    "capabilities": [
        {
            "name": "add",
            "description": "计算两个数字的和",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }
        }
    ]
}

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD

# 2. 具体的业务逻辑
def do_math_add(params):
    return params["a"] + params["b"]

# 3. 实现 A2A 的通信接口
@app.post("/agent/rpc")
async def handle_rpc(request: Request):
    data = await request.json()
    
    # 解析 JSON-RPC 标准包
    method = data.get("method")
    params = data.get("params")
    msg_id = data.get("id")
    
    result = None
    error = None

    # 简单的路由分发
    if method == "add":
        try:
            # 真实场景中这里通常会调用 LLM 或复杂逻辑
            result = do_math_add(params)
        except Exception as e:
            error = {"code": 500, "message": str(e)}
    else:
        error = {"code": -32601, "message": "Method not found"}

    # 返回标准的 JSON-RPC 响应
    response = {
        "jsonrpc": "2.0",
        "id": msg_id,
    }
    if error:
        response["error"] = error
    else:
        response["result"] = result
        
    return response

if __name__ == "__main__":
    print("启动 MathExpert Agent，端口 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8001)