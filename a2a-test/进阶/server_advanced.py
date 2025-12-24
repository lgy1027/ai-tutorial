from fastapi import FastAPI, Request, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn
import asyncio
import uuid
from typing import Optional

app = FastAPI()

# === 1. 模拟数据库 (内存存储) ===
# 在真实项目中，这里应该是 Redis 或 MySQL
# 结构: { "task_id": {"status": "processing"|"completed", "result": ...} }
TASK_STORE = {}

# === 2. 安全配置 (鉴权) ===
# 这是一个简单的 Token 验证，真实场景可用 OAuth2
API_TOKEN = "sk-my-secret-token-2025"

async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="无效的 Token，禁止访问")

# === 3. 数据校验模型 (Schema) ===
# 使用 Pydantic 严格定义输入，拒绝非法数据
class MathParams(BaseModel):
    a: float
    b: float

# === 4. 模拟耗时的 AI 任务 ===
async def heavy_ai_calculation(task_id: str, a: float, b: float):
    print(f"⚙️ [后台] 任务 {task_id} 开始处理...")
    await asyncio.sleep(5)  # 模拟 AI 思考 5 秒
    result = a + b
    # 更新任务状态
    TASK_STORE[task_id] = {"status": "completed", "result": result}
    print(f"✅ [后台] 任务 {task_id} 完成，结果: {result}")

# === 5. A2A 接口实现 ===

# 依然要暴露名片，这次我们不需要鉴权，公开可读
@app.get("/.well-known/agent.json")
async def get_agent_card():
    return {
        "version": "1.0.0",
        "metadata": {"name": "ProMathExpert", "auth_required": True},
        "capabilities": [
            {
                "name": "async_add",
                "description": "异步计算加法（需鉴权）",
                "mode": "asynchronous" # 声明这是异步能力
            }
        ]
    }

# 查询任务状态的接口 (A2A 常见模式: Task Polling)
@app.get("/agent/tasks/{task_id}", dependencies=[Depends(verify_token)])
async def get_task_status(task_id: str):
    task = TASK_STORE.get(task_id)
    if not task:
        return {"status": "not_found"}
    return task

# RPC 入口，增加了 dependencies=[Depends(verify_token)] 进行安检
@app.post("/agent/rpc", dependencies=[Depends(verify_token)])
async def handle_rpc(request: Request):
    data = await request.json()
    method = data.get("method")
    params_raw = data.get("params")
    
    # 异步模式：不直接返回结果，而是返回 Task ID
    if method == "async_add":
        try:
            # 1. 校验数据
            valid_params = MathParams(**params_raw)
            
            # 2. 生成任务 ID
            task_id = str(uuid.uuid4())
            
            # 3. 初始状态写入数据库
            TASK_STORE[task_id] = {"status": "processing", "result": None}
            
            # 4. 启动后台任务 (不阻塞 HTTP 返回)
            asyncio.create_task(heavy_ai_calculation(task_id, valid_params.a, valid_params.b))
            
            # 5. 立即告诉客户端：任务接单了！
            return {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {
                    "task_id": task_id,
                    "message": "任务已接受，请稍后查询",
                    "status_endpoint": f"/agent/tasks/{task_id}"
                }
            }
            
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": 500, "message": str(e)}}
    
    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)