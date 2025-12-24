import requests
import time
import uuid

TARGET_URL = "http://127.0.0.1:8000"
MY_TOKEN = "sk-my-secret-token-2025"

def run_async_task(a, b):
    headers = {
        "Authorization": f"Bearer {MY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 2. 发起请求
    payload = {
        "jsonrpc": "2.0",
        "method": "async_add",
        "params": {"a": a, "b": b},
        "id": str(uuid.uuid4())
    }
    
    print(f"[1/3] 提交计算请求: {a} + {b} ...")
    try:
        resp = requests.post(f"{TARGET_URL}/agent/rpc", json=payload, headers=headers)
        
        if resp.status_code == 401:
            print("鉴权失败！请检查 Token。")
            return

        data = resp.json()
        
        # 3. 获取任务信息
        result_data = data.get("result", {})
        task_id = result_data.get("task_id")
        print(f"任务提交成功，获得 Task ID: {task_id}")
        
        # 4. 开始轮询 (Polling) - 等待结果
        print("[2/3] 等待结果中...", end="", flush=True)
        while True:
            # 查询状态
            status_resp = requests.get(f"{TARGET_URL}/agent/tasks/{task_id}", headers=headers)
            status_data = status_resp.json()
            
            if status_data["status"] == "completed":
                print("\n[3/3] 任务完成！")
                print(f"最终结果: {status_data['result']}")
                break
            elif status_data["status"] == "not_found":
                print("\n任务丢失！")
                break
            
            # 没完成就等 1 秒再问
            print(".", end="", flush=True)
            time.sleep(1)
            
    except Exception as e:
        print(f"\n发生错误: {e}")

if __name__ == "__main__":
    # 尝试故意传个字符串测试校验，或者传数字测试正常流程
    run_async_task(100.5, 200.5)