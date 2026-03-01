import os
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class IntentAnalysis(BaseModel):
    intent: Literal["refund", "status_check", "human_service", "others"]
    confidence: float = Field(ge=0, le=1)
    refined_query: str = Field(description="去除语气词并结合上下文重写后的干净问题")

def get_llm_intent(user_input: str, history: str = "") -> IntentAnalysis:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    
    prompt = f"""对话历史: {history}
用户当前输入: {user_input}

请分析用户真实意图并重写问题。
请以 JSON 格式输出，包含以下字段：
- intent: 意图类型，可选值为 refund, status_check, human_service, others
- confidence: 置信度，范围 0-1
- refined_query: 去除语气词并结合上下文重写后的干净问题

只输出 JSON，不要输出其他内容。"""

    # 使用 JSON Mode 强制结构化输出
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[{"role": "system", "content": "你是一个意图识别专家。请只输出 JSON 格式的结果。"},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    # 手动解析 JSON
    content = completion.choices[0].message.content
    data = json.loads(content)
    return IntentAnalysis(**data)

# 测试
print(get_llm_intent("那它呢？", history="用户：查一下顺丰单号123。AI：正在为您查询。"))