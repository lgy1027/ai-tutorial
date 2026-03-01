import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import random
from typing import List, Dict
from sklearn.model_selection import train_test_split

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(CURRENT_DIR, "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = os.getenv("OPENAI_MODEL")

# 意图定义：增加“冲突描述”，让生成的数据更具区分度
INTENT_CONFIG = {
    "order_query": {
        "desc": "查询物流状态、查单号、问发货没、看快递到哪了",
        "keywords": ["单号", "快递", "物流", "到哪了", "发货"],
        "confusing_with": "refund_apply" # 易混淆意图
    },
    "refund_apply": {
        "desc": "申请退款、退货、不想要了、退钱、退货流程",
        "keywords": ["退款", "退货", "钱", "不想要", "流程"],
        "confusing_with": "order_query"
    },
    "addr_modify": {
        "desc": "改收货地址、改电话、改名字、改派送时间",
        "keywords": ["地址", "电话", "姓名", "改一下", "写错了"],
        "confusing_with": "order_query"
    },
    "oos": {
        "desc": "无关闲聊、评价产品好坏、竞品对比、AI身份问询、无意义乱码",
        "keywords": ["你好", "厉害", "笨", "对比", "价格"],
        "confusing_with": "None"
    }
}

# 场景模拟：确保数据覆盖真各种“脏数据”
SCENARIOS = [
    "极其简短的碎片化表达（如：发货没、改地址）",
    "带有强烈情绪或错别字的口语（如：tm到底什么时候发、地址写错啦快改）",
    "包含指代不明的表达（如：那个帮我查下、这个也一样处理）",
    "包含干扰词的表达（如：虽然我想退款，但先查下物流吧 —— 这种属于多意图，需按主意图生成）"
]

def generate_samples(label: str, config: Dict, count: int) -> List[str]:
    print(f"正在为意图 [{label}] 生成 {count} 条高质量样本...")
    all_questions = []
    
    # 分批生成，避免一次性请求过多导致多样性下降
    batch_size = 10
    for i in range(0, count, batch_size):
        scenario = random.choice(SCENARIOS)
        prompt = f"""
        你是一个电商语料专家。请为意图【{label}】生成{batch_size}条中文用户输入。
        意图定义：{config['desc']}
        目标场景：{scenario}
        
        注意：
        1. 必须包含关键词干扰：尝试在句子中加入 {config['confusing_with']} 相关的词汇，但保持真实意图仍是 {label}。
        2. 模拟真实性：加入语气词（那个、额、哈）、错别字、不规范标点。
        3. 绝对不要输出任何编号或解释，只输出一个纯 JSON 数组，如：["样本1", "样本2"]
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": "你是一个JSON格式语料生成器。"},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            raw = json.loads(response.choices[0].message.content)
            
            # 鲁棒性提取列表
            questions = []
            if isinstance(raw, list): questions = raw
            else: questions = raw.get("samples", next(iter(raw.values())) if raw else [])
            
            if isinstance(questions, list):
                all_questions.extend([str(q) for q in questions])
            
        except Exception as e:
            print(f"生成批次失败: {e}")
            
    return all_questions[:count]

def save_jsonl(data: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    total_dataset = []
    SAMPLES_PER_INTENT = 100 # 建议训练集每个类别至少200条
    
    for label, config in INTENT_CONFIG.items():
        samples = generate_samples(label, config, SAMPLES_PER_INTENT)
        for s in samples:
            total_dataset.append({"text": s, "label": label})
    
    # 科学切分：6:2:2
    train_data, temp_data = train_test_split(
        total_dataset, test_size=0.4, random_state=42, stratify=[d['label'] for d in total_dataset]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=[d['label'] for d in temp_data]
    )
    
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_data,  os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")

if __name__ == "__main__":
    main()