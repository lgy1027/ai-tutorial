import json
import os
import random
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset")
INTENTS = ["order_query", "addr_modify", "refund_apply", "oos"]
ID2LABEL = {i: name for i, name in enumerate(INTENTS)}

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

def generate_multilabel_batch(mode="mixed", count=10):
    """
    mode: 'single' (单意图) 或 'mixed' (多意图融合)
    """
    if mode == "single":
        target = random.choice(INTENTS)
        prompt = f"""请生成 {count} 条纯粹的电商客服语料，意图仅限于：【{target}】。
        格式要求：只输出 JSON 数组，每个元素包含 "text" 和 "labels" (One-Hot向量，4位)。
        示例：{{"text": "刚才那个单子发哪了", "labels": [1, 0, 0, 0]}}"""
    else:
        # 随机抽取两个业务意图进行强行融合
        s1, s2 = random.sample([i for i in INTENTS if i != "oos"], 2)
        prompt = f"""请生成 {count} 条【深度融合】的复合意图语料。
        每句话必须【同时包含】两个诉求：1. {s1} 2. {s2}。
        不要分成两句话，要像真实用户一样揉在一起说。
        格式要求：只输出 JSON 数组，labels 必须有两个位置是 1。
        示例：{{"text": "我那个快递到哪了？顺便把地址改到上海。", "labels": [1, 1, 0, 0]}}"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": f"你是一个语料专家。标签顺序固定为：{INTENTS}。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        content = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            samples = json.loads(match.group())
            # 过滤掉不符合格式的条目
            valid_samples = [s for s in samples if "text" in s and "labels" in s and sum(s["labels"]) > 0]
            return valid_samples
        return []
    except Exception as e:
        print(f" 出错: {e}")
        return []

def main():
    all_data = []
    print(" 正在生成真正支持【多意图】的 One-Hot 数据集...")

    # 生成单意图数据
    for _ in tqdm(range(40), desc="📥 采集单意图"):
        all_data.extend(generate_multilabel_batch("single", 15))
        time.sleep(0.1)

    # 生成混合意图数据
    for _ in tqdm(range(30), desc=" 采集复合意图"):
        all_data.extend(generate_multilabel_batch("mixed", 15))
        time.sleep(0.1)

    if not all_data:
        print(" 未捕获到数据，请检查网络或 API。")
        return

    random.shuffle(all_data)
    
    # 统计单/多意图分布
    single_count = sum(1 for d in all_data if sum(d["labels"]) == 1)
    multi_count = sum(1 for d in all_data if sum(d["labels"]) > 1)
    print(f"\n 生成完毕！总数: {len(all_data)} | 单意图: {single_count} | 多意图: {multi_count}")

    # 切分保存 (8:1:1)
    os.makedirs(DATA_DIR, exist_ok=True)
    train_idx = int(len(all_data) * 0.8)
    val_idx = int(len(all_data) * 0.9)
    
    files = {
        "train.jsonl": all_data[:train_idx],
        "val.jsonl": all_data[train_idx:val_idx],
        "test.jsonl": all_data[val_idx:]
    }

    for filename, data in files.items():
        with open(os.path.join(DATA_DIR, filename), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f" 已保存: {filename}")

if __name__ == "__main__":
    main()