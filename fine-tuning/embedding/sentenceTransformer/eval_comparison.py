import json
import os
import torch
import random
from sentence_transformers import SentenceTransformer, util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 你的微调模型路径
FINETUNED_MODEL_PATH = os.path.join(CURRENT_DIR, "output_model_final")
# 基座模型名称
BASE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# 数据集路径
DATA_FILE = os.path.join(CURRENT_DIR, "finetune_data_mined.jsonl")

# BGE 指令
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

def load_test_cases(file_path, num_cases=3):
    """从数据集中随机抽取几个案例"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return random.sample(data, min(num_cases, len(data)))

def compare_models():
    print("⏳ 正在加载模型 (这可能需要一点时间)...")
    
    # 加载基座模型 (CPU跑推理就够了，不用GPU也行)
    base_model = SentenceTransformer(BASE_MODEL_NAME)
    
    # 加载微调后的模型
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"❌ 找不到微调模型: {FINETUNED_MODEL_PATH}")
        return
    ft_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    
    # 获取测试用例
    test_cases = load_test_cases(DATA_FILE, num_cases=3)
    
    print("\n" + "="*50)
    print("🚀 微调效果大比拼")
    print("="*50)

    for i, case in enumerate(test_cases):
        query = case['query']
        pos_doc = case['pos'][0]
        # 挑选一个最难的负例 (Hard Negative)
        neg_doc = case['neg'][0] if case.get('neg') else "无负例数据"
        
        print(f"\n📄 [案例 {i+1}] 用户提问: {query}")
        print(f"✅ 正确答案片段 (Positive): {pos_doc[:30]}...")
        print(f"❌ 相似干扰片段 (Negative): {neg_doc[:30]}...")
        
        # --- 1. 基座模型打分 ---
        # 注意：Query 要加指令，Doc 不需要
        q_emb_base = base_model.encode(QUERY_INSTRUCTION + query, convert_to_tensor=True)
        p_emb_base = base_model.encode(pos_doc, convert_to_tensor=True)
        n_emb_base = base_model.encode(neg_doc, convert_to_tensor=True)
        
        score_pos_base = util.cos_sim(q_emb_base, p_emb_base).item()
        score_neg_base = util.cos_sim(q_emb_base, n_emb_base).item()
        
        # --- 2. 微调模型打分 ---
        q_emb_ft = ft_model.encode(QUERY_INSTRUCTION + query, convert_to_tensor=True)
        p_emb_ft = ft_model.encode(pos_doc, convert_to_tensor=True)
        n_emb_ft = ft_model.encode(neg_doc, convert_to_tensor=True)
        
        score_pos_ft = util.cos_sim(q_emb_ft, p_emb_ft).item()
        score_neg_ft = util.cos_sim(q_emb_ft, n_emb_ft).item()
        
        print("-" * 30)
        print(f"🤖 基座模型评分:")
        print(f"   - 正确答案相似度: {score_pos_base:.4f}")
        print(f"   - 干扰项相似度:   {score_neg_base:.4f}")
        diff_base = score_pos_base - score_neg_base
        print(f"   👉 区分度 (正-负): {diff_base:.4f} {'⚠️ 危险' if diff_base < 0.05 else ''}")
        
        print(f"🔥 微调模型评分:")
        print(f"   - 正确答案相似度: {score_pos_ft:.4f}")
        print(f"   - 干扰项相似度:   {score_neg_ft:.4f}")
        diff_ft = score_pos_ft - score_neg_ft
        print(f"   👉 区分度 (正-负): {diff_ft:.4f} {'🌟 优秀' if diff_ft > diff_base else ''}")
        
        if diff_ft > diff_base:
            print("📈 结论: 微调后，干扰项被成功推远！")
        else:
            print("🤔 结论: 提升不明显，可能是该案例太简单。")

if __name__ == "__main__":
    compare_models()
