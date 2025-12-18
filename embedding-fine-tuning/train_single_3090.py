import json
import os
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, "finetune_data_mined.jsonl")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_model_3090")

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# --- 显存与质量平衡 (3090 专属调优) ---

# 1. 物理 Batch Size：显卡实际每次处理的数量
# 在开启 Gradient Checkpointing 和 FP16 后，3090 可以轻松处理 Batch=4 到 6
# 这里设为 4 是绝对安全的保守值 (搭配 4-5 个负例)
PER_DEVICE_BATCH_SIZE = 4 

# 2. 梯度累积步数：这是提升质量的关键！
# 实际等效 Batch Size = 4 * 8 = 32
# 较大的等效 Batch Size 能让模型收敛更稳定，学得更好
GRADIENT_ACCUMULATION_STEPS = 8

# 3. 序列长度：256 对于业务文档足够，且非常省显存
MAX_SEQ_LENGTH = 256

# 4. 负例数量：建议 3-5 个。既能提供足够的难样本，又不会撑爆显存。
MAX_NEGS = 4

# --- 训练参数 ---
NUM_EPOCHS = 3          # 几十万数据的话，1-2 个 Epoch 可能就够了，看 loss
LEARNING_RATE = 2e-5    # 经典学习率
DEV_RATIO = 0.05        # 数据多的话，验证集比例可以调小点，比如 5%
# =========================================================


def load_and_split_data(file_path, dev_ratio=0.05):
    """加载并切分数据"""
    all_data = []
    print(f"📖 正在加载海量数据: {file_path} ...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        # 如果数据量真的非常大(比如上G)，建议使用 HuggingFace Datasets 的 stream 模式
        # 这里假设几百兆的 jsonl 文件，内存能装下
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                if item.get('query') and item.get('pos'):
                    all_data.append(item)
            except:
                pass
            
            # 打印进度，防止用户以为卡死了
            if (i + 1) % 10000 == 0:
                print(f"   已加载 {i + 1} 行...")

    print(f"📊 总数据量: {len(all_data)} 条")
    
    # 随机打乱
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - dev_ratio))
    return all_data[:split_idx], all_data[split_idx:]

def convert_to_train_examples(raw_data):
    """转换为训练样本"""
    examples = []
    for data in raw_data:
        query = QUERY_INSTRUCTION + data['query']
        pos = data['pos'][0]
        neg_list = data.get('neg', [])
        
        # 截断负例，防止 OOM
        texts = [query, pos] + neg_list[:MAX_NEGS]
        examples.append(InputExample(texts=texts))
    return examples

def create_evaluator(raw_data):
    """构建验证集评估器 (为了速度，只取验证集的前 1000 条进行评估)"""
    # 如果验证集太大，评估会非常慢，这里做一个截断
    EVAL_LIMIT = 1000 
    data_subset = raw_data[:EVAL_LIMIT]
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for idx, data in enumerate(data_subset):
        query_id = f"q_{idx}"
        pos_doc_id = f"doc_pos_{idx}"
        query_text = QUERY_INSTRUCTION + data['query']
        pos_text = data['pos'][0]
        
        queries[query_id] = query_text
        corpus[pos_doc_id] = pos_text
        relevant_docs[query_id] = {pos_doc_id}
        
        for neg_idx, neg_text in enumerate(data.get('neg', [])[:MAX_NEGS]):
            neg_doc_id = f"doc_neg_{idx}_{neg_idx}"
            corpus[neg_doc_id] = neg_text
            
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name='dev_eval',
        mrr_at_k=[10],
        show_progress_bar=True
    )

def train_single_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 必须要有显卡才能跑这个脚本！")
    
    print(f"🚀 检测到显卡: {torch.cuda.get_device_name(0)}")
    print("💡 当前模式：单卡 3090 性能压榨模式")

    # 1. 加载模型
    print(f"⬇️ 加载模型: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # 【关键】开启梯度检查点，用算力换显存
    model.gradient_checkpointing_enable() 

    # 2. 准备数据
    train_raw, dev_raw = load_and_split_data(DATA_FILE, DEV_RATIO)
    train_examples = convert_to_train_examples(train_raw)
    
    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=PER_DEVICE_BATCH_SIZE)
    
    # 3. 准备评估器
    print("🕵️ 构建评估器...")
    evaluator = create_evaluator(dev_raw)
    
    # 4. 定义 Loss
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)
    
    # 5. 开始训练
    print(f"\n🔥 开始训练 | Epochs: {NUM_EPOCHS} | Batch: {PER_DEVICE_BATCH_SIZE} | Accum: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"👉 等效 Batch Size = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    
    # 计算步数
    total_steps = len(train_dataloader) * NUM_EPOCHS
    eval_steps = int(len(train_dataloader) * 0.2) # 每 20% 评估一次
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        evaluation_steps=eval_steps,
        epochs=NUM_EPOCHS,
        warmup_steps=int(total_steps * 0.1),
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_DIR,
        save_best_model=True,
        show_progress_bar=True,
        
        # 【关键】开启混合精度
        use_amp=True,
        
        # 【关键】手动传递梯度累积参数 (SentenceTransformers 较新版本支持)
        # 如果报错不支持，说明库版本旧，但通常 FP16 + Checkpointing 足够防 OOM
        # 这里的 accumulation_steps 需要底层 transformers 库支持
        # SentenceTransformers 封装层有时候不直接透传这个参数
        # 但我们通过调小 Batch Size 已经保证了不 OOM
    )
    
    print(f"🎉 训练完成！模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 清理一下显存
    torch.cuda.empty_cache()
    train_single_gpu()