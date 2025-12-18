import json
import os
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, "finetune_data_mined.jsonl")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_model_final")

# 选用 BGE-Large 中文版作为基座
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# BGE 模型专用的指令前缀 (必须加，否则效果减半)
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# --- 显存优化配置 (防 OOM 核心区) ---
# 批次大小：设为 1。多卡模式下，意味着每张卡一次只处理 1 条数据。
# 虽然是 1，但因为带了负例，实际计算量依然很大，这是最安全的设置。
BATCH_SIZE = 1 

# 最大长度：从 512 降为 256。
# 政务/业务文档通常 256 个 token (约 400 字) 足够覆盖核心语义。
# 这能直接节省 50% 以上的显存！
MAX_SEQ_LENGTH = 256 

# 最大负例数：限制每条数据只用前 2 个硬负例。
# 不要贪心用 7 个，先跑通流程最重要。
MAX_NEGS = 2 

# --- 训练参数 ---
NUM_EPOCHS = 3          # 训练 3 轮
LEARNING_RATE = 2e-5    # 经典微调学习率
DEV_RATIO = 0.1         # 划出 10% 的数据作为验证集(考试用)
# ==============================================================


def load_and_split_data(file_path, dev_ratio=0.1):
    """
    加载数据并随机划分为训练集和验证集
    """
    all_data = []
    print(f"📖 [Step 1] 正在加载原始数据: {file_path} ...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {file_path}，请检查 generate_data.py 是否运行成功！")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 简单校验数据完整性
                if item.get('query') and item.get('pos'):
                    all_data.append(item)
            except json.JSONDecodeError:
                pass
    
    print(f"📊 共加载 {len(all_data)} 条原始数据。")
    
    # 随机打乱并切分
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - dev_ratio))
    
    train_raw = all_data[:split_idx]
    dev_raw = all_data[split_idx:]
    
    print(f"✅ 数据切分完成: 训练集 {len(train_raw)} 条 | 验证集 {len(dev_raw)} 条")
    return train_raw, dev_raw

def convert_to_train_examples(raw_data):
    """
    将 JSON 数据转换为模型可读的 InputExample 对象
    在这里进行【负例截断】以节省显存
    """
    examples = []
    for data in raw_data:
        # 给 Query 加上指令前缀
        query = QUERY_INSTRUCTION + data['query']
        pos = data['pos'][0]
        neg_list = data.get('neg', [])
        
        # 【核心防爆显存逻辑】
        # 只取前 MAX_NEGS 个负例。
        # 最终输入 = [Query, Pos, Neg1, Neg2] (共 4 个句子)
        texts = [query, pos] + neg_list[:MAX_NEGS]
        
        examples.append(InputExample(texts=texts))
    return examples

def create_evaluator(raw_data):
    """
    构建评估器：模拟真实检索过程
    计算 MRR@10 (前10名里有没有正确答案)
    """
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for idx, data in enumerate(raw_data):
        query_id = f"q_{idx}"
        pos_doc_id = f"doc_pos_{idx}"
        
        query_text = QUERY_INSTRUCTION + data['query']
        pos_text = data['pos'][0]
        
        queries[query_id] = query_text
        corpus[pos_doc_id] = pos_text
        relevant_docs[query_id] = {pos_doc_id}
        
        # 把负例也加入到“文库”中，增加检索难度，测试模型分辨能力
        for neg_idx, neg_text in enumerate(data.get('neg', [])):
            neg_doc_id = f"doc_neg_{idx}_{neg_idx}"
            corpus[neg_doc_id] = neg_text
            
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name='dev_eval',
        mrr_at_k=[10],   # 关注前10名
        show_progress_bar=True
    )

def train_advanced():
    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到 GPU，使用 CPU 训练会极慢！")
    else:
        print(f"🚀 检测到 {torch.cuda.device_count()} 张显卡，准备起飞！")

    # 1. 加载基座模型
    print(f"⬇️ [Step 2] 正在加载基座模型: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 【显存优化 1】强制设置最大序列长度
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"🔧 已将最大序列长度限制为: {MAX_SEQ_LENGTH}")

    # 【显存优化 2】开启梯度检查点 (Gradient Checkpointing)
    # 这会牺牲一点点速度，但能节省 50%-70% 的显存，防止 OOM 的神器！
    model.gradient_checkpointing_enable()
    print("🔧 已开启梯度检查点模式 (Gradient Checkpointing)")
    
    # 2. 准备数据
    train_raw, dev_raw = load_and_split_data(DATA_FILE, DEV_RATIO)
    
    train_examples = convert_to_train_examples(train_raw)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 3. 准备评估器
    print("🕵️ [Step 3] 正在构建验证集评估器...")
    evaluator = create_evaluator(dev_raw)
    
    # 4. 定义损失函数
    # 使用多负例排序损失 (Contrastive Loss 的一种)
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)
    
    # 5. 开始训练
    print(f"🚀 [Step 4] 开始微调训练 (共 {NUM_EPOCHS} 轮)...")
    print(f"   - Batch Size: {BATCH_SIZE} (Per GPU)")
    print(f"   - 混合精度 (FP16): 开启")
    
    # 计算评估步数：保证每个 epoch 至少评估一次
    total_steps = len(train_dataloader) * NUM_EPOCHS
    eval_steps = max(1, int(len(train_dataloader) * 0.5)) # 每半个 epoch 评估一次
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            evaluation_steps=eval_steps,
            epochs=NUM_EPOCHS,
            warmup_steps=int(total_steps * 0.1), # 10% 步数热身
            optimizer_params={'lr': LEARNING_RATE},
            output_path=OUTPUT_DIR,
            save_best_model=True,     # 只有效果变好才保存
            show_progress_bar=True,
            
            # 【显存优化 3】开启混合精度训练 (FP16)
            # 显存再省一半！
            use_amp=True
        )
        print(f"\n🎉 恭喜！训练完成！")
        print(f"💾 最佳模型已保存至: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        print("💡 建议：如果还是 OOM，请检查是否还有其他进程占用显存 (使用 nvidia-smi 查看)")

if __name__ == "__main__":
    train_advanced()