# 
import json
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "BAAI/bge-reranker-large" 
OUTPUT_DIR = "./output_rerank_st_final"
DATA_DIR = "./dataset_rerank"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

BATCH_SIZE = 2 
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

def load_dataset(file_path):
    """
    加载 jsonl 数据并转换为 SentenceTransformers 需要的 InputExample 格式
    Rerank 任务是二分类：(Query, Doc) -> 0 或 1
    """
    samples = []
    print(f"正在加载数据: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                query = data['query']
                
                # 1. 正例 (Label = 1.0)
                for pos_doc in data['pos']:
                    samples.append(InputExample(texts=[query, pos_doc], label=1.0))
                
                # 2. 负例 (Label = 0.0)
                # 限制负例数量，防止数据不平衡
                for neg_doc in data.get('neg', [])[:4]:
                    samples.append(InputExample(texts=[query, neg_doc], label=0.0))
                    
            except json.JSONDecodeError:
                continue
                
    print(f"加载完成，共 {len(samples)} 个样本对")
    return samples

def train_rerank():
    # 1. 初始化模型
    print(f"⬇加载基座模型: {MODEL_NAME} ...")
    # num_labels=1 会自动使用 BCEWithLogitsLoss (适合打分任务)
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_LENGTH)

    # 2. 准备数据
    train_samples = load_dataset(TRAIN_FILE)
    val_samples = load_dataset(VAL_FILE)

    if not train_samples:
        raise ValueError("训练集为空！")

    # SentenceTransformers 需要用 DataLoader 封装
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 3. 准备评估器
    # 既然有验证集，就用验证集；如果没有，从训练集切一点
    eval_data = val_samples if val_samples else train_samples[:50]
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_data, name='Rerank-Eval')

    # 计算预热步数
    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

    print(f"  开始使用 SentenceTransformers 训练...")
    print(f"   - 混合精度 (AMP): 关闭 (最稳模式)")
    print(f"   - Batch Size: {BATCH_SIZE}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        save_best_model=True,
        show_progress_bar=True,
        
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': LEARNING_RATE},
        
        use_amp=False 
    )
    
    # 5. 双重保险：强制保存一次
    print(f"正在保存最终模型到: {OUTPUT_DIR} ...")
    model.save(OUTPUT_DIR)
    print("训练完成！")

if __name__ == "__main__":
    train_rerank()