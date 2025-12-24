# pytorch微调版本
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "BAAI/bge-reranker-large" 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_rerank_final")

# 训练超参数
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8 # 等效 Batch=8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512

# 数据路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(CURRENT_DIR, "dataset_rerank/train.jsonl")

class RerankDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"正在加载数据: {file_path} ...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    query = item['query']
                    # 展开正例
                    for pos in item['pos']:
                        self.data.append((query, pos, 1.0)) # label 1
                    # 展开负例 (取前4个)
                    for neg in item.get('neg', [])[:4]:
                        self.data.append((query, neg, 0.0)) # label 0
                except: continue
        print(f"加载完成，共 {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, doc, label = self.data[idx]
        # 手动 Tokenize: [CLS] query [SEP] doc [SEP]
        features = self.tokenizer(
            query, 
            doc, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': features['input_ids'].squeeze(0),
            'attention_mask': features['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def train():
    # 1. 初始化模型和分词器
    print(f"⬇加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # num_labels=1 表示做回归打分
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 2. 准备数据
    dataset = RerankDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(dataloader) // GRADIENT_ACCUMULATION
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    
    # 4. 损失函数 (BCEWithLogitsLoss 自带 Sigmoid，数值更稳定)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("开始PyTorch 训练...")
    global_step = 0
    total_loss = 0
    
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(range(len(dataloader)), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(dataloader):
            # 搬运数据到 GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1) # [batch]
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 梯度累积
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                # 更新权重
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防爆炸
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f"{total_loss * GRADIENT_ACCUMULATION / (step+1):.4f}"})

            progress_bar.update(1)
        progress_bar.close()
        
        # 每个 Epoch 强制保存一次
        print(f"Saving model checkpoint for epoch {epoch+1}...")
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # 5. 最终保存
    print(f"训练结束，保存最终模型到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()