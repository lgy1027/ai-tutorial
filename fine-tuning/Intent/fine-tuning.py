import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score

# --- 环境配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
MODEL_ID = "answerdotai/ModernBERT-base" 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "modernbert_multilabel_v4")

INTENTS = ["order_query", "addr_modify", "refund_apply", "oos"]
id2label = {i: label for i, label in enumerate(INTENTS)}
label2id = {label: i for i, label in enumerate(INTENTS)}

# --- 1. 数据加载 ---
dataset = load_dataset(
    "json", 
    data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl"),
        "test": os.path.join(DATA_DIR, "test.jsonl")
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 预处理函数
def preprocess_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=128)
    # 直接转为 float32 的 numpy 数组
    result["labels"] = [np.array(l, dtype=np.float32) for l in examples["labels"]]
    return result

tokenized_ds = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

# 显式设置数据集格式，强制指定 labels 为 float
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 自定义 Data Collator
class MultilabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 提取 labels 并确保它们是 float tensor
        labels = [f.pop("labels") for f in features]
        batch = super().__call__(features)
        # 重新放回 labels，并强制转为 float
        batch["labels"] = torch.stack(labels).float() 
        return batch

data_collator = MultilabelDataCollator(tokenizer=tokenizer)

# 指标与模型
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.Tensor(logits)).numpy()
    predictions = (probs > 0.5).astype(int)
    f1_samples = f1_score(labels, predictions, average="samples", zero_division=0)
    subset_acc = accuracy_score(labels, predictions)
    return {"f1_samples": f1_samples, "subset_accuracy": subset_acc}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(INTENTS),
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification"
)

# 训练配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_samples",
    save_total_limit=1,
    bf16=True, 
    logging_steps=10,
    warmup_steps=100, 
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🚀 开始多标签意图识别模型微调 (Float 强化版)...")
trainer.train()

# 错题集分析保持不变...
def analyze_error_cases():
    print("\n📊 正在生成多标签错题报告...")
    preds_output = trainer.predict(tokenized_ds["test"])
    logits = preds_output.predictions
    probs = torch.sigmoid(torch.Tensor(logits)).numpy()
    y_pred = (probs > 0.5).astype(int)
    y_true = preds_output.label_ids.astype(int)
    
    error_mask = ~np.all(y_pred == y_true, axis=1)
    error_indices = np.where(error_mask)[0]
    
    errors = []
    raw_test_texts = dataset["test"]["text"]
    for idx in error_indices:
        true_names = [id2label[i] for i, val in enumerate(y_true[idx]) if val == 1]
        pred_names = [id2label[i] for i, val in enumerate(y_pred[idx]) if val == 1]
        errors.append({
            "text": raw_test_texts[idx],
            "true_labels": "|".join(true_names),
            "pred_labels": "|".join(pred_names) if pred_names else "NONE",
            "scores": [f"{id2label[i]}:{probs[idx][i]:.4f}" for i in range(len(INTENTS))]
        })
    pd.DataFrame(errors).to_csv(f"{OUTPUT_DIR}/multilabel_errors.csv", index=False, encoding="utf-8-sig")

analyze_error_cases()
trainer.save_model(f"{OUTPUT_DIR}/final_model")