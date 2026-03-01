import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# 1. 加载数据
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.2) # 80% 训练, 20% 验证

# 2. 加载模型和分词器
model_id = "answerdotai/ModernBERT-base" # 也可以选更小的版本
tokenizer = AutoTokenizer.from_pretrained(model_id)
num_labels = len(intents)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

# 3. 预处理函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 训练参数配置
training_args = TrainingArguments(
    output_dir="./intent_model_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True, # 开启半精度加速
    logging_steps=10
)

# 5. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./final_intent_model")