# 模型微调教程

本教程涵盖 Embedding 模型和 Rerank 模型的微调实战，帮助你在特定领域提升模型效果。

## 目录结构

```
fine-tuning/
├── embedding/          # Embedding 模型微调
│   └── sentenceTransformer/
│       ├── train_single_3090.py      # 单卡 3090 训练脚本
│       ├── train_advanced.py         # 高级训练配置
│       ├── generate_data_from_dir.py # 数据生成脚本
│       ├── eval_comparison.py        # 模型对比评估
│       └── finetune_data_mined.jsonl # 训练数据
└── rerank/             # Rerank 模型微调
    ├── train_rerank.py               # CrossEncoder 微调
    ├── train_rerank_torch.py         # PyTorch 原生实现
    ├── data_factory.py               # 数据工厂
    └── eval_rerank.py                # 模型评估
```

## Embedding 模型微调

### 基座模型
- `BAAI/bge-large-zh-v1.5`

### 训练脚本
| 文件 | 说明 |
|------|------|
| `train_single_3090.py` | 针对单卡 3090 优化，包含显存优化技巧 |
| `train_advanced.py` | 高级训练配置，支持分布式训练 |
| `generate_data_from_dir.py` | 从文档目录生成训练数据 |
| `eval_comparison.py` | 微调前后模型效果对比 |

### 显存优化技巧
- **梯度累积**: 小 Batch Size 累积，等效大 Batch
- **混合精度 (AMP)**: FP16 训练，减少显存占用
- **梯度检查点**: 用计算换显存
- **负例采样**: 控制负例数量 (3-5 个)

```python
# 关键配置
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 256
MAX_NEGS = 4
```

## Rerank 模型微调

### 基座模型
- `BAAI/bge-reranker-large`

### 训练脚本
| 文件 | 说明 |
|------|------|
| `train_rerank.py` | SentenceTransformers 方式微调 |
| `train_rerank_torch.py` | PyTorch 原生实现 |
| `data_factory.py` | 训练数据生成 |
| `eval_rerank.py` | 模型评估脚本 |

### 任务类型
Rerank 微调是二分类任务：
- 输入: `(Query, Document)` 对
- 输出: 相关性分数 (0-1)

```python
# 数据格式
samples.append(InputExample(texts=[query, pos_doc], label=1.0))  # 正例
samples.append(InputExample(texts=[query, neg_doc], label=0.0))  # 负例
```

## 快速开始

### Embedding 微调
```bash
cd embedding/sentenceTransformer
python train_single_3090.py
```

### Rerank 微调
```bash
cd rerank
python train_rerank.py
```

## 数据格式

训练数据为 JSONL 格式：
```json
{"query": "查询文本", "pos": ["正例文档1", "正例文档2"], "neg": ["负例文档1", "负例文档2"]}
```

## 依赖

- sentence-transformers
- torch
- transformers
