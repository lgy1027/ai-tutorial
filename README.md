# AI 教程

本仓库记录我在 AI 应用开发过程中的一些教程和示例，重点放在 RAG、Agent、工作流编排、本地知识库和模型微调这些方向。

这里不是只放“能跑的 Demo”。每个专题都尽量回答三个问题：

- 这个技术解决什么问题；
- 它在真实项目里应该放在哪一层；
- 示例代码跑通以后，下一步怎样往工程化方向走。

## 模块总览

| 模块 | 适合解决的问题 | 入口 |
| --- | --- | --- |
| LangChain | LLM 应用基础组件、Prompt、LCEL、RAG、Memory、Agent | [langchain/README.md](langchain/README.md) |
| LangGraph | 多步骤任务、状态图、流式输出、多智能体、人机交互 | [langgraph/README.md](langgraph/README.md) |
| LlamaIndex | RAG 数据接入、切块、索引、检索、评估和观测 | [llamaindex/README.md](llamaindex/README.md) |
| DeepAgents | 子智能体委派、混合存储、人机审批、深度任务执行 | [deepagents/README.md](deepagents/README.md) |
| A2A 协议 | Agent 之间的标准通信、服务发现和消息交互 | [a2a-test/README.md](a2a-test/README.md) |
| 模型微调 | Embedding / Rerank 模型在领域数据上的训练与评估 | [fine-tuning/README.md](fine-tuning/README.md) |

## 学习路线

如果你是按 AI 应用开发路径学习，可以按这个顺序看：

```text
LangChain 基础
  -> LangGraph 工作流
  -> LlamaIndex RAG 数据链路
  -> DeepAgents 深度智能体
  -> A2A 多 Agent 通信
  -> Embedding / Rerank 微调
```

如果你只关心知识库和 RAG，可以直接从 LangChain 的 RAG 章节和 LlamaIndex 系列开始。

如果你已经在做 Agent 项目，可以重点看 LangGraph、DeepAgents 和 A2A 协议。

## 教程

### LangChain 从入门到工程实践

完整的 LangChain 学习路径，涵盖 LLM 接口、Prompt 工程、LCEL 链、RAG 系统、Memory 记忆、Agents 智能体等核心内容。

*   [LangChain 教程目录](langchain/README.md)

**内容包括**：
1. LangChain 入门 - LLM 接口与 Prompt Templates
2. LCEL 深度解析 - 模块化链条构建
3. RAG 基础 - 数据加载、切分与向量化
4. RAG 核心 - 向量数据库与高级检索器
5. Memory 记忆 - 为 AI 应用赋予记忆
6. Agents - 让 AI 学会行动与思考
7. RAG 实战 - 构建生产级 RAG 应用
8. RAG 高级优化 - 提升检索质量、防范幻觉
9. Agent 最佳实践 - 调试、评估与部署
10. LangChain 1.0 工程实践 - 新特性与生产部署

### LangGraph 工作流编排

基于状态图的工作流编排框架，适用于构建复杂的多步骤 AI 应用。

LangChain 更像是一组 LLM 应用组件，LangGraph 更关注“流程怎么走、状态怎么保存、什么时候中断、什么时候继续”。如果你的 Agent 不再是一次简单调用，而是包含工具调用、循环、审批、多 Agent 协作，就应该看 LangGraph。

*   [LangGraph 教程目录](langgraph/README.md)

**内容包括**：
1. LangGraph 入门 - StateGraph 基础与 Agent 构建
2. Stream 模式 - 流式输出与实时响应
3. 多智能体 - Agent 协作与任务分发
4. 人机交互 - Human-in-the-loop 中断机制
5. LangServe 部署 - 生产环境服务化

**核心关注点**：
- `StateGraph`：用状态图描述任务流转。
- `Checkpointer`：保存会话状态，支持中断后恢复。
- `ToolNode`：把工具调用放到明确的节点里。
- Human-in-the-loop：在关键步骤前暂停，等待用户审批。

### LlamaIndex RAG 数据框架

面向 RAG 和知识库应用的数据框架教程，重点不是罗列 API，而是从数据进入系统开始，讲清楚文档切块、索引、检索、查询、Agent、Workflow、评估和观测之间的关系。主线之外，也补充了高级检索、GraphRAG、复杂文档解析、Agent / Workflow 进阶这些更贴近项目问题的专题。

*   [LlamaIndex 教程目录](llamaindex/README.md)

**内容包括**：
1. LlamaIndex 在 RAG 系统里的位置 - Document、Node、Index、Retriever、QueryEngine
2. 文档切块与 IngestionPipeline - chunk、metadata、cache 和不同切块策略
3. Index / Retriever / QueryEngine - 拆开检索链路，观察 source nodes
4. QueryEngine / ChatEngine / Agent - 单轮查询、多轮对话和工具调用边界
5. Workflows - 把多步骤 RAG 显式编排出来
6. 生产化 - 固定问题集、Response Evaluation、Retrieval Evaluation、Tracing
7. 进阶专题 - 高级检索、GraphRAG、复杂文档解析、Agent / Workflow 进阶

### DeepAgents 深度智能体

深度智能体框架实战，展示企业级 Agent 应用架构。

*   [DeepAgents 教程目录](deepagents/README.md)

**核心特性**：
- 子智能体委派（Subagents）
- 混合后端存储（State + Store）
- Human-in-the-loop 中断策略
- 大文件自动拦截与转存

**适合关注的场景**：
- 主 Agent 需要把任务委派给不同子 Agent。
- 临时状态和长期记忆需要分开管理。
- 写文件、执行任务等动作需要人工确认。
- Agent 可能生成较大内容，需要避免一次性写入上下文或内存。

### A2A 协议（Agent-to-Agent）

Agent 间通信协议示例，实现多 Agent 协作。

*   [A2A 教程目录](a2a-test/README.md)

**示例内容**：
- 基础 Client-Server 通信
- 高级 A2A 交互模式
- 天气查询 Agent 完整示例

**核心流程**：
1. 通过 Agent Card 发现远端 Agent。
2. 初始化 A2A Client。
3. 构造消息并发送给 Agent。
4. 接收响应，处理多轮或流式交互。

这个专题更偏 Agent 系统之间的通信边界，而不是单个 Agent 内部怎么推理。

### 模型微调实战

RAG 系统做一段时间后，经常会遇到一个问题：通用 Embedding 或 Rerank 模型在领域数据上效果不够稳定。这个目录主要放 Embedding 和 Rerank 的微调脚本、数据构造和评估方式。

*   [模型微调教程目录](fine-tuning/README.md)

#### Embedding 模型微调
- 单卡 3090 优化训练脚本
- 梯度累积、混合精度、梯度检查点等显存优化技巧
- 负例采样策略与数据生成

#### Rerank 模型微调
- CrossEncoder 微调实战
- 二分类任务训练
- 模型评估与对比

**基座模型**：
- Embedding: `BAAI/bge-large-zh-v1.5`
- Rerank: `BAAI/bge-reranker-large`

## 快速开始

不同专题的依赖不完全一样，建议进入对应目录后按该目录的 README 安装。下面只给一个最小入口。

### LangChain 示例

```bash
# 克隆仓库
git clone https://github.com/lgy1027/ai-tutorial.git
cd ai-tutorial

# 安装 LangChain / LangGraph 常用依赖
pip install langchain langchain-openai langgraph sentence-transformers torch

# 配置环境变量
cp langchain/.env\ example langchain/.env
# 编辑 .env 文件，填写 OPENAI_API_KEY 等配置

# 运行示例
python langchain/1、langchain-入门.py
```

### LlamaIndex 示例

```bash
cd llamaindex
pip install -r requirements.txt

# 复制 .env example 为 .env 后，填写 OpenAI 兼容接口配置
python code/01_architecture/01_minimal_rag.py
```

### LangGraph 示例

```bash
cd langgraph
python "1、langgraph（一）入门.py"
```

### A2A 示例

```bash
cd a2a-test

# 先启动服务端
python weather_agent/server.py

# 再运行客户端
python client.py
```

### 微调示例

```bash
cd fine-tuning/embedding/sentenceTransformer
python train_single_3090.py
```

## 目录结构

```
ai-tutorial/
├── langchain/          # LangChain 完整教程
├── langgraph/          # LangGraph 工作流教程
├── llamaindex/         # LlamaIndex RAG 数据框架教程
├── deepagents/         # DeepAgents 示例
├── a2a-test/           # A2A 协议示例
├── docs/               # 补充文档
├── fine-tuning/        # 模型微调
│   ├── embedding/      # Embedding 微调
│   └── rerank/         # Rerank 微调
└── images/             # 图片资源
```

## 关注我们

文本内容可在公众号查看，欢迎关注我们的公众号和个人账号，获取更多 AI 相关资讯。

### 公众号

<img src="images/公众号.jpg" width="300" height="290">
