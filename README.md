# AI 教程

本仓库包含一系列关于智能体、工作流和本地知识库的教程；涉及技术包括不限于 LangChain、LangGraph、DeepAgents、A2A 协议、Embedding/Rerank 微调等。

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

**内容包括**：
1. LangGraph 入门 - StateGraph 基础与 Agent 构建
2. Stream 模式 - 流式输出与实时响应
3. 多智能体 - Agent 协作与任务分发
4. 人机交互 - Human-in-the-loop 中断机制
5. LangServe 部署 - 生产环境服务化

### DeepAgents 深度智能体

深度智能体框架实战，展示企业级 Agent 应用架构。

**核心特性**：
- 子智能体委派（Subagents）
- 混合后端存储（State + Store）
- Human-in-the-loop 中断策略
- 大文件自动拦截与转存

### A2A 协议（Agent-to-Agent）

Agent 间通信协议示例，实现多 Agent 协作。

**示例内容**：
- 基础 Client-Server 通信
- 高级 A2A 交互模式
- 天气查询 Agent 完整示例

### 模型微调实战

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

```bash
# 克隆仓库
git clone https://github.com/lgy1027/ai-tutorial.git

# 安装依赖
pip install langchain langchain-openai langgraph sentence-transformers torch

# 配置环境变量
cp langchain/.env\ example langchain/.env
# 编辑 .env 文件，填写 OPENAI_API_KEY 等配置

# 运行示例
python langchain/1、langchain-入门.py
```

## 目录结构

```
ai-tutorial/
├── langchain/          # LangChain 完整教程
├── langgraph/          # LangGraph 工作流教程
├── deepagents/         # DeepAgents 示例
├── a2a-test/           # A2A 协议示例
├── fine-tuning/        # 模型微调
│   ├── embedding/      # Embedding 微调
│   └── rerank/         # Rerank 微调
└── images/             # 图片资源
```

## 关注我们

文本内容可在公众号查看，欢迎关注我们的公众号和个人账号，获取更多 AI 相关资讯。

### 公众号

<img src="images/公众号.jpg" width="300" height="290">
