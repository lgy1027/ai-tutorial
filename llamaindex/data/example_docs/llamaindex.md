# LlamaIndex 学习笔记

LlamaIndex 是面向 LLM 应用的数据框架。它提供数据加载、索引、检索、查询引擎、聊天引擎、智能体、工作流、评估和观测等能力。

LlamaIndex 的核心包是 llama-index-core，具体模型、向量数据库、Reader、Tool 等能力通常放在独立 integration 包中。

在最小 RAG 示例中，SimpleDirectoryReader 读取文档，VectorStoreIndex 构建索引，as_query_engine 创建查询入口。

当系统变复杂时，可以拆开 QueryEngine，显式控制 Retriever、Node Postprocessor 和 Response Synthesizer。
