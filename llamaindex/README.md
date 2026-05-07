# LlamaIndex 教程

![LlamaIndex 教程路线](images/00-series-roadmap.svg)

这个系列不是把 LlamaIndex API 逐个过一遍，而是按一个 RAG 系统从 Demo 到可维护应用的顺序来写：

```text
对象边界 -> 数据处理 -> 检索查询 -> 交互入口 -> 流程编排 -> 评估观测
```

每一篇都尽量回答两个问题：这一层解决什么问题，以及它和前后几层怎么衔接。

## 目录结构

```text
llamaindex/
├── README.md
├── .env example
├── data/
│   └── example_docs/          # 本地示例文档
├── docs/                      # 6 期 Markdown 教程
├── images/                    # 教程配图
└── code/                      # 配套代码与每期示例
```

## 环境准备

```bash
pip install -r requirements.txt
```

如果需要使用 OpenAI 兼容接口，可以复制 `.env example` 为 `.env`，并填写：

```bash
OPENAI_API_KEY=""
OPENAI_BASE_URL=""
OPENAI_MODEL=""
LLAMAINDEX_CONTEXT_WINDOW="32768"
EMBEDDING_MODEL=""
```

## 教程列表

1. [先看懂 LlamaIndex 的架构边界](docs/01-architecture-boundary.md)
2. [数据进入系统之前，RAG 已经决定了一半效果](docs/02-ingestion-pipeline.md)
3. [VectorStoreIndex 不是终点，Retriever 才是 RAG 的控制面](docs/03-index-retriever-query-engine.md)
4. [Query Engine、Chat Engine、Agent 的边界](docs/04-query-chat-agent.md)
5. [Workflows：把多步骤 RAG 显式编排出来](docs/05-workflows.md)
6. [生产化：评估、观测与可替换架构](docs/06-production-evaluation-observability.md)

## 运行示例

```bash
cd llamaindex

# 第 1 期：架构边界
python code/01_architecture/01_minimal_rag.py
python code/01_architecture/02_document_to_node.py
python code/01_architecture/03_settings_and_source_debug.py

# 第 2 期：IngestionPipeline
python code/02_ingestion/01_basic_splitter.py
python code/02_ingestion/02_metadata_ingestion.py
python code/02_ingestion/03_ingestion_cache.py
python code/02_ingestion/04_chunk_size_compare.py
python code/02_ingestion/05_splitter_strategy_compare.py

# 第 3 期：Index / Retriever / QueryEngine
python code/03_querying/01_default_query_engine.py
python code/03_querying/02_explicit_retriever.py
python code/03_querying/03_custom_query_engine.py
python code/03_querying/04_node_postprocessor.py
python code/03_querying/05_persist_and_reload.py

# 第 4 期：QueryEngine / ChatEngine / Agent
python code/04_engines_agents/01_query_engine.py
python code/04_engines_agents/02_chat_engine.py
python code/04_engines_agents/03_query_engine_tool.py
python code/04_engines_agents/04_function_agent.py

# 第 5 期：Workflows
python code/05_workflows/01_basic_workflow.py
python code/05_workflows/02_multi_step_rag_workflow.py
python code/05_workflows/03_branching_workflow.py

# 第 6 期：评估和观测
python code/06_production/01_fixed_question_set.py
python code/06_production/02_response_evaluation.py
python code/06_production/03_retrieval_evaluation.py
python code/06_production/04_trace_callback.py
```

这些代码优先保持清晰，不追求一次覆盖所有高级参数。每一期文章会先解释抽象边界，再给出最小可运行代码，最后补充工程化注意点。
