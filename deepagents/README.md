# DeepAgents 教程

DeepAgents 是一个深度智能体框架，支持子智能体委派、混合存储后端和人机协同等企业级特性。

## 示例文件

- `main.py` - 完整的 DeepAgents 示例

## 核心特性

### 子智能体委派
主智能体可以将复杂任务委派给专门的子智能体处理。

```python
research_subagent_config = {
    "name": "deep_researcher",
    "description": "专门用于执行复杂的互联网信息检索和分析任务。",
    "system_prompt": "你是一个严谨的研究员...",
    "tools": [internet_search],
    "model": model
}

agent = create_deep_agent(
    model=model,
    subagents=[research_subagent_config],
)
```

### 混合后端存储
支持 State 和 Store 两种存储后端，可根据数据类型自动路由。

```python
def hybrid_backend_factory(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),  # 临时文件
        routes={
            "/memories/": StoreBackend(runtime)  # 长期记忆
        }
    )
```

### Human-in-the-loop
支持在关键操作前暂停，等待人类审批。

```python
agent = create_deep_agent(
    model=model,
    interrupt_on={
        "task": {"allowed_decisions": ["approve", "reject"]},
        "write_file": {"allowed_decisions": ["approve", "reject", "edit"]}
    },
    checkpointer=checkpointer,
)
```

### 大文件拦截
自动检测大文件输出并转存，避免内存溢出。

## 快速开始

```bash
# 配置环境变量
export OPENAI_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"

# 运行示例
python main.py
```

## 依赖

- deepagents
- langchain-openai
- langgraph
- tavily-python
