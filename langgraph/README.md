# LangGraph 教程

本教程帮助你掌握 LangGraph 工作流编排框架，构建复杂的多步骤 AI 应用。

## 教程内容

1. **LangGraph 入门**
   - `1、langgraph（一）入门.py`
   - StateGraph 状态图基础
   - 节点与边的定义
   - ReAct Agent 构建

2. **Stream 模式**
   - `2、langgraph（二）stream模式.py`
   - 流式输出实现
   - 实时响应处理
   - 多种流模式对比

3. **多智能体**
   - `3、langgraph（三）多智能体.py`
   - Agent 协作模式
   - 任务分发与结果汇总
   - 多 Agent 通信机制

4. **人机交互**
   - `4、langgraph（四）人机交互.py`
   - Human-in-the-loop 实现
   - 中断与恢复机制
   - 用户审批流程

5. **LangServe 服务部署**
   - `5、langgraph（五）LangServe服务部署.py`
   - FastAPI 集成
   - 生产环境部署
   - API 接口设计

## 核心概念

### StateGraph
状态图是 LangGraph 的核心，用于定义工作流的节点和边。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.set_entry_point("agent")
workflow.add_edge("tool", "agent")
```

### Checkpointer
用于持久化会话状态，支持断点续传。

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

### ToolNode
预构建的工具执行节点，简化工具调用流程。

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=[search_tool])
```

## 快速开始

```bash
# 配置环境变量
cp ../langchain/.env .env

# 运行示例
python "1、langgraph（一）入门.py"
```

## 依赖

- langgraph
- langchain-openai
- langchain-tavily
- python-dotenv
