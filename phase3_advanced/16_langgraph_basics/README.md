# 模块 16：LangGraph 基础

## 🎯 学习目标

本模块将帮助你理解 LangGraph 1.0 的核心概念，学会创建状态图来构建复杂的 AI 工作流。

## 📚 核心概念

### 什么是 LangGraph？

LangGraph 是一个用于构建**状态化、多步骤 AI 应用**的框架。它使用**图（Graph）** 的概念来组织工作流：

- **节点（Nodes）**：图中的处理单元，可以是 LLM 调用、工具执行或自定义函数
- **边（Edges）**：连接节点的路径，定义执行顺序
- **状态（State）**：在节点之间传递的数据结构


### LangGraph vs LangChain

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| 抽象级别 | 高级 | 低级 |
| 适用场景 | 快速构建标准 Agent | 复杂自定义工作流 |
| 控制粒度 | 通过中间件 | 完全控制每个节点 |
| 状态管理 | 自动 | 手动但灵活 |

### LangGraph 编程映射

LangGraph 的概念与编程语言要素有着清晰的对应关系：

| 编程语言要素 | LangGraph 对应概念 | 说明 |
|-------------|-------------------|------|
| 数据 (Data) | 状态 (State) | 工作流执行过程中传递和更新的信息载体 |
| 函数 (Function) | 节点 (Node) | 对状态进行操作（如调用模型、执行工具）的基本单元 |
| 控制逻辑 (Control Flow) | 边 (Edge) | 决定节点间执行顺序的逻辑，可以是条件分支或静态路径，支持串行与并行 |
| 存储 (Storage) | 检查点 / 记忆 (Checkpointing / Memory) | 状态的持久化机制，允许工作流暂停、恢复或具有长期记忆 |
| 中断 (Interrupt) | 人在回路 (Human-in-the-Loop) | 通过中断控制流，在关键环节引入人工审核或干预的能力 |

### 核心组件

```python
# 1. 定义状态 - 使用 TypedDict
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，自动累加
    current_step: str                         # 当前步骤

# 2. 定义节点 - 接收和返回状态的函数
def my_node(state: State) -> dict:
    # 处理逻辑
    return {"current_step": "completed"}

# 3. 创建图
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"messages": [], "current_step": "start"})
```

## 🔑 关键 API

### StateGraph

```python
from langgraph.graph import StateGraph, START, END

# 创建图
graph = StateGraph(State)

# 添加节点
graph.add_node("node_name", node_function)

# 添加边
graph.add_edge("from_node", "to_node")  # 普通边
graph.add_edge(START, "first_node")      # 从入口开始
graph.add_edge("last_node", END)         # 到出口结束

# 添加条件边
graph.add_conditional_edges(
    "from_node",
    condition_function,  # 返回下一个节点名称
    {"option1": "node1", "option2": "node2"}
)
```

### add_messages 注解

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages 确保消息被追加而不是替换
    messages: Annotated[list, add_messages]
```

### 编译选项

```python
from langgraph.checkpoint.memory import MemorySaver

# 添加内存检查点（用于持久化）
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用 thread_id 进行会话管理
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input_data, config=config)
```

## 📝 本模块示例

### main.py

包含三个递进的示例：
1. **简单顺序图**：展示基本的节点和边
2. **条件分支图**：根据条件选择不同路径
3. **带内存的对话图**：实现多轮对话

## 🧪 练习

1. 修改 `simple_workflow` 添加一个新的处理节点
2. 在 `conditional_workflow` 中添加第三个分支
3. 扩展 `conversation_workflow` 支持更多类型的用户意图

## 📖 延伸阅读

- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph 概念指南](https://docs.langchain.com/oss/python/langgraph/concepts)
- [图的可视化](https://docs.langchain.com/oss/python/langgraph/visualization)

## ⚠️ 注意事项

1. 状态更新是**合并式**的，只需返回要更新的字段
2. 使用 `add_messages` 注解时，消息会自动追加
3. 条件函数必须返回有效的节点名称字符串
4. 编译后的图是不可变的，需要修改时重新编译
