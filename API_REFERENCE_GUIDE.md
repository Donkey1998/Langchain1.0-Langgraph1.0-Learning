# LangChain 1.0 & LangGraph 1.0 API 参考指南

> 本文档汇总了项目中使用的所有核心 API，便于学习和开发时快速查阅。

---

## 📚 目录

- [一、核心模型 API](#一核心模型-api)
- [二、提示词模板 API](#二提示词模板-api)
- [三、消息类型 API](#三消息类型-api)
- [四、工具开发 API](#四工具开发-api)
- [五、Agent API](#五agent-api)
- [六、结构化输出 API](#六结构化输出-api)
- [七、验证和重试 API](#七验证和重试-api)
- [八、输出解析器 API](#八输出解析器-api)
- [九、Runnable 接口详解](#九runnable-接口详解)
- [十、RAG 相关 API](#十rag-相关-api)
- [十一、LangGraph 核心 API](#十一langgraph-核心-api)
- [十二、向量数据库 API](#十二向量数据库-api)
- [十三、文档处理 API](#十三文档处理-api)
- [十四、状态持久化 API](#十四状态持久化-api)
- [十五、外部服务集成](#十五外部服务集成)
- [十六、多模态 API](#十六多模态-api)
- [十七、MCP 集成](#十七mcp-model-context-protocol-集成)
- [十八、错误处理模式](#十八错误处理模式)
- [十九、图可视化 API](#十九图可视化-api)
- [二十、中间件系统 API](#二十中间件系统-api)
- [二十一、回调系统 API](#二十一回调系统-api)
- [二十二、消息管理 API](#二十二消息管理-api)
- [二十三、工具高级 API](#二十三工具高级-api)
- [二十四、配置管理 API](#二十四配置管理-api)
- [二十五、状态管理 API](#二十五状态管理-api)
- [二十六、异步 API](#二十六异步-api)

---

## 一、核心模型 API

### 1.1 模型初始化 - `init_chat_model`

**用途**: 统一的模型初始化接口

**导入路径**: `from langchain.chat_models import init_chat_model`

#### 基本语法

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "provider:model_name",  # 提供商:模型名称
    api_key="your-api-key",  # API 密钥（可选）
    temperature=0.7,         # 温度参数（0.0-2.0）
    max_tokens=1000,         # 最大 token 数
)
```

#### 参数详解

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `str` | ✅ | 无 | 格式为 `"provider:model_name"` |
| `api_key` | `str` | ❌ | 从环境变量读取 | API 密钥 |
| `temperature` | `float` | ❌ | `1.0` | 控制输出随机性（0.0-2.0） |
| `max_tokens` | `int` | ❌ | 模型默认值 | 限制输出 token 数 |

#### 支持的提供商格式

```python
# Groq（推荐，免费且快速）
"groq:llama-3.3-70b-versatile"
"groq:mixtral-8x7b-32768"
"groq:gemma2-9b-it"

# OpenAI
"openai:gpt-4o-mini"
"openai:gpt-4"
"openai:gpt-3.5-turbo"

# Anthropic Claude
"anthropic:claude-sonnet-4-5-20250929"
"anthropic:claude-3-haiku-20240307"

# Google Gemini
"gemini:gemini-pro"

# HuggingFace（需要适配器）
"huggingface:MODEL_NAME"
```

#### 使用示例

```python
import os
from dotenv import load_dotenv
load_dotenv()

# 方式 1: 从环境变量读取 API key
model = init_chat_model("groq:llama-3.3-70b-versatile")

# 方式 2: 显式传递 API key
model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 方式 3: 配置温度和 token 限制
model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    temperature=0.0,    # 最确定性输出
    max_tokens=500      # 限制输出长度
)
```

---

### 1.2 模型调用 - `invoke`

**用途**: 同步调用 LLM 模型

#### 基本语法

```python
response = model.invoke(input, config=None)
```

#### 三种输入格式

##### 格式 1: 纯字符串（最简单）

```python
response = model.invoke("什么是机器学习？用一句话解释")
print(response.content)
```

**适用场景**: 简单的一次性问答

##### 格式 2: 字典列表（推荐，最灵活）

```python
messages = [
    {"role": "system", "content": "你是一个专业的 Python 编程导师"},
    {"role": "user", "content": "什么是装饰器？"}
]

response = model.invoke(messages)
print(response.content)
```

**适用场景**: 需要设置系统提示、多轮对话

##### 格式 3: 消息对象列表（类型安全）

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个 Python 专家"),
    HumanMessage(content="什么是生成器？")
]

response = model.invoke(messages)
```

**适用场景**: 大型项目，需要类型检查

#### 返回值结构

```python
response = model.invoke("Hello")

# 1. 主要内容
response.content              # str - AI 的回复文本
response.response_metadata    # dict - 响应元数据
response.id                   # str - 消息唯一 ID
response.usage_metadata       # dict - Token 使用情况

# 2. 访问 Token 使用情况
usage = response.response_metadata.get('token_usage', {})
print(f"提示 tokens: {usage.get('prompt_tokens')}")
print(f"完成 tokens: {usage.get('completion_tokens')}")
print(f"总计 tokens: {usage.get('total_tokens')}")
```

#### 常见错误处理

```python
try:
    response = model.invoke("Hello")
    print(response.content)
except ValueError as e:
    print(f"配置错误: {e}")
except ConnectionError as e:
    print(f"网络错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

---

### 1.3 流式输出 - `stream`

**用途**: 实时流式输出响应

```python
# 流式输出（逐字返回）
for chunk in model.stream("写一首关于编程的诗"):
    print(chunk.content, end="", flush=True)
```

---

## 二、提示词模板 API

### 2.1 PromptTemplate - 简单文本模板

**导入路径**: `from langchain_core.prompts import PromptTemplate`

#### 创建模板

```python
from langchain_core.prompts import PromptTemplate

# 方法 1: from_template（推荐）
template = PromptTemplate.from_template(
    "将以下文本翻译成{language}：\n{text}"
)

# 方法 2: 显式指定变量
template = PromptTemplate(
    input_variables=["product", "feature"],
    template="为{product}写一句广告语，重点突出{feature}特点。"
)
```

#### 使用模板

```python
# 方式 1: format() - 返回字符串
prompt_str = template.format(language="中文", text="Hello")
print(prompt_str)  # "将以下文本翻译成中文：\nHello"

# 方式 2: invoke() - 返回 PromptValue
prompt_value = template.invoke({"language": "中文", "text": "Hello"})
print(prompt_value.text)
```

#### 部分变量预填充

```python
template = PromptTemplate.from_template("你是一个{role}，请{task}")

# 预填充 role
partial_template = template.partial(role="Python 导师")

# 现在只需要提供 task
prompt = partial_template.format(task="解释装饰器")
```

---

### 2.2 ChatPromptTemplate - 聊天消息模板

**导入路径**: `from langchain_core.prompts import ChatPromptTemplate`

#### 创建模板

```python
from langchain_core.prompts import ChatPromptTemplate

# 使用元组格式（推荐）
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", "{question}")
])
```

#### 消息角色类型

| 角色字符串 | 含义 | 用途 |
|-----------|------|------|
| `"system"` | 系统消息 | 设定 AI 的行为、角色、规则 |
| `"user"` / `"human"` | 用户消息 | 用户的输入/问题 |
| `"assistant"` / `"ai"` | AI 消息 | AI 的回复（用于对话历史） |

#### 使用模板

```python
# 返回消息列表
messages = template.format_messages(
    role="Python 导师",
    question="什么是装饰器？"
)

# 直接传递给模型
response = model.invoke(messages)
```

#### 高级用法

```python
# 部分变量
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，目标用户是{audience}"),
    ("user", "{task}")
])

customer_support_template = template.partial(
    role="客服专员",
    audience="普通用户"
)

# 现在只需要提供 task
messages = customer_support_template.format_messages(task="解释退款政策")
```

---

### 2.3 LCEL 链式调用

**用途**: 使用管道运算符 `|` 连接组件

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# 创建组件
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("user", "{input}")
])

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 使用 | 创建链
chain = template | model

# 直接调用链
response = chain.invoke({
    "role": "Python 导师",
    "input": "什么是装饰器？"
})

print(response.content)
```

---

## 三、消息类型 API

### 3.1 消息类型概览

**导入路径**: `from langchain_core.messages import ...`

| 消息类型 | 类名 | 对应字典格式 | 用途 |
|---------|------|-------------|------|
| 系统消息 | `SystemMessage` | `{"role": "system"}` | 设定 AI 的行为、角色、规则 |
| 用户消息 | `HumanMessage` | `{"role": "user"}` | 用户的输入 |
| AI 消息 | `AIMessage` | `{"role": "assistant"}` | AI 的回复 |
| 工具消息 | `ToolMessage` | `{"role": "tool"}` | 工具执行结果 |

### 3.2 使用消息对象

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 创建消息
system_msg = SystemMessage(content="你是一个友好的助手")
human_msg = HumanMessage(content="你好")
ai_msg = AIMessage(content="你好！我能帮你什么？")

# 构建对话历史
messages = [system_msg, human_msg, ai_msg]
messages.append(HumanMessage(content="今天天气怎么样？"))

# 调用模型
response = model.invoke(messages)
```

### 3.3 对话历史管理

```python
# 核心原则：每次调用必须传入完整历史
conversation = [
    {"role": "system", "content": "你是一个友好的助手"}
]

# 第一轮
conversation.append({"role": "user", "content": "我叫小明"})
r1 = model.invoke(conversation)

# 保存 AI 回复到历史
conversation.append({"role": "assistant", "content": r1.content})

# 第二轮（基于上下文）
conversation.append({"role": "user", "content": "我刚才说我叫什么？"})
r2 = model.invoke(conversation)
print(r2.content)  # "你说你叫小明。"
```

---

## 四、工具开发 API

### 4.1 @tool 装饰器

**导入路径**: `from langchain_core.tools import tool`

#### 基本用法

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """
    清晰的工具描述（AI 读这个！）

    参数:
        param: 参数说明

    返回:
        返回值说明
    """
    return "结果字符串"
```

#### 最佳实践

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def search_weather(city: str, unit: Optional[str] = "celsius") -> str:
    """
    查询指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"
        unit: 温度单位，"celsius"（摄氏度）或 "fahrenheit"（华氏度），默认为摄氏度

    返回:
        天气信息字符串，包括温度、天气状况等
    """
    # 实现天气查询逻辑
    return f"{city}今天天气晴朗，温度25℃"
```

#### 参数类型注解

```python
from typing import List, Optional
from pydantic import BaseModel, Field

# 使用 Pydantic 模型
class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")

@tool(args_schema=CalculatorInput)
def calculator(input: CalculatorInput) -> str:
    """计算两个数字的和"""
    return f"{input.a} + {input.b} = {input.a + input.b}"

# 或者直接类型注解
@tool
def multiply(x: int, y: int) -> int:
    """计算两个数字的乘积"""
    return x * y
```

---

## 五、Agent API

### 5.1 create_agent - 创建 Agent

**导入路径**: `from langchain.agents import create_agent`

#### 基本语法

```python
from langchain.agents import create_agent

agent = create_agent(
    model=init_chat_model("groq:llama-3.3-70b-versatile"),
    tools=[tool1, tool2],
    system_prompt="Agent 的行为指令"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})

final_answer = response['messages'][-1].content
```

#### 参数详解

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | 模型实例 | ✅ | 使用 `init_chat_model` 创建的模型 |
| `tools` | `list[tool]` | ✅ | 工具列表 |
| `system_prompt` | `str` | ❌ | 系统提示，定义 Agent 行为 |
| `checkpoint` | `CheckpointSaver` | ❌ | 状态持久化 |

#### 完整示例

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式

    ⚠️ 安全警告: eval() 在生产环境中有代码注入风险，仅用于演示。
    生产环境应使用 numexpr 或限制输入格式。
    """
    try:
        # 仅用于演示目的，生产环境需要更安全的实现
        result = eval(expression)
        return f"结果: {result}"
    except:
        return "计算失败"

# 创建 Agent
agent = create_agent(
    model=init_chat_model("groq:llama-3.3-70b-versatile"),
    tools=[calculator],
    system_prompt="""你是一个数学助手。
使用 calculator 工具计算数学表达式。
"""
)

# 调用 Agent
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "计算 123 * 456"}
    ]
})

print(response['messages'][-1].content)
```

### 5.2 Agent 执行循环

```python
# 查看完整历史
for msg in response['messages']:
    print(f"{msg.type}: {msg.content}")

# 获取最终答案
final = response['messages'][-1].content

# 流式输出
for chunk in agent.stream(input):
    print(chunk)
```

---

## 六、结构化输出 API

### 6.1 with_structured_output

**用途**: 将 LLM 输出转为结构化 Pydantic 对象

#### 基本用法

```python
from pydantic import BaseModel, Field

# 定义数据模型
class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

# 创建结构化输出 LLM
structured_llm = model.with_structured_output(Person)

# 调用
result = structured_llm.invoke("张三是一名 30 岁的软件工程师")

# result 是 Person 实例
print(result.name)       # "张三"
print(result.age)        # 30
print(result.occupation) # "软件工程师"
```

#### 高级特性

```python
from typing import Optional, List
from enum import Enum

# 枚举类型
class Priority(str, Enum):
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"

# 嵌套模型
class Address(BaseModel):
    city: str
    district: str

class Company(BaseModel):
    name: str
    address: Address

# 列表提取
class Person(BaseModel):
    name: str
    age: int

class PeopleList(BaseModel):
    people: List[Person]

structured_llm = model.with_structured_output(PeopleList)
result = structured_llm.invoke("张三 30岁，李四 25岁")
```

---

## 七、验证和重试 API

### 7.1 with_retry - 重试机制

**用途**: 当 LLM 调用失败时自动重试

**导入路径**: 内置方法，无需额外导入

#### 基本语法

```python
# 带重试的 LLM
llm_with_retry = model.with_retry(
    retry_if_exception_type=(ConnectionError, TimeoutError),
    stop_after_attempt=3
)

# 调用时会自动重试
response = llm_with_retry.invoke("你的问题")
```

#### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retry_if_exception_type` | `tuple[type]` | `(Exception,)` | 触发重试的异常类型 |
| `stop_after_attempt` | `int` | `3` | 最大重试次数 |
| `wait_exponential_jitter` | `bool` | `True` | 是否使用指数退避+抖动 |

#### 完整示例

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 配置重试
robust_model = model.with_retry(
    retry_if_exception_type=(
        ConnectionError,   # 网络错误
        TimeoutError,      # 超时错误
        APIConnectionError # API 连接错误
    ),
    stop_after_attempt=3,  # 最多重试 3 次
    wait_exponential_jitter=True  # 指数退避
)

# 使用
try:
    response = robust_model.invoke("你好")
except Exception as e:
    print(f"重试 3 次后仍然失败: {e}")
```

---

### 7.2 with_fallbacks - 降级机制

**用途**: 当主模型失败时，自动切换到备用模型

```python
from langchain.chat_models import init_chat_model

# 主模型和备用模型
primary_model = init_chat_model("openai:gpt-4o-mini")
fallback_model = init_chat_model("groq:llama-3.3-70b-versatile")

# 配置降级
robust_llm = primary_model.with_fallbacks([fallback_model])

# 如果 primary_model 失败，自动使用 fallback_model
response = robust_llm.invoke("你的问题")
```

#### 多级降级

```python
# 多个备用模型（按优先级）
model_1 = init_chat_model("openai:gpt-4o-mini")
model_2 = init_chat_model("groq:llama-3.3-70b-versatile")
model_3 = init_chat_model("anthropic:claude-3-haiku-20240307")

robust_llm = model_1.with_fallbacks([model_2, model_3])

# 依次尝试：model_1 -> model_2 -> model_3
response = robust_llm.invoke("问题")
```

---

### 7.3 组合使用：重试 + 降级

**用途**: 构建高可用的 LLM 调用链

```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

# 定义结构化输出模型
class Output(BaseModel):
    result: str

# 主模型和备用模型
primary = init_chat_model("openai:gpt-4o-mini")
fallback = init_chat_model("groq:llama-3.3-70b-versatile")

# 最佳实践顺序：结构化输出 -> 重试 -> 降级
# 1. 先配置结构化输出
structured = primary.with_structured_output(Output)

# 2. 再配置重试
with_retry = structured.with_retry(
    retry_if_exception_type=(ConnectionError, TimeoutError),
    stop_after_attempt=3
)

# 3. 最后配置降级
fallback_structured = fallback.with_structured_output(Output)
robust_llm = with_retry.with_fallbacks([fallback_structured])

# 使用
result = robust_llm.invoke("提取信息")
```

---

### 7.4 Pydantic 验证器

**用途**: 自定义数据验证逻辑

**导入路径**: `from pydantic import field_validator`

#### field_validator - 字段验证器

```python
from pydantic import BaseModel, Field, field_validator

class UserInfo(BaseModel):
    """用户信息模型"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """验证姓名不为空"""
        if v.strip() == "":
            raise ValueError('姓名不能为空')
        return v.strip()

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """验证邮箱格式"""
        if '@' not in v:
            raise ValueError('邮箱格式不正确')
        return v.lower()

# 使用
try:
    user = UserInfo(name="  张三  ", age=25, email="Test@Example.COM")
    print(user.name)   # "张三"（已去除空格）
    print(user.email)  # "test@example.com"（已转小写）
except ValueError as e:
    print(f"验证失败: {e}")
```

#### model_validator - 模型级验证器

```python
from pydantic import BaseModel, model_validator

class Order(BaseModel):
    """订单模型"""
    start_date: str
    end_date: str

    @model_validator(mode='after')
    def validate_dates(self):
        """验证结束日期必须晚于开始日期"""
        if self.end_date < self.start_date:
            raise ValueError('结束日期必须晚于开始日期')
        return self
```

#### 配合 LLM 使用

```python
from pydantic import BaseModel, Field, field_validator
from langchain.chat_models import init_chat_model

class Product(BaseModel):
    """产品信息"""
    name: str = Field(description="产品名称")
    price: float = Field(description="价格", gt=0)
    stock: int = Field(description="库存", ge=0)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v) < 2:
            raise ValueError('产品名称至少 2 个字符')
        return v

# 创建结构化输出
model = init_chat_model("groq:llama-3.3-70b-versatile")
structured_llm = model.with_structured_output(Product)

# 调用（LLM 输出会自动验证）
result = structured_llm.invoke("iPhone 15，价格 6999 元，库存 100 台")
print(result.name)   # "iPhone 15"
print(result.price)  # 6999.0
print(result.stock)  # 100
```

---

## 八、输出解析器 API

### 8.1 StrOutputParser - 字符串解析器

**用途**: 从 LLM 输出中提取纯文本字符串

**导入路径**: `from langchain_core.output_parsers import StrOutputParser`

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# 创建组件
model = init_chat_model("groq:llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),
    ("user", "{question}")
])

# 使用管道符连接，最后加 StrOutputParser
chain = prompt | model | StrOutputParser()

# 调用（直接返回字符串，而非 AIMessage 对象）
response = chain.invoke({"question": "什么是 Python？"})
print(response)  # 纯文本字符串
print(type(response))  # <class 'str'>
```

#### 对比：有无 StrOutputParser

```python
# 不使用 StrOutputParser
chain = prompt | model
result = chain.invoke({"question": "你好"})
print(type(result))  # <class 'AIMessage'>
print(result.content)  # 需要访问 .content 属性

# 使用 StrOutputParser
chain = prompt | model | StrOutputParser()
result = chain.invoke({"question": "你好"})
print(type(result))  # <class 'str'>
print(result)  # 直接使用
```

---

### 8.2 JsonOutputParser - JSON 解析器

**用途**: 将 LLM 输出解析为 JSON/字典格式

**导入路径**: `from langchain_core.output_parsers import JsonOutputParser`

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义数据模型
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    city: str = Field(description="城市")

# 创建 JSON 解析器
parser = JsonOutputParser(pydantic_object=Person)

# 在提示词中包含格式说明
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息提取助手。\n{format_instructions}"),
    ("user", "{question}")
])

# 部分填充格式说明
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)

# 构建链
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "question": "张三今年 28 岁，住在北京"
})

print(result)  # {"name": "张三", "age": 28, "city": "北京"}
print(type(result))  # <class 'dict'>
```

#### get_format_instructions 输出示例

```python
print(parser.get_format_instructions())
# 输出类似：
# The output should be formatted as a JSON instance that conforms to
# the JSON schema below.
#
# {
#   "name": {"type": "string", "description": "姓名"},
#   "age": {"type": "integer", "description": "年龄"},
#   "city": {"type": "string", "description": "城市"}
# }
```

---

### 8.3 CommaSeparatedListOutputParser - 列表解析器

**用途**: 将逗号分隔的文本解析为列表

**导入路径**: `from langchain_core.output_parsers import CommaSeparatedListOutputParser`

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "列出 5 种编程语言。\n{format_instructions}"),
    ("user", "{question}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({"question": "列出流行的编程语言"})
print(result)  # ["Python", "JavaScript", "Java", "C++", "Go"]
print(type(result))  # <class 'list'>
```

---

## 九、Runnable 接口详解

### 9.1 RunnablePassthrough - 数据透传

**用途**: 在 LCEL 链中透传输入数据

**导入路径**: `from langchain_core.runnables import RunnablePassthrough`

```python
from langchain_core.runnables import RunnablePassthrough

# 基本用法：透传整个输入
chain = RunnablePassthrough() | model
result = chain.invoke("你好")  # "你好" 被直接传递给 model

# 在字典中使用：保留原始查询
chain = {
    "context": retriever,  # 从检索器获取上下文
    "query": RunnablePassthrough()  # 透传用户查询
} | prompt | model | StrOutputParser()

result = chain.invoke("什么是 RAG？")
# query 会被设置为 "什么是 RAG？"
# context 会被设置为检索到的文档
```

#### 组合使用示例

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题：\n{context}"),
    ("user", "{query}")
])

# 构建链
chain = {
    "context": lambda x: "这是上下文内容",
    "query": RunnablePassthrough()  # 透传用户输入
} | prompt | model | StrOutputParser()

result = chain.invoke("用户的问题")
```

---

### 9.2 RunnableLambda - 自定义函数

**用途**: 将任意 Python 函数包装为 Runnable 组件

**导入路径**: `from langchain_core.runnables import RunnableLambda`

```python
from langchain_core.runnables import RunnableLambda

# 定义处理函数
def to_uppercase(text: str) -> str:
    return text.upper()

def add_prefix(text: str) -> str:
    return f"处理结果: {text}"

# 包装为 Runnable
uppercase_runnable = RunnableLambda(to_uppercase)
prefix_runnable = RunnableLambda(add_prefix)

# 组合到链中
chain = (
    prompt
    | model
    | StrOutputParser()
    | uppercase_runnable
    | prefix_runnable
)

result = chain.invoke({"question": "hello"})
# 输出: "处理结果: HELLO WORLD..."
```

#### 复杂处理示例

```python
def format_docs(docs: list) -> str:
    """将文档列表格式化为字符串"""
    return "\n\n".join([doc.page_content for doc in docs])

def extract_answer(response: dict) -> str:
    """从响应中提取答案"""
    return response.get("answer", "无法回答")

# 在 RAG 链中使用
chain = {
    "context": retriever | RunnableLambda(format_docs),
    "query": RunnablePassthrough()
} | prompt | model | StrOutputParser()
```

---

### 9.3 RunnableParallel - 并行执行

**用途**: 同时执行多个 Runnable，合并结果

**导入路径**: `from langchain_core.runnables import RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 定义多个处理函数
def get_sentiment(text: str) -> str:
    return "positive"  # 模拟情感分析

def get_keywords(text: str) -> list:
    return ["关键词1", "关键词2"]  # 模拟关键词提取

def get_summary(text: str) -> str:
    return "这是摘要"  # 模拟摘要生成

# 并行执行
parallel_chain = RunnableParallel(
    sentiment=RunnableLambda(get_sentiment),
    keywords=RunnableLambda(get_keywords),
    summary=RunnableLambda(get_summary)
)

result = parallel_chain.invoke("这是一段文本")
# 输出: {
#     "sentiment": "positive",
#     "keywords": ["关键词1", "关键词2"],
#     "summary": "这是摘要"
# }
```

---

### 9.4 LCEL 链式调用完整示例

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# 初始化
model = init_chat_model("groq:llama-3.3-70b-versatile")

# 定义处理函数
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。上下文：{context}"),
    ("user", "{question}")
])

# 完整的 RAG 链
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 调用
answer = rag_chain.invoke("什么是 LangChain？")
```

---

## 十、RAG 相关 API

### 10.1 文档加载器

**导入路径**: `from langchain_community.document_loaders import ...`

```python
# 文本文件
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

# PDF 文件
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
# pages = loader.load()  # 返回每一页作为单独的文档

# 网页
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()

# CSV 文件
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()

# Markdown 文件
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md")
documents = loader.load()

# 目录加载（批量加载）
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

# Word 文档
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
documents = loader.load()
```

#### Document 对象结构

```python
from langchain_core.documents import Document

# Document 结构
doc = Document(
    page_content="文档的主要内容文本",
    metadata={
        "source": "文件路径",
        "page": 1,
        "title": "文档标题"
    }
)

# 访问属性
print(doc.page_content)  # 文本内容
print(doc.metadata)      # 元数据字典
```

### 10.2 文本分割器

#### RecursiveCharacterTextSplitter（推荐）

**导入路径**: `from langchain_text_splitters import RecursiveCharacterTextSplitter`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最大字符数
    chunk_overlap=50,      # 块之间的重叠
    separators=["\n\n", "\n", "。", " ", ""],  # 分割优先级
    length_function=len,   # 计算长度的函数
)

chunks = splitter.split_documents(documents)
# 或
texts = splitter.split_text("长文本内容...")
```

#### 其他分割器

```python
# 字符分割器（简单固定长度）
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Token 分割器（按 token 数）
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,      # token 数
    chunk_overlap=50,
    encoding_name="cl100k_base"  # OpenAI 的编码
)

# 代码分割器（保留代码结构）
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    MarkdownTextSplitter
)

python_splitter = PythonCodeTextSplitter(chunk_size=1000)
md_splitter = MarkdownTextSplitter(chunk_size=1000)
```

### 10.3 向量嵌入

#### HuggingFace Embeddings（免费）

**导入路径**: `from langchain_huggingface import HuggingFaceEmbeddings`

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 使用免费的 HuggingFace 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # 其他常用模型：
    # "sentence-transformers/all-mpnet-base-v2"  # 更准确但慢
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多语言
)

# 嵌入单个查询
vector = embeddings.embed_query("什么是 RAG?")
print(f"向量维度: {len(vector)}")  # all-MiniLM-L6-v2 是 384 维

# 批量嵌入文档
vectors = embeddings.embed_documents(["文本1", "文本2"])

# 获取向量维度
query_vector = embeddings.embed_query("test")
dimension = len(query_vector)
```

#### OpenAI Embeddings（付费）

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 维
    # or "text-embedding-3-large"   # 3072 维
    api_key="your-api-key"
)
```

### 10.4 向量存储

#### Chroma（本地，免费）

**导入路径**: `from langchain_community.vectorstores import Chroma`

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 从文档创建（自动嵌入并存储）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
    collection_name="my_collection"   # 集合名称
)

# 从已有索引加载
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# 添加文档
vectorstore.add_documents(new_documents)

# 删除文档
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

#### InMemoryVectorStore（内存中）

**导入路径**: `from langchain_core.vectorstores import InMemoryVectorStore`

```python
from langchain_core.vectorstores import InMemoryVectorStore

# 创建内存向量存储
vectorstore = InMemoryVectorStore(embeddings)

# 添加文档
vectorstore.add_documents(documents)

# 搜索
results = vectorstore.similarity_search("查询文本", k=3)
```

#### Pinecone（云端，免费层）

**导入路径**: `from langchain_pinecone import PineconeVectorStore`

```python
from langchain_pinecone import PineconeVectorStore

# 从文档创建
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# 从已有索引加载
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)
```

### 10.5 向量检索

#### 相似度搜索

```python
# 基本相似度搜索
results = vectorstore.similarity_search(
    "查询文本",
    k=3  # 返回前 3 个结果
)

# 带分数的相似度搜索
results = vectorstore.similarity_search_with_score(
    "查询文本",
    k=3
)
for doc, score in results:
    print(f"分数: {score:.4f}")
    print(f"内容: {doc.page_content}\n")

# 按相似度阈值过滤
results = vectorstore.similarity_search_with_relevance_scores(
    "查询文本",
    k=3,
    score_threshold=0.7  # 只保留相似度 > 0.7 的结果
)
```

#### 转换为 Retriever

```python
# 将 VectorStore 转换为 Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 或 "mmr"（最大边际相关性）
    search_kwargs={
        "k": 3,                # 返回结果数
        "score_threshold": 0.7  # 相似度阈值
    }
)

# 使用 Retriever
results = retriever.invoke("查询文本")
```

### 10.6 BM25 检索器

**导入路径**: `from langchain_community.retrievers import BM25Retriever`

```python
from langchain_community.retrievers import BM25Retriever

# 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=3  # 返回前 3 个结果
)

# 使用
results = bm25_retriever.invoke("查询文本")

# 修改 k 值
bm25_retriever.k = 5
results = bm25_retriever.invoke("新查询")
```

**BM25 优势**：
- 精确匹配专有名词
- 适合代码、版本号查询
- 速度快，无需嵌入

### 10.7 混合检索器

**导入路径**: `from langchain_classic.retrievers import EnsembleRetriever`

```python
from langchain_classic.retrievers import EnsembleRetriever

# 创建混合检索器（组合多个检索器）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 权重平衡
)

# 使用
results = ensemble_retriever.invoke("查询文本")

# 不同的权重配置
weights_semantic = [0.4, 0.6]    # 偏向语义（文章、对话）
weights_exact = [0.6, 0.4]        # 偏向精确匹配（代码、配置）
weights_balanced = [0.5, 0.5]     # 平衡（默认推荐）
```

**权重选择建议**：
- 技术文档：`[0.4, 0.6]` - 稍偏向语义
- 代码库：`[0.6, 0.4]` - 偏向精确匹配
- 混合内容：`[0.5, 0.5]` - 平衡

### 10.8 RAG 链构建

#### 使用 LCEL 构建 RAG 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RAG 提示词模板
rag_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。根据以下上下文回答问题：\n\n{context}"),
    ("user", "{question}")
])

# 创建 RAG 链
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | rag_template
    | model
    | StrOutputParser()
)

# 调用
response = rag_chain.invoke("什么是 RAG？")
```

#### 使用 Agent 构建交互式 RAG

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# 将检索器封装为工具
@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关信息"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 创建 RAG Agent
rag_agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="""你是一个智能助手。
使用 search_knowledge_base 工具检索信息，然后基于检索到的内容回答问题。
如果检索到的信息不足以回答问题，请如实告知用户。"""
)

# 调用
response = rag_agent.invoke({
    "messages": [{"role": "user", "content": "LangChain 有哪些特性？"}]
})

final_answer = response['messages'][-1].content
```

### 10.9 RAG 高级特性

#### 元数据过滤

```python
# 只搜索特定元数据的文档
results = vectorstore.similarity_search(
    "查询",
    k=3,
    filter={"source": "important_docs.pdf"}  # 只搜索这个来源的文档
)

# 多条件过滤
results = vectorstore.similarity_search(
    "查询",
    k=3,
    filter={
        "category": "技术",
        "year": 2024
    }
)
```

#### MMR 检索（最大边际相关性）

```python
# 使用 MMR 增加结果多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,  # 初始获取 10 个，然后选 5 个最多样化的
        "lambda_mult": 0.5  # 0=多样性优先，1=相关性优先
    }
)

results = retriever.invoke("查询文本")
```

#### 上下文压缩

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 压缩检索到的文档，只保留相关部分
compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 使用压缩后的检索器
results = compression_retriever.invoke("查询文本")
```

### 10.10 RAG 最佳实践

#### 分块策略

```python
# 根据文档类型选择分块配置
configs = {
    "代码": {"chunk_size": 500, "chunk_overlap": 50},
    "文章": {"chunk_size": 1000, "chunk_overlap": 200},
    "书籍": {"chunk_size": 1500, "chunk_overlap": 300},
    "问答": {"chunk_size": 300, "chunk_overlap": 30}
}

splitter = RecursiveCharacterTextSplitter(**configs["文章"])
```

#### 检索数量 (k 值)

```python
# k 值选择
k_values = {
    "简单问题": 1,      # 只取最相关的
    "常规问题": 3,      # 推荐（平衡）
    "复杂问题": 5,      # 更全面
    "研究深度": 10      # 大量上下文（注意成本）
}
```

#### 权重配置

```python
# 根据数据类型选择混合检索权重
weight_configs = {
    "技术文档": [0.4, 0.6],    # 偏向语义
    "代码库": [0.6, 0.4],      # 偏向精确
    "混合内容": [0.5, 0.5],     # 平衡
    "对话": [0.3, 0.7]         # 强语义
}

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=weight_configs["混合内容"]
)
```

---

## 十一、LangGraph 核心 API

### 11.1 StateGraph - 状态图

**导入路径**: `from langgraph.graph import StateGraph, START, END`

#### 基本语法

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_step: str

# 2. 定义节点
def my_node(state: State) -> dict:
    return {"current_step": "completed"}

# 3. 创建图
graph = StateGraph(State)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"messages": [], "current_step": "start"})
```

#### 添加条件边

```python
def route_function(state: State) -> str:
    if state["current_step"] == "done":
        return "end"
    return "continue"

graph.add_conditional_edges(
    "my_node",               # 参数1: 起始节点名称 - 指定从哪个节点开始进行条件路由
    route_function,          # 参数2: 路由函数 - 接收状态作为输入，返回字符串决定下一步走向哪个节点
    {"continue": "next_node", "end": END}  # 参数3: 条件映射字典 - 定义路由函数返回值与目标节点的映射关系
)
```

### 11.2 状态持久化

**导入路径**: `from langgraph.checkpoint.memory import MemorySaver`

```python
from langgraph.checkpoint.memory import MemorySaver

# 添加内存检查点
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用 thread_id 进行会话管理
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input_data, config=config)
```

---

## 十二、向量数据库 API

### 12.1 Pinecone 集成

**导入路径**: `from langchain_pinecone import PineconeVectorStore`

```python
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 从文档创建（自动嵌入并存储）
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# 从已有索引加载
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)

# 检索相似文档
docs = vectorstore.similarity_search("查询文本", k=3)
```

### 12.2 Chroma 集成

**导入路径**: `from langchain_chroma import Chroma`

```python
from langchain_chroma import Chroma

# 从文档创建
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 检索
docs = vectorstore.similarity_search("查询", k=3)
```

---

## 十三、文档处理 API

### 13.1 PDF 处理

```python
from langchain_community.document_loaders import PyPDFLoader

# 使用 PyPDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# 使用 PyMuPDF（更快）
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("document.pdf")
pages = loader.load()
```

### 13.2 Word 文档

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
docs = loader.load()
```

### 13.3 Markdown

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("document.md")
docs = loader.load()
```

---

## 十四、状态持久化 API

### 14.1 MemorySaver

**导入路径**: `from langgraph.checkpoint.memory import MemorySaver`

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建内存检查点
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用 thread_id
config = {"configurable": {"thread_id": "session_123"}}
response = app.invoke(input_state, config=config)
```

### 14.2 SQLite 检查点

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建 SQLite 检查点
conn = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=conn)
```

---

## 十五、外部服务集成

### 15.1 LangSmith 可观测性

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key"

# 自动追踪所有 LangChain 调用
response = model.invoke("Hello")
```

### 15.2 Groq API

```python
# 环境变量
export GROQ_API_KEY="gsk_..."

# 使用
from langchain.chat_models import init_chat_model
model = init_chat_model("groq:llama-3.3-70b-versatile")
```

### 15.3 OpenAI API

```python
# 环境变量
export OPENAI_API_KEY="sk-..."

# 使用
from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4o-mini")
```

---

## 📊 RAG 快速参考

### RAG 完整工作流程

```python
# ========== 1. 加载文档 ==========
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# 单个文件
loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

# 目录批量加载
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()


# ========== 2. 分割文本 ==========
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " ", ""]
)
chunks = splitter.split_documents(documents)


# ========== 3. 创建向量嵌入 ==========
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ========== 4. 创建向量存储 ==========
# 方式 A: Chroma（本地，推荐）
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 方式 B: InMemoryVectorStore（内存）
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 方式 C: Pinecone（云端）
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)


# ========== 5. 创建检索器 ==========
# 向量检索器
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# BM25 检索器（精确匹配）
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 混合检索器（向量 + BM25）
from langchain_classic.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # [BM25权重, 向量权重]
)


# ========== 6. 创建 RAG 链 ==========
# 方式 A: LCEL 链（简单）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答：\n\n{context}"),
    ("user", "{question}")
])

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("什么是 RAG？")


# 方式 B: Agent（交互式）
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """在文档库中搜索信息"""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

rag_agent = create_agent(
    model=model,
    tools=[search_docs],
    system_prompt="使用 search_docs 工具搜索信息，然后回答问题。"
)

response = rag_agent.invoke({
    "messages": [{"role": "user", "content": "查询内容"}]
})
```

### RAG 检索方法对比

| 检索方法 | 优势 | 劣势 | 适用场景 |
|---------|------|------|----------|
| **向量检索** | 理解语义、同义词 | 精确匹配差 | 文章、对话 |
| **BM25** | 精确匹配、速度快 | 不理解语义 | 代码、专有名词 |
| **混合检索** | 兼顾语义和精确 | 稍慢 | 生产环境推荐 |

### RAG 配置建议

#### 分块配置

```python
# 根据文档类型选择
configs = {
    "代码": {"chunk_size": 500, "chunk_overlap": 50},
    "文章": {"chunk_size": 1000, "chunk_overlap": 200},
    "书籍": {"chunk_size": 1500, "chunk_overlap": 300},
    "问答": {"chunk_size": 300, "chunk_overlap": 30}
}
```

#### 混合检索权重

```python
# 根据数据类型选择
weights = {
    "技术文档": [0.4, 0.6],    # 偏向语义
    "代码库": [0.6, 0.4],      # 偏向精确
    "混合内容": [0.5, 0.5],     # 平衡（推荐）
    "对话": [0.3, 0.7]         # 强语义
}
```

#### k 值选择（检索数量）

```python
k_values = {
    "简单问题": 1,      # 只取最相关的
    "常规问题": 3,      # 推荐（平衡）
    "复杂问题": 5,      # 更全面
    "研究深度": 10      # 大量上下文
}
```

### 常用向量数据库对比

| 特性 | Chroma | Pinecone | InMemoryVectorStore |
|-----|--------|----------|---------------------|
| 部署 | 本地 | 云端 | 内存 |
| 成本 | 免费 | 免费层 10GB | 免费 |
| 速度 | 中 | 快 | 最快 |
| 规模 | 小-中 | 大规模 | 小 |
| **推荐** | 开发/学习 | 生产环境 | 测试/演示 |

---

## 📊 API 快速参考表

### 核心导入速查

```python
# ========== 模型 ==========
from langchain.chat_models import init_chat_model

# ========== 提示词 ==========
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# ========== 消息 ==========
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ========== 工具 ==========
from langchain_core.tools import tool

# ========== Agent ==========
from langchain.agents import create_agent

# ========== 结构化输出 ==========
from pydantic import BaseModel, Field, field_validator, model_validator

# ========== 输出解析器 ==========
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser

# ========== Runnable 接口 ==========
from langchain_core.runnables import (
    RunnablePassthrough,   # 透传数据
    RunnableLambda,        # 自定义函数
    RunnableParallel       # 并行执行
)

# ========== LangGraph ==========
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ========== RAG - 文档加载 ==========
from langchain_community.document_loaders import (
    TextLoader,           # 文本文件
    PyPDFLoader,          # PDF 文件
    WebBaseLoader,        # 网页
    CSVLoader,            # CSV 文件
    DirectoryLoader,      # 目录批量加载
    Docx2txtLoader,       # Word 文档
    UnstructuredMarkdownLoader  # Markdown
)

# ========== RAG - 文本分割 ==========
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 递归分割（推荐）
    CharacterTextSplitter,           # 字符分割
    TokenTextSplitter,               # Token 分割
    PythonCodeTextSplitter,          # Python 代码
    MarkdownTextSplitter             # Markdown
)

# ========== RAG - 向量嵌入 ==========
from langchain_huggingface import HuggingFaceEmbeddings  # 免费 HuggingFace
from langchain_openai import OpenAIEmbeddings            # OpenAI（付费）

# ========== RAG - 向量存储 ==========
from langchain_community.vectorstores import Chroma      # 本地 Chroma
from langchain_core.vectorstores import InMemoryVectorStore  # 内存存储
from langchain_pinecone import PineconeVectorStore       # Pinecone 云

# ========== RAG - 检索器 ==========
from langchain_community.retrievers import BM25Retriever  # BM25 关键词检索
from langchain_classic.retrievers import EnsembleRetriever  # 混合检索

# ========== RAG - 核心 ==========
from langchain_core.documents import Document  # 文档对象

# ========== 状态持久化 ==========
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# ========== 多模态 ==========
import base64  # 图像编码

# ========== MCP ==========
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ========== 回调系统 ==========
from langchain_core.callbacks import BaseCallbackHandler

# ========== 中间件 ==========
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware

# ========== 消息管理 ==========
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import trim_messages

# ========== 工具高级 ==========
from langchain_core.tools import StructuredTool, ToolException, BaseTool

# ========== 配置管理 ==========
from langchain_core.runnables import RunnableConfig
```

---

## 十六、多模态 API

### 16.1 图像输入

**用途**: 向支持视觉的 LLM 发送图像进行分析

**支持的模型**: OpenAI GPT-4o、Google Gemini、Anthropic Claude 3.5+

#### Base64 编码方式

```python
import base64
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

# 读取并编码图像
def encode_image(image_path: str) -> str:
    """将图像文件编码为 Base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 编码图像
base64_image = encode_image("images/example.jpg")

# 创建支持视觉的模型
model = init_chat_model("openai:gpt-4o-mini")

# 构建图像消息
message = HumanMessage(content=[
    {"type": "text", "text": "请描述这张图片的内容"},
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
])

# 调用模型
response = model.invoke([message])
print(response.content)
```

#### URL 方式（公开图像）

```python
# 直接使用公开图像 URL
message = HumanMessage(content=[
    {"type": "text", "text": "这张图片里有什么？"},
    {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.jpg"
        }
    }
])

response = model.invoke([message])
```

#### 多图像输入

```python
# 同时分析多张图像
message = HumanMessage(content=[
    {"type": "text", "text": "比较这两张图片的差异"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image1_base64}"}
    },
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image2_base64}"}
    }
])

response = model.invoke([message])
```

#### 支持的图像格式

| 格式 | MIME 类型 | 说明 |
|------|-----------|------|
| JPEG | `image/jpeg` | 照片、复杂图像 |
| PNG | `image/png` | 截图、带透明度图像 |
| GIF | `image/gif` | 动画（部分模型支持） |
| WebP | `image/webp` | 现代格式，更小体积 |

---

### 16.2 文件处理

**用途**: 处理各种文件格式的数据

#### CSV 文件处理

```python
import csv
from langchain_core.documents import Document

# 读取 CSV 文件
def load_csv(file_path: str) -> list[Document]:
    """将 CSV 文件转换为 Document 列表"""
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 将每行转换为文档
            content = ", ".join([f"{k}: {v}" for k, v in row.items()])
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "type": "csv"}
            )
            documents.append(doc)
    return documents

# 使用
docs = load_csv("data/products.csv")
print(f"加载了 {len(docs)} 条记录")
```

#### JSON 文件处理

```python
import json
from langchain_core.documents import Document

def load_json(file_path: str) -> list[Document]:
    """将 JSON 文件转换为 Document 列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    # 处理 JSON 数组
    if isinstance(data, list):
        for i, item in enumerate(data):
            doc = Document(
                page_content=json.dumps(item, ensure_ascii=False),
                metadata={"source": file_path, "index": i, "type": "json"}
            )
            documents.append(doc)
    # 处理单个 JSON 对象
    else:
        doc = Document(
            page_content=json.dumps(data, ensure_ascii=False),
            metadata={"source": file_path, "type": "json"}
        )
        documents.append(doc)

    return documents

# 使用
docs = load_json("data/products.json")
```

#### Excel 文件处理

```python
import pandas as pd
from langchain_core.documents import Document

def load_excel(file_path: str, sheet_name: str = None) -> list[Document]:
    """将 Excel 文件转换为 Document 列表"""
    # 读取 Excel
    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path)

    documents = []
    for idx, row in df.iterrows():
        content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "row": idx,
                "type": "excel"
            }
        )
        documents.append(doc)

    return documents

# 使用
docs = load_excel("data/sales.xlsx")
```

---

### 16.3 混合模态处理

**用途**: 同时处理文本、图像等多种输入

#### 文本 + 图像 RAG

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# 创建多模态提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个图像分析助手。根据图像和文本上下文回答问题。"),
    ("user", "{input}")
])

# 构建输入
def create_multimodal_input(query: str, image_path: str = None):
    """创建多模态输入"""
    content = []

    # 添加文本
    content.append({"type": "text", "text": query})

    # 添加图像（如果有）
    if image_path:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    return HumanMessage(content=content)

# 使用
model = init_chat_model("openai:gpt-4o-mini")
message = create_multimodal_input(
    query="这张图表显示了什么趋势？",
    image_path="charts/sales_trend.png"
)
response = model.invoke([message])
```

#### 批量图像分析

```python
import os
from pathlib import Path

def analyze_image_directory(
    directory: str,
    prompt: str,
    model
) -> dict[str, str]:
    """批量分析目录中的所有图像"""
    results = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    for file_path in Path(directory).iterdir():
        if file_path.suffix.lower() in image_extensions:
            try:
                base64_image = encode_image(str(file_path))
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ])
                response = model.invoke([message])
                results[file_path.name] = response.content
            except Exception as e:
                results[file_path.name] = f"错误: {str(e)}"

    return results

# 使用
model = init_chat_model("openai:gpt-4o-mini")
results = analyze_image_directory(
    "images/products",
    "描述这个产品的外观特征",
    model
)
for filename, description in results.items():
    print(f"{filename}: {description[:100]}...")
```

---

### 16.4 多模态最佳实践

#### 图像大小优化

```python
from PIL import Image
import io

def optimize_image(image_path: str, max_size: int = 1024) -> str:
    """优化图像大小以减少 token 消耗"""
    with Image.open(image_path) as img:
        # 调整大小
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 转换为 JPEG
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 使用优化后的图像
optimized_base64 = optimize_image("large_image.png")
```

#### 错误处理

```python
def safe_image_analysis(model, image_path: str, prompt: str) -> dict:
    """安全的图像分析，包含错误处理"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return {"success": False, "error": "文件不存在"}

        # 检查文件大小（限制 20MB）
        if os.path.getsize(image_path) > 20 * 1024 * 1024:
            return {"success": False, "error": "文件过大（最大 20MB）"}

        # 编码图像
        base64_image = encode_image(image_path)

        # 调用模型
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ])
        response = model.invoke([message])

        return {"success": True, "content": response.content}

    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## 十七、MCP (Model Context Protocol) 集成

### 17.1 MCP 概述

**用途**: 通过 MCP 协议集成外部工具和服务

**MCP** 是一种标准化的工具协议，允许 LLM 与外部系统（文件系统、数据库、API）进行交互。

---

### 17.2 创建 MCP 服务器

**导入路径**: `from mcp.server import Server`

```python
# filesystem_server.py - 文件系统 MCP 服务器
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建服务器
server = Server("filesystem-server")

@server.list_tools()
async def list_tools():
    """列出可用工具"""
    return [
        Tool(
            name="read_file",
            description="读取文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="写入文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["path", "content"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """执行工具调用"""
    if name == "read_file":
        with open(arguments["path"], "r") as f:
            content = f.read()
        return [TextContent(type="text", text=content)]

    elif name == "write_file":
        with open(arguments["path"], "w") as f:
            f.write(arguments["content"])
        return [TextContent(type="text", text="文件写入成功")]

    raise ValueError(f"未知工具: {name}")

# 启动服务器
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### 17.3 MCP 工具适配器

**用途**: 将 MCP 工具转换为 LangChain 工具

```python
# utils/mcp_adapter.py
import subprocess
import json
from langchain_core.tools import tool

def create_mcp_tools(server_path: str) -> list:
    """从 MCP 服务器创建 LangChain 工具

    参数:
        server_path: MCP 服务器 Python 文件路径

    返回:
        LangChain 工具列表
    """
    # 这里是简化的适配器实现
    # 实际项目中需要使用 MCP SDK 进行完整的进程管理

    @tool
    def read_file(path: str) -> str:
        """读取文件内容

        参数:
            path: 文件路径

        返回:
            文件内容
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @tool
    def write_file(path: str, content: str) -> str:
        """写入文件内容

        参数:
            path: 文件路径
            content: 要写入的内容

        返回:
            操作结果
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "文件写入成功"

    return [read_file, write_file]


def create_multi_mcp_tools(server_paths: list[str]) -> list:
    """从多个 MCP 服务器创建工具

    参数:
        server_paths: MCP 服务器文件路径列表

    返回:
        合并后的 LangChain 工具列表
    """
    all_tools = []
    for server_path in server_paths:
        tools = create_mcp_tools(server_path)
        all_tools.extend(tools)
    return all_tools
```

---

### 17.4 使用 MCP 工具

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from utils.mcp_adapter import create_mcp_tools, create_multi_mcp_tools

# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile")

# 方式 1: 加载单个 MCP 服务器的工具
tools = create_mcp_tools("servers/filesystem_server.py")

# 方式 2: 加载多个 MCP 服务器的工具
tools = create_multi_mcp_tools([
    "servers/filesystem_server.py",
    "servers/search_server.py"
])

# 创建 Agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""你是一个文件管理助手。
使用提供的工具来读取和写入文件。
"""
)

# 调用 Agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "读取 config.json 文件的内容"}]
})

print(response['messages'][-1].content)
```

---

### 17.5 MCP 服务器示例：搜索服务

```python
# servers/search_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from duckduckgo_search import DDGS

server = Server("search-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="web_search",
            description="在网络上搜索信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "web_search":
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        output = "\n\n".join([
            f"**{r['title']}**\n{r['body']}\n{r['href']}"
            for r in results
        ])
        return [TextContent(type="text", text=output)]

    raise ValueError(f"未知工具: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 十八、错误处理模式

### 18.1 超时处理

**用途**: 防止 LLM 调用长时间阻塞

```python
import concurrent.futures
from typing import Optional

def invoke_with_timeout(
    model,
    query: str,
    timeout: float = 30.0
) -> dict:
    """带超时的 LLM 调用

    参数:
        model: LLM 模型实例
        query: 查询文本
        timeout: 超时时间（秒）

    返回:
        包含 success 和 content/error 的字典
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.invoke, query)
            result = future.result(timeout=timeout)
            return {
                "success": True,
                "content": result.content
            }
    except concurrent.futures.TimeoutError:
        return {
            "success": False,
            "error": f"请求超时（超过 {timeout} 秒）"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# 使用
model = init_chat_model("groq:llama-3.3-70b-versatile")
result = invoke_with_timeout(model, "你好", timeout=10.0)

if result["success"]:
    print(result["content"])
else:
    print(f"错误: {result['error']}")
```

---

### 18.2 自定义错误处理器

**用途**: 统一处理不同类型的错误

```python
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class ErrorContext:
    """错误上下文"""
    operation: str
    query: str
    timestamp: str

class ErrorHandler:
    """统一错误处理器"""

    # 错误类型到用户消息的映射
    USER_MESSAGES = {
        "ConnectionError": "网络连接失败，请检查网络后重试",
        "TimeoutError": "请求超时，请稍后重试",
        "AuthenticationError": "API 密钥无效，请检查配置",
        "RateLimitError": "请求过于频繁，请稍后重试",
        "ValidationError": "输入数据验证失败",
        "ModelNotFoundError": "模型不可用，请尝试其他模型"
    }

    # 可重试的错误类型
    RETRYABLE_ERRORS = {
        "ConnectionError",
        "TimeoutError",
        "RateLimitError"
    }

    def handle(
        self,
        error: Exception,
        context: ErrorContext
    ) -> dict:
        """处理错误并返回友好的响应

        参数:
            error: 异常对象
            context: 错误上下文

        返回:
            包含错误信息的字典
        """
        error_type = type(error).__name__

        return {
            "success": False,
            "error_type": error_type,
            "user_message": self.USER_MESSAGES.get(
                error_type,
                f"处理失败: {str(error)}"
            ),
            "can_retry": error_type in self.RETRYABLE_ERRORS,
            "context": context
        }

# 使用
error_handler = ErrorHandler()

try:
    response = model.invoke("你好")
except Exception as e:
    ctx = ErrorContext(
        operation="invoke",
        query="你好",
        timestamp="2024-01-15 10:30:00"
    )
    result = error_handler.handle(e, ctx)

    print(result["user_message"])
    if result["can_retry"]:
        print("可以重试此操作")
```

---

### 18.3 重试装饰器

**用途**: 为函数添加自动重试功能

```python
import functools
import time
from typing import Callable, Type, Tuple

def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """重试装饰器

    参数:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 延迟增长倍数
        exceptions: 触发重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"第 {attempt + 1} 次尝试失败: {e}")
                        print(f"等待 {current_delay} 秒后重试...")
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator

# 使用
@retry_on_error(
    max_attempts=3,
    delay=1.0,
    backoff=2.0,
    exceptions=(ConnectionError, TimeoutError)
)
def call_llm(model, query: str):
    """带重试的 LLM 调用"""
    return model.invoke(query)

# 调用
model = init_chat_model("groq:llama-3.3-70b-versatile")
response = call_llm(model, "你好")
```

---

### 18.4 优雅降级

**用途**: 在主功能失败时提供备用方案

```python
from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class FallbackConfig:
    """降级配置"""
    enable_cache: bool = True
    enable_default_response: bool = True
    default_response: str = "抱歉，服务暂时不可用"

class GracefulDegradation:
    """优雅降级处理器"""

    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.cache = {}

    def invoke_with_fallback(
        self,
        model,
        query: str,
        fallback_model=None
    ) -> dict:
        """带降级的调用

        优先级:
        1. 主模型
        2. 备用模型（如果提供）
        3. 缓存（如果启用）
        4. 默认响应
        """
        # 尝试主模型
        try:
            response = model.invoke(query)
            # 缓存成功响应
            if self.config.enable_cache:
                self.cache[query] = response.content
            return {
                "success": True,
                "content": response.content,
                "source": "primary_model"
            }
        except Exception as primary_error:
            print(f"主模型失败: {primary_error}")

        # 尝试备用模型
        if fallback_model:
            try:
                response = fallback_model.invoke(query)
                return {
                    "success": True,
                    "content": response.content,
                    "source": "fallback_model"
                }
            except Exception as fallback_error:
                print(f"备用模型失败: {fallback_error}")

        # 尝试缓存
        if self.config.enable_cache and query in self.cache:
            return {
                "success": True,
                "content": self.cache[query],
                "source": "cache"
            }

        # 返回默认响应
        if self.config.enable_default_response:
            return {
                "success": False,
                "content": self.config.default_response,
                "source": "default"
            }

        return {
            "success": False,
            "error": "所有方案都失败",
            "source": "none"
        }

# 使用
degradation = GracefulDegradation(FallbackConfig(
    enable_cache=True,
    default_response="服务暂时不可用，请稍后重试"
))

primary_model = init_chat_model("openai:gpt-4o-mini")
fallback_model = init_chat_model("groq:llama-3.3-70b-versatile")

result = degradation.invoke_with_fallback(
    primary_model,
    "什么是 AI？",
    fallback_model=fallback_model
)

print(f"来源: {result['source']}")
print(f"内容: {result['content']}")
```

---

## 十九、图可视化 API

### 19.1 get_graph - 获取图结构

**用途**: 获取 LangGraph 应用的图结构，用于可视化和调试

```python
from langgraph.graph import StateGraph, START, END

# 编译图
app = graph.compile()

# 获取图对象
graph_obj = app.get_graph()
```

---

### 19.2 print_ascii - ASCII 可视化

**用途**: 在终端中以 ASCII 艺术形式打印图结构

```python
# 打印 ASCII 图
app.get_graph().print_ascii()

# 输出示例:
#            ┌─────────────┐
#            │   START     │
#            └──────┬──────┘
#                   │
#            ┌──────▼──────┐
#            │   agent     │
#            └──────┬──────┘
#                   │
#        ┌──────────┼──────────┐
#        │          │          │
# ┌──────▼──────┐ ┌─▼───┐ ┌────▼────┐
# │   tools     │ │ end │ │ continue│
# └─────────────┘ └─────┘ └─────────┘
```

---

### 19.3 draw_mermaid - Mermaid 图表

**用途**: 生成 Mermaid 格式的图表代码，可在支持 Mermaid 的环境中渲染

```python
# 获取 Mermaid 代码
mermaid_code = app.get_graph().draw_mermaid()
print(mermaid_code)

# 输出示例:
# graph TD
#     START --> agent
#     agent --> tools
#     tools --> agent
#     agent --> END
```

#### 在线渲染

将生成的 Mermaid 代码复制到以下位置渲染：
- [Mermaid Live Editor](https://mermaid.live/)
- Markdown 文档中的 ` ```mermaid ` 代码块
- GitHub README 文件

---

### 19.4 完整示例

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    step: str

# 定义节点
def agent_node(state: State) -> dict:
    return {"step": "agent_done"}

def tools_node(state: State) -> dict:
    return {"step": "tools_done"}

# 路由函数
def should_continue(state: State) -> str:
    if state["step"] == "need_tools":
        return "tools"
    return "end"

# 构建图
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

# 编译
app = graph.compile()

# 可视化
print("=== ASCII 图 ===")
app.get_graph().print_ascii()

print("\n=== Mermaid 图 ===")
print(app.get_graph().draw_mermaid())
```

---

## 二十、中间件系统 API

### 20.1 AgentMiddleware - 中间件基类

**用途**: 创建自定义中间件，在模型调用前后执行自定义逻辑

**导入路径**: `from langchain.agents.middleware import AgentMiddleware`

```python
from langchain.agents.middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    """自定义中间件"""

    def before_model(self, state, runtime):
        """模型调用前的钩子

        参数:
            state: 当前状态
            runtime: 运行时信息

        返回:
            可修改状态或返回 None
        """
        print(f"准备调用模型，消息数: {len(state.get('messages', []))}")
        # 可以在这里添加、修改或过滤消息
        return None

    def after_model(self, state, runtime):
        """模型响应后的钩子

        参数:
            state: 当前状态（包含模型响应）
            runtime: 运行时信息

        返回:
            可修改状态或返回 None
        """
        last_message = state.get('messages', [])[-1] if state.get('messages') else None
        if last_message:
            print(f"模型响应: {last_message.content[:100]}...")
        return None
```

---

### 20.2 SummarizationMiddleware - 摘要中间件

**用途**: 自动对旧消息进行摘要，保持上下文窗口在限制内

**导入路径**: `from langchain.agents.middleware import SummarizationMiddleware`

```python
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 创建摘要中间件
summarization_middleware = SummarizationMiddleware(
    model=model,                        # 用于生成摘要的模型
    max_tokens_before_summary=2000,     # 触发摘要的 token 阈值
    max_summary_tokens=200              # 摘要的最大 token 数
)

# 配置到 Agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是助手",
    middleware=[summarization_middleware]  # 添加中间件
)
```

#### 工作原理

1. 每次 Agent 调用前，检查消息历史的 token 数
2. 如果超过 `max_tokens_before_summary`，则：
   - 保留最近的几条消息
   - 将旧消息发送给 LLM 生成摘要
   - 用摘要替换旧消息

---

### 20.3 自定义中间件示例

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    """日志记录中间件"""

    def __init__(self, log_file: str = "agent.log"):
        self.log_file = log_file

    def before_model(self, state, runtime):
        """记录调用前状态"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "before_model",
            "message_count": len(state.get("messages", []))
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return None

    def after_model(self, state, runtime):
        """记录调用后状态"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "after_model",
            "last_message_type": type(state["messages"][-1]).__name__ if state.get("messages") else None
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return None


class TokenCounterMiddleware(AgentMiddleware):
    """Token 计数中间件"""

    def __init__(self, model):
        self.model = model
        self.total_tokens = 0

    def after_model(self, state, runtime):
        """统计 token 使用"""
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if last_message and hasattr(last_message, "usage_metadata"):
            tokens = last_message.usage_metadata.get("total_tokens", 0)
            self.total_tokens += tokens
            print(f"本次: {tokens} tokens, 累计: {self.total_tokens} tokens")
        return None

# 使用
logging_middleware = LoggingMiddleware()
token_counter = TokenCounterMiddleware(model)

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[logging_middleware, token_counter]
)
```

---

## 二十一、回调系统 API

### 21.1 BaseCallbackHandler - 回调基类

**用途**: 创建自定义回调处理器，监控和记录 LLM 调用过程

**导入路径**: `from langchain_core.callbacks import BaseCallbackHandler`

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器"""
    pass
```

---

### 21.2 回调方法详解

```python
from langchain_core.callbacks import BaseCallbackHandler

class DetailedCallbackHandler(BaseCallbackHandler):
    """详细的回调处理器"""

    # ========== LLM 相关 ==========

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始时触发"""
        print(f"[LLM 开始] 提示词: {prompts[0][:50]}...")

    def on_llm_end(self, response, **kwargs):
        """LLM 结束时触发"""
        print(f"[LLM 结束] 输出: {response.generations[0][0].text[:50]}...")

    def on_llm_error(self, error, **kwargs):
        """LLM 出错时触发"""
        print(f"[LLM 错误] {error}")

    def on_llm_new_token(self, token, **kwargs):
        """每个新 token 时触发（流式输出）"""
        print(token, end="", flush=True)

    # ========== 工具相关 ==========

    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具开始时触发"""
        tool_name = serialized.get("name", "unknown")
        print(f"[工具开始] {tool_name}: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """工具结束时触发"""
        print(f"[工具结束] 输出: {output}")

    def on_tool_error(self, error, **kwargs):
        """工具出错时触发"""
        print(f"[工具错误] {error}")

    # ========== Chain 相关 ==========

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Chain 开始时触发"""
        print(f"[Chain 开始] 输入: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        """Chain 结束时触发"""
        print(f"[Chain 结束] 输出: {outputs}")

    def on_chain_error(self, error, **kwargs):
        """Chain 出错时触发"""
        print(f"[Chain 错误] {error}")

    # ========== Agent 相关 ==========

    def on_agent_action(self, action, **kwargs):
        """Agent 执行动作时触发"""
        print(f"[Agent 动作] {action.tool}: {action.tool_input}")

    def on_agent_finish(self, finish, **kwargs):
        """Agent 完成时触发"""
        print(f"[Agent 完成] {finish.return_values}")
```

---

### 21.3 使用回调处理器

```python
from langchain_core.callbacks import BaseCallbackHandler

# 创建回调处理器
handler = DetailedCallbackHandler()

# 方式 1: 在 invoke 中使用
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config={"callbacks": [handler]}
)

# 方式 2: 在链中使用
chain = prompt | model | StrOutputParser()
response = chain.invoke(
    {"query": "问题"},
    config={"callbacks": [handler]}
)

# 方式 3: 多个回调处理器
handler1 = LoggingCallbackHandler()
handler2 = MetricsCallbackHandler()

response = agent.invoke(
    query,
    config={"callbacks": [handler1, handler2]}
)
```

---

### 21.4 实用回调处理器示例

```python
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
import json

class PerformanceCallbackHandler(BaseCallbackHandler):
    """性能监控回调"""

    def __init__(self):
        self.start_times = {}
        self.metrics = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.start_times[f"tool_{tool_name}"] = datetime.now()

    def on_tool_end(self, output, **kwargs):
        # 这里需要知道是哪个工具结束
        pass

    def get_metrics(self):
        return self.metrics


class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调"""

    def on_llm_new_token(self, token, **kwargs):
        """实时打印每个 token"""
        print(token, end="", flush=True)


class FileLoggingCallbackHandler(BaseCallbackHandler):
    """文件日志回调"""

    def __init__(self, log_file: str = "langchain.log"):
        self.log_file = log_file

    def _log(self, event: str, data: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._log("tool_start", {
            "tool": serialized.get("name"),
            "input": input_str
        })

    def on_tool_end(self, output, **kwargs):
        self._log("tool_end", {"output": str(output)})

    def on_tool_error(self, error, **kwargs):
        self._log("tool_error", {"error": str(error)})
```

---

## 二十二、消息管理 API

### 22.1 MessagesPlaceholder - 消息占位符

**用途**: 在提示词模板中为对话历史预留位置

**导入路径**: `from langchain_core.prompts import MessagesPlaceholder`

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建带消息占位符的模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{query}")
])

# 使用 - chat_history 可以是消息列表
chain = prompt | model | StrOutputParser()

response = chain.invoke({
    "query": "我刚才说了什么？",
    "chat_history": [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好张三！"}
    ]
})
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `variable_name` | `str` | 必需 | 变量名，用于传入消息历史 |
| `optional` | `bool` | `False` | 是否可选（为 True 时不传也不会报错） |

---

### 22.2 trim_messages - 消息修剪

**用途**: 修剪消息列表以适应上下文窗口限制

**导入路径**: `from langchain_core.messages import trim_messages`

```python
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是助手"),
    HumanMessage(content="问题1"),
    AIMessage(content="回答1"),
    HumanMessage(content="问题2"),
    AIMessage(content="回答2"),
    # ... 更多消息
    HumanMessage(content="最新问题"),
]

# 基本修剪
trimmed = trim_messages(
    messages,
    max_count=10,           # 最大消息数量
    strategy="last",        # 保留最新的消息
)

print(f"原始消息数: {len(messages)}")
print(f"修剪后: {len(trimmed)}")
```

#### 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `messages` | `list` | 消息列表 |
| `max_count` | `int` | 最大消息数量 |
| `max_tokens` | `int` | 最大 token 数（可选） |
| `strategy` | `str` | `"last"` 保留最新 / `"first"` 保留最早 |
| `token_counter` | `callable` | token 计数函数（通常是 model） |
| `include_system` | `bool` | 是否始终包含系统消息 |
| `allow_partial` | `bool` | 是否允许截断最后一条消息 |

#### 带 Token 限制的修剪

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 基于 token 数修剪
trimmed = trim_messages(
    messages,
    max_tokens=4000,        # 最大 4000 tokens
    strategy="last",
    token_counter=model,    # 使用模型计算 token
    include_system=True     # 始终保留系统消息
)
```

---

### 22.3 消息类型检查

**用途**: 检查消息类型以便处理

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

for msg in messages:
    if isinstance(msg, SystemMessage):
        print(f"系统: {msg.content}")
    elif isinstance(msg, HumanMessage):
        print(f"用户: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"AI: {msg.content}")
        # 检查是否有工具调用
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  工具调用: {tc['name']}")
    elif isinstance(msg, ToolMessage):
        print(f"工具结果: {msg.content}")
```

---

## 二十三、工具高级 API

### 23.1 bind_tools - 工具绑定

**用途**: 将工具直接绑定到模型，无需创建 Agent

**导入路径**: 内置方法

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天天气晴朗"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    ⚠️ 安全警告: eval() 在生产环境中有代码注入风险，仅用于演示。
    """
    # 仅用于演示目的，生产环境应使用更安全的实现
    return str(eval(expression))

model = init_chat_model("groq:llama-3.3-70b-versatile")
tools = [get_weather, calculate]

# 绑定工具到模型
model_with_tools = model.bind_tools(tools)

# 调用
response = model_with_tools.invoke("北京天气怎么样？")

# 检查是否需要调用工具
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具名: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
        print(f"ID: {tool_call['id']}")
```

#### bind_tools vs create_agent

| 方式 | 适用场景 | 特点 |
|------|----------|------|
| `bind_tools` | 简单场景、手动控制 | 模型返回工具调用，需手动执行 |
| `create_agent` | 复杂场景、自动循环 | 自动执行工具、循环直到完成 |

---

### 23.2 StructuredTool - 结构化工具

**用途**: 从函数创建具有完整元数据的工具

**导入路径**: `from langchain_core.tools import StructuredTool`

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 定义输入模型
class CalculatorInput(BaseModel):
    a: float = Field(description="第一个数字")
    b: float = Field(description="第二个数字")
    operation: str = Field(description="运算: add, subtract, multiply, divide")

# 定义函数
def calculator_func(input: CalculatorInput) -> str:
    ops = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "错误: 除数不能为零"
    }
    result = ops.get(input.operation, lambda a, b: "未知运算")(input.a, input.b)
    return f"结果: {result}"

# 创建结构化工具
calculator_tool = StructuredTool.from_function(
    func=calculator_func,
    name="calculator",
    description="执行数学运算",
    args_schema=CalculatorInput
)

# 使用
print(calculator_tool.invoke({"a": 10, "b": 5, "operation": "multiply"}))
# 输出: 结果: 50.0
```

---

### 23.3 ToolException - 工具异常

**用途**: 定义工具专用异常，配合 `handle_tool_error` 使用

**导入路径**: `from langchain_core.tools import ToolException`

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def search_database(query: str) -> str:
    """搜索数据库

    参数:
        query: 搜索关键词

    返回:
        搜索结果
    """
    # 参数验证
    if not query or len(query.strip()) == 0:
        raise ToolException("搜索关键词不能为空")

    # 模拟数据库连接失败
    if "error" in query.lower():
        raise ToolException("数据库连接失败，请稍后重试")

    return f"找到 3 条关于 '{query}' 的记录"


@tool(handle_tool_error="处理工具错误时发生异常")
def risky_tool(data: str) -> str:
    """可能失败的工具"""
    if not data:
        raise ToolException("数据无效")
    return f"处理成功: {data}"
```

#### handle_tool_error 参数

| 值 | 行为 |
|-----|------|
| `False` (默认) | 异常向上抛出 |
| `True` | 返回异常信息作为工具输出 |
| `str` | 返回指定的错误消息 |

---

### 23.4 BaseTool - 工具基类

**用途**: 创建完全自定义的工具类

**导入路径**: `from langchain_core.tools import BaseTool`

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")

class SearchTool(BaseTool):
    """自定义搜索工具"""

    name: str = "custom_search"
    description: str = "搜索信息"
    args_schema: Type[BaseModel] = SearchInput

    # 工具配置
    return_direct: bool = False  # 是否直接返回结果（不继续 Agent 循环）

    def _run(self, query: str, max_results: int = 5) -> str:
        """同步执行"""
        # 实现搜索逻辑
        results = [f"结果 {i+1}: {query}" for i in range(max_results)]
        return "\n".join(results)

    async def _arun(self, query: str, max_results: int = 5) -> str:
        """异步执行"""
        return self._run(query, max_results)

# 使用
search_tool = SearchTool()
print(search_tool.invoke({"query": "Python", "max_results": 3}))
```

---

## 二十四、配置管理 API

### 24.1 RunnableConfig - 运行配置

**用途**: 创建和管理运行时配置

**导入路径**: `from langchain_core.runnables import RunnableConfig`

```python
from langchain_core.runnables import RunnableConfig

# 创建配置
config = RunnableConfig(
    # 元数据 - 用于追踪和调试
    metadata={
        "user_id": "user_123",
        "session_id": "session_abc",
        "environment": "production"
    },

    # 标签 - 用于分类和过滤
    tags=["chat", "production", "v2"],

    # 可配置参数
    configurable={
        "thread_id": "conversation_001",
        "user_id": "user_123"
    },

    # 最大并发数
    max_concurrency=5
)

# 使用配置
response = chain.invoke("问题", config=config)
```

---

### 24.2 configurable 参数详解

**用途**: 传递可配置的运行时参数

```python
# thread_id - 会话管理
config = {
    "configurable": {
        "thread_id": "user_session_001"
    }
}

# 带检查点的调用
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config
)

# 后续调用会保持上下文
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我刚才说了什么？"}]},
    config=config  # 相同的 thread_id
)
```

---

### 24.3 metadata 和 tags 使用

```python
from langchain_core.runnables import RunnableConfig

# 用于 LangSmith 追踪的配置
config = RunnableConfig(
    metadata={
        "user_id": "user_123",
        "query_type": "rag",
        "model_version": "v2"
    },
    tags=["production", "rag", "customer_facing"]
)

response = chain.invoke(query, config=config)

# 在 LangSmith 中可以按 metadata 和 tags 过滤追踪记录
```

---

### 24.4 回调配置

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    pass

handler = MyHandler()

# 方式 1: 在 RunnableConfig 中配置
config = RunnableConfig(
    callbacks=[handler]
)

# 方式 2: 在字典中配置
config = {
    "callbacks": [handler]
}

response = chain.invoke(query, config=config)
```

---

## 二十五、状态管理 API

### 25.1 get_state - 获取状态

**用途**: 获取 LangGraph 应用的当前状态

```python
from langgraph.checkpoint.memory import MemorySaver

# 编译带检查点的图
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 配置 thread_id
config = {"configurable": {"thread_id": "session_001"}}

# 执行
app.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=config)

# 获取状态
state = app.get_state(config)

print(state.values)     # 状态值字典
print(state.next)       # 下一步要执行的节点
print(state.config)     # 配置信息
print(state.created_at) # 创建时间
print(state.parent_config)  # 父状态配置（用于时间旅行）
```

---

### 25.2 update_state - 更新状态

**用途**: 手动更新 LangGraph 应用状态

```python
# 手动更新状态
app.update_state(
    config,
    {"messages": [{"role": "assistant", "content": "手动添加的消息"}]}
)

# 验证更新
state = app.get_state(config)
print(state.values)
```

---

### 25.3 get_state_history - 状态历史

**用途**: 获取状态变更历史

```python
# 获取状态历史
history = list(app.get_state_history(config))

for i, state in enumerate(history):
    print(f"状态 {i}:")
    print(f"  值: {state.values}")
    print(f"  下一步: {state.next}")
    print()
```

---

### 25.4 完整状态管理示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int

def increment(state: State) -> dict:
    return {"count": state.get("count", 0) + 1}

# 构建图
graph = StateGraph(State)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_edge("increment", END)

# 编译
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 配置
config = {"configurable": {"thread_id": "demo"}}

# 多次执行
for i in range(3):
    result = app.invoke({"messages": [], "count": 0}, config=config)
    print(f"执行 {i+1}: count = {result['count']}")

# 检查状态
state = app.get_state(config)
print(f"最终 count: {state.values['count']}")

# 查看历史
history = list(app.get_state_history(config))
print(f"历史记录数: {len(history)}")
```

---

## 二十六、异步 API

### 26.1 异步调用方法

**用途**: 使用异步 API 提高并发性能

```python
import asyncio
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 异步调用
async def main():
    # 单次异步调用
    response = await model.ainvoke("你好")
    print(response.content)

asyncio.run(main())
```

---

### 26.2 异步流式输出

```python
async def stream_example():
    # 异步流式
    async for chunk in model.astream("写一首诗"):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_example())
```

---

### 26.3 异步批处理

```python
async def batch_example():
    queries = ["问题1", "问题2", "问题3"]

    # 批量处理
    results = await model.abatch(queries)

    for query, result in zip(queries, results):
        print(f"{query}: {result.content[:50]}...")

asyncio.run(batch_example())
```

---

### 26.4 astream_events - 流式事件

**用途**: 获取细粒度的执行事件流

```python
async def stream_events_example():
    # 获取详细事件流
    async for event in model.astream_events("写一个故事", version="v2"):
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            # 模型输出 token
            print(event["data"]["chunk"].content, end="")

        elif event_type == "on_chain_start":
            print(f"\n[Chain 开始] {event['name']}")

        elif event_type == "on_chain_end":
            print(f"\n[Chain 结束] {event['name']}")

asyncio.run(stream_events_example())
```

---

### 26.5 异步 API 速查表

| 同步方法 | 异步方法 | 说明 |
|---------|---------|------|
| `invoke()` | `ainvoke()` | 单次调用 |
| `stream()` | `astream()` | 流式输出 |
| `batch()` | `abatch()` | 批量处理 |
| - | `astream_events()` | 事件流 |

---

## 🔗 相关资源

- [LangChain 1.0 官方文档](https://docs.langchain.com/oss/python/langchain/)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/)
- [迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [API 参考文档](https://docs.langchain.com/oss/python/api_reference/)

---

**文档版本**: 3.0
**最后更新**: 2025-03-26
**新增内容**: 图可视化、中间件、回调系统、消息管理、工具高级API、配置管理、状态管理、异步API
**基于项目**: Langchain1.0-Langgraph1.0-Learning
