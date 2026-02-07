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
- [七、RAG 相关 API](#七rag-相关-api)
- [八、LangGraph 核心 API](#八langgraph-核心-api)
- [九、向量数据库 API](#九向量数据库-api)
- [十、文档处理 API](#十文档处理-api)
- [十一、状态持久化 API](#十一状态持久化-api)
- [十二、外部服务集成](#十二外部服务集成)

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
    """计算数学表达式"""
    try:
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

## 七、RAG 相关 API

### 7.1 文档加载器

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

### 7.2 文本分割器

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

### 7.3 向量嵌入

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

### 7.4 向量存储

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

### 7.5 向量检索

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

### 7.6 BM25 检索器

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

### 7.7 混合检索器

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

### 7.8 RAG 链构建

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

### 7.9 RAG 高级特性

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

### 7.10 RAG 最佳实践

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

## 八、LangGraph 核心 API

### 8.1 StateGraph - 状态图

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

### 8.2 状态持久化

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

## 九、向量数据库 API

### 9.1 Pinecone 集成

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

### 9.2 Chroma 集成

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

## 十、文档处理 API

### 10.1 PDF 处理

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

### 10.2 Word 文档

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
docs = loader.load()
```

### 10.3 Markdown

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("document.md")
docs = loader.load()
```

---

## 十一、状态持久化 API

### 11.1 MemorySaver

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

### 11.2 SQLite 检查点

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建 SQLite 检查点
conn = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=conn)
```

---

## 十二、外部服务集成

### 12.1 LangSmith 可观测性

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key"

# 自动追踪所有 LangChain 调用
response = model.invoke("Hello")
```

### 12.2 Groq API

```python
# 环境变量
export GROQ_API_KEY="gsk_..."

# 使用
from langchain.chat_models import init_chat_model
model = init_chat_model("groq:llama-3.3-70b-versatile")
```

### 12.3 OpenAI API

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
# 模型
from langchain.chat_models import init_chat_model

# 提示词
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 消息
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 工具
from langchain_core.tools import tool

# Agent
from langchain.agents import create_agent

# 结构化输出
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# RAG - 文档加载
from langchain_community.document_loaders import (
    TextLoader,           # 文本文件
    PyPDFLoader,          # PDF 文件
    WebBaseLoader,        # 网页
    CSVLoader,            # CSV 文件
    DirectoryLoader,      # 目录批量加载
    Docx2txtLoader,       # Word 文档
    UnstructuredMarkdownLoader  # Markdown
)

# RAG - 文本分割
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 递归分割（推荐）
    CharacterTextSplitter,           # 字符分割
    TokenTextSplitter,               # Token 分割
    PythonCodeTextSplitter,          # Python 代码
    MarkdownTextSplitter             # Markdown
)

# RAG - 向量嵌入
from langchain_huggingface import HuggingFaceEmbeddings  # 免费 HuggingFace
from langchain_openai import OpenAIEmbeddings            # OpenAI（付费）

# RAG - 向量存储
from langchain_community.vectorstores import Chroma      # 本地 Chroma
from langchain_core.vectorstores import InMemoryVectorStore  # 内存存储
from langchain_pinecone import PineconeVectorStore       # Pinecone 云

# RAG - 检索器
from langchain_community.retrievers import BM25Retriever  # BM25 关键词检索
from langchain_classic.retrievers import EnsembleRetriever  # 混合检索

# RAG - 核心
from langchain_core.documents import Document  # 文档对象

# 持久化
from langgraph.checkpoint.memory import MemorySaver
```

---

## 🔗 相关资源

- [LangChain 1.0 官方文档](https://docs.langchain.com/oss/python/langchain/)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/)
- [迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [API 参考文档](https://docs.langchain.com/oss/python/api_reference/)

---

**文档版本**: 1.0
**最后更新**: 2026-01-30
**基于项目**: Langchain1.0-Langgraph1.0-Learning
