# 模块 24：MCP 集成

## 🎯 学习目标

本模块将帮助你掌握 MCP (Model Context Protocol) 的使用方法，学会：
1. 创建简单的 MCP 服务器
2. 实现 Filesystem MCP - 文件系统访问
3. 实现 Web Search MCP - 网络搜索
4. 在 LangChain Agent 中集成 MCP 工具

## 📚 核心概念

### 什么是 MCP？

MCP (Model Context Protocol) 是一个开放标准协议，用于标准化 AI 模型与外部数据源的交互。它就像是"AI 世界的 USB-C 接口"：

- **标准化接口**：统一的工具和资源定义方式
- **语言无关**：支持 Python、JavaScript、Rust 等多种语言
- **双向通信**：支持客户端-服务器架构
- **安全性**：提供身份验证和权限控制

### MCP 核心组件

#### 1. Tools (工具)
工具是 LLM 可以调用的函数，具有副作用或计算能力：

```python
@mcp.tool()
def calculate(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b
```

#### 2. Resources (资源)
资源是只读数据，类似于 GET 端点：

```python
@mcp.resource("config://settings")
def get_settings() -> str:
    """获取配置"""
    return '{"theme": "dark"}'
```

#### 3. Prompts (提示词)
提示词是可重用的模板：

```python
@mcp.prompt()
def review_code(code: str) -> str:
    return f"请审查以下代码：\n{code}"
```

## 🔑 关键 API

### FastMCP

FastMCP 是快速构建 MCP 服务器的装饰器框架（类似 FastAPI）：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def my_tool(param: str) -> str:
    """工具描述"""
    return "result"

if __name__ == "__main__":
    mcp.run()  # 启动服务器
```

### LangChain 集成

将 MCP 工具转换为 LangChain 工具：

```python
from utils.mcp_adapter import create_mcp_tools

# 获取 MCP 工具
tools = create_mcp_tools("servers/filesystem_server.py")

# 创建 Agent（LangChain 1.0 API）
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个文件管理助手"
)
```

## 📝 本模块示例

### main.py

包含四个递进的示例：

1. **示例 1: Filesystem MCP** - 创建文件系统访问服务器
   - 读取文件内容
   - 列出目录文件
   - 写入文件

2. **示例 2: Web Search MCP** - 创建网络搜索服务器
   - 网络搜索（使用 DuckDuckGo）
   - 新闻搜索

3. **示例 3: LangChain 集成** - 在 Agent 中使用 MCP 工具
   - 将 MCP 工具转换为 LangChain 工具
   - Agent 自动选择工具

4. **示例 4: 完整工作流** - 多 MCP 服务协同
   - 读取配置文件
   - 搜索网络信息
   - 生成综合报告

### 服务器实现

- `servers/filesystem_server.py` - 文件系统 MCP 服务器
- `servers/search_server.py` - 网络搜索 MCP 服务器

### 适配器

- `utils/mcp_adapter.py` - MCP 工具转 LangChain 工具

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install mcp duckduckgo-search
```

### 2. 配置环境变量

确保 `.env` 文件包含必要的配置：

```env
API_KEY=your_api_key
BASE_URL=your_base_url
MODEL=your_model_name
MODEL_PROVIDER=your_provider
```

### 3. 运行示例

```bash
cd phase3_advanced/24_mcp_integration
python main.py
```

## 📖 详细说明

### Filesystem MCP

文件系统 MCP 服务器提供了安全的文件操作能力：

```python
@mcp.tool()
def read_file(file_path: str) -> str:
    """读取文件内容（带安全检查）"""
    # 限制文件大小 1MB
    # 返回文件内容
```

**安全特性：**
- 文件大小限制（1MB）
- 路径验证
- 异常处理

### Web Search MCP

网络搜索 MCP 服务器使用 DuckDuckGo API：

```python
@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """网络搜索"""
    # 使用 DuckDuckGo 搜索
    # 返回格式化结果
```

**优势：**
- 无需 API Key
- 完全免费
- 结果丰富

### MCP 工具适配器

适配器负责将 MCP 工具转换为 LangChain 格式：

```python
class MCPToolAdapter:
    """连接 MCP 服务器并转换工具"""

    async def get_langchain_tools(self):
        """获取 LangChain 格式的工具"""
        # 连接 MCP 服务器
        # 转换工具格式
        # 返回工具列表
```

## 🧪 练习

1. **基础练习**：
   - 修改 `filesystem_server.py` 添加文件复制功能
   - 在 `search_server.py` 中添加图片搜索功能

2. **进阶练习**：
   - 创建一个新的 MCP 服务器（如天气、计算器等）
   - 实现一个同时使用多个 MCP 服务的 Agent

3. **挑战练习**：
   - 实现 MCP 资源（`@mcp.resource`）
   - 实现 MCP 提示词（`@mcp.prompt`）
   - 构建一个完整的 MCP 生态系统

## 📚 延伸阅读

### 官方文档
- [MCP 官方规范](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)

### 中文资源
- [LangChain 接入 MCP 实战指南](https://www.ningto.com/blog/2026/2026-01-22-langchain-%E6%8E%A5%E5%85%A5-mcp-%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97)
- [Model Context Protocol (MCP) Python SDK 权威指南](https://blog.csdn.net/qiwsir/article/details/156709461)

### 社区资源
- [轻松将 LangChain 代理连接到 Bright Data Web MCP](https://www.bright.cn/blog/ai/langchain-mcp-adapters-with-web-mcp)

## ⚠️ 注意事项

1. **安全性**：
   - MCP 工具可以执行代码，务必验证输入
   - 限制文件系统访问范围
   - 避免路径遍历攻击

2. **性能**：
   - 大文件操作会增加延迟
   - 网络搜索有 API 限速
   - 考虑缓存机制

3. **错误处理**：
   - 工具内部捕获异常
   - 返回友好的错误消息
   - 记录错误日志

4. **依赖管理**：
   - `mcp` 包版本 >= 1.7.1
   - `duckduckgo-search` 最新版本
   - 使用虚拟环境隔离依赖

## 🔧 故障排除

### 问题 1: ImportError: No module named 'mcp'

**解决方案：**
```bash
pip install mcp>=1.7.1
```

### 问题 2: DuckDuckGo 搜索失败

**解决方案：**
```bash
pip install --upgrade duckduckgo-search
```

### 问题 3: Agent 不调用工具

**可能原因：**
- 系统提示词不够明确
- 工具描述不清楚
- 模型能力不足

**解决方案：**
- 优化 system_prompt
- 改进工具描述
- 使用更强大的模型

## 💡 最佳实践

1. **工具命名**：使用清晰、描述性的名称
2. **文档字符串**：详细说明工具功能和参数
3. **错误处理**：捕获并友好地报告错误
4. **参数验证**：验证输入参数的有效性
5. **资源限制**：限制文件大小、搜索结果等

## 🎓 学习路径

完成本模块后，建议继续学习：

1. **模块 16**: LangGraph 基础
2. **模块 17**: 多智能体系统
3. **模块 18**: 条件路由
4. **综合项目**: 构建完整的 MCP 应用

## 📄 许可

本模块是 LangChain 1.0 & LangGraph 1.0 学习项目的一部分。

---

**祝你学习愉快！** 🎉
