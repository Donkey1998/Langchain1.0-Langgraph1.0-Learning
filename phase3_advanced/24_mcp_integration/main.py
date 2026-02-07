"""
模块 24：MCP 集成
==================

本模块演示如何：
1. 创建 MCP (Model Context Protocol) 服务器
2. 实现 Filesystem MCP - 文件系统访问
3. 实现 Web Search MCP - 网络搜索
4. 在 LangChain Agent 中使用 MCP 工具

MCP 是连接 AI 模型与外部数据源的开放标准协议

作者: LangChain 1.0 & LangGraph 1.0 学习项目
"""

import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径（用于导入本地模块）
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

# 初始化模型
print("正在初始化模型...")
model = init_chat_model(
    model=MODEL,
    model_provider=MODEL_PROVIDER,
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================================
# 示例 1: Filesystem MCP - 文件系统访问
# ============================================================================

def example_1_filesystem_mcp():
    """
    示例 1: Filesystem MCP - 文件系统访问

    核心概念：
    - FastMCP: 快速构建 MCP 服务器的装饰器框架
    - @mcp.tool(): 定义可被 LLM 调用的工具函数
    - 直接调用 MCP 工具进行文件操作
    """
    print("\n" + "="*70)
    print("示例 1: Filesystem MCP - 文件系统访问")
    print("="*70)

    # 导入 MCP 适配器
    from utils.mcp_adapter import create_mcp_tools

    # 获取 Filesystem MCP 工具
    server_script = str(current_dir / "servers" / "filesystem_server.py")
    print(f"\n正在连接 Filesystem MCP 服务器: {server_script}")

    try:
        tools = create_mcp_tools(server_script)
        print(f"成功加载 {len(tools)} 个文件系统工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # 测试工具
        print("\n" + "-"*70)
        print("测试 1: 列出当前目录文件")
        print("-"*70)
        list_files_tool = next(t for t in tools if t.name == "list_files")
        result = list_files_tool.invoke({"directory": str(current_dir)})
        print(result[:500])

        print("\n" + "-"*70)
        print("测试 2: 读取 README 文件")
        print("-"*70)
        read_file_tool = next(t for t in tools if t.name == "read_file")
        readme_path = str(current_dir / "README.md")
        if Path(readme_path).exists():
            result = read_file_tool.invoke({"file_path": readme_path})
            print(result[:500])
        else:
            print("README.md 文件不存在（将在后面创建）")

        print("\n" + "-"*70)
        print("测试 3: 写入测试文件")
        print("-"*70)
        write_file_tool = next(t for t in tools if t.name == "write_file")
        test_file = str(current_dir / "test_mcp.txt")
        result = write_file_tool.invoke({
            "file_path": test_file,
            "content": "这是 MCP 写入的测试文件\n创建时间: 2026-02-02"
        })
        print(result)

        # 清理测试文件
        if Path(test_file).exists():
            Path(test_file).unlink()
            print("已清理测试文件")

        print("\n✅ Filesystem MCP 测试完成！")

    except ImportError as e:
        print(f"\n错误: {e}")
        print("请确保已安装 mcp 包: pip install mcp")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例 2: Web Search MCP - 网络搜索
# ============================================================================

def example_2_web_search_mcp():
    """
    示例 2: Web Search MCP - 网络搜索

    核心概念：
    - DuckDuckGo 搜索 API（免费，无需 API Key）
    - 网络搜索工具的实现
    - 搜索结果格式化
    """
    print("\n" + "="*70)
    print("示例 2: Web Search MCP - 网络搜索")
    print("="*70)

    # 导入 MCP 适配器
    from utils.mcp_adapter import create_mcp_tools

    # 获取 Web Search MCP 工具
    server_script = str(current_dir / "servers" / "search_server.py")
    print(f"\n正在连接 Web Search MCP 服务器: {server_script}")

    try:
        tools = create_mcp_tools(server_script)
        print(f"成功加载 {len(tools)} 个搜索工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # 测试搜索工具
        print("\n" + "-"*70)
        print("测试 1: 网络搜索 'LangChain MCP 集成'")
        print("-"*70)
        search_web_tool = next(t for t in tools if t.name == "search_web")
        result = search_web_tool.invoke({
            "query": "LangChain MCP 集成教程",
            "max_results": 3
        })
        print(result[:800])

        print("\n" + "-"*70)
        print("测试 2: 新闻搜索 '人工智能最新进展'")
        print("-"*70)
        search_news_tool = next((t for t in tools if t.name == "search_news"), None)
        if search_news_tool:
            result = search_news_tool.invoke({
                "query": "人工智能最新进展",
                "max_results": 2
            })
            print(result[:800])
        else:
            print("新闻搜索工具未找到")

        print("\n✅ Web Search MCP 测试完成！")

    except ImportError as e:
        print(f"\n错误: {e}")
        print("请确保已安装 mcp 和 duckduckgo-search 包:")
        print("  pip install mcp duckduckgo-search")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例 3: LangChain Agent + MCP 工具集成
# ============================================================================

def example_3_langchain_agent_with_mcp():
    """
    示例 3: LangChain Agent 集成 MCP 工具

    核心概念：
    - 使用 create_agent 创建 Agent（LangChain 1.0 API）
    - 将 MCP 工具转换为 LangChain 工具
    - Agent 自动决定何时使用哪个工具
    """
    print("\n" + "="*70)
    print("示例 3: LangChain Agent + MCP 工具集成")
    print("="*70)

    # 导入 MCP 适配器
    from utils.mcp_adapter import create_multi_mcp_tools

    print("\n正在加载 MCP 工具...")

    try:
        # 同时加载多个 MCP 服务器的工具
        server_scripts = [
            str(current_dir / "servers" / "filesystem_server.py"),
            str(current_dir / "servers" / "search_server.py")
        ]

        tools = create_multi_mcp_tools(server_scripts)
        print(f"成功加载 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # 创建 Agent（使用 LangChain 1.0 API）
        print("\n正在创建 Agent...")
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="""你是一个智能助手，可以使用文件系统和网络搜索工具。

可用的工具：
- read_file: 读取文件内容
- list_files: 列出目录中的文件
- write_file: 写入文件
- search_web: 网络搜索
- search_news: 新闻搜索

请根据用户的问题选择合适的工具。"""
        )

        print("Agent 创建成功！")

        # 测试场景 1: 读取文件
        print("\n" + "-"*70)
        print("场景 1: 读取并分析项目 README")
        print("-"*70)

        # 先创建一个测试用的 README 文件
        readme_content = """# LangChain MCP 集成示例

这是一个学习项目，演示如何使用 MCP (Model Context Protocol)。

## 功能特点
- 文件系统访问
- 网络搜索
- LangChain Agent 集成

## 学习目标
掌握 MCP 协议的基本使用方法
"""
        readme_path = current_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')

        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "请读取当前目录的 README.md 文件，并总结其内容。"
            }]
        })

        print(f"\nAgent 回复:\n{response['messages'][-1].content}")

        # 测试场景 2: 网络搜索
        print("\n" + "-"*70)
        print("场景 2: 网络搜索补充信息")
        print("-"*70)

        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "请搜索 'MCP Model Context Protocol' 的最新信息，并简要介绍。"
            }]
        })

        print(f"\nAgent 回复:\n{response['messages'][-1].content}")

        print("\n✅ Agent 集成测试完成！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例 4: 完整工作流 - 多工具协同
# ============================================================================

def example_4_complete_workflow():
    """
    示例 4: 完整工作流 - 多 MCP 工具协同

    场景：Agent 读取配置文件，搜索网络获取最新信息，生成报告
    """
    print("\n" + "="*70)
    print("示例 4: 完整工作流 - 多 MCP 工具协同")
    print("="*70)

    # 导入 MCP 适配器
    from utils.mcp_adapter import create_multi_mcp_tools

    print("\n场景：AI 助手分析技术栈并生成学习建议")
    print("-"*70)

    try:
        # 创建测试数据
        config_file = current_dir / "tech_stack.txt"
        config_content = """技术栈配置
==========
当前技术栈：
- Python 3.10+
- LangChain 1.0
- LangGraph 1.0
- OpenAI GPT-4o

学习目标：
1. 掌握 LangChain 基础
2. 学习 Agent 开发
3. 集成外部工具
"""
        config_file.write_text(config_content, encoding='utf-8')

        # 加载 MCP 工具
        server_scripts = [
            str(current_dir / "servers" / "filesystem_server.py"),
            str(current_dir / "servers" / "search_server.py")
        ]

        tools = create_multi_mcp_tools(server_scripts)

        # 创建 Agent
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="""你是一个技术顾问助手。

任务：
1. 读取配置文件了解当前技术栈
2. 搜索网络获取最新信息
3. 生成学习建议报告

请使用可用的工具完成任务。"""
        )

        print("\n正在执行工作流...")

        # 执行完整工作流
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": """请执行以下任务：
1. 读取 tech_stack.txt 文件，了解我的技术栈
2. 搜索 'LangChain 1.0 新功能' 获取最新信息
3. 结合文件内容和搜索结果，为我生成一份学习建议报告
"""
            }]
        })

        print(f"\n工作流执行结果:")
        print("="*70)
        print(response['messages'][-1].content)

        # 清理测试文件
        if config_file.exists():
            config_file.unlink()
            print("\n已清理测试文件")

        print("\n✅ 完整工作流测试完成！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    print("\n" + "="*70)
    print(" LangChain 1.0 - MCP 集成 (Model Context Protocol)")
    print("="*70)
    print("\nMCP 是连接 AI 模型与外部数据源的开放标准协议")
    print("本模块演示如何创建和使用 MCP 服务器\n")

    try:
        # 示例 1: Filesystem MCP
        example_1_filesystem_mcp()
        input("\n按 Enter 继续下一个示例...")

        # 示例 2: Web Search MCP
        example_2_web_search_mcp()
        input("\n按 Enter 继续下一个示例...")

        # 示例 3: LangChain Agent 集成
        example_3_langchain_agent_with_mcp()
        input("\n按 Enter 继续下一个示例...")

        # 示例 4: 完整工作流
        example_4_complete_workflow()

        # 总结
        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. MCP 是连接 AI 与外部系统的标准协议")
        print("  2. FastMCP 提供装饰器模式快速构建服务器")
        print("  3. @mcp.tool() 定义可被 LLM 调用的工具")
        print("  4. @mcp.resource() 定义只读数据源")
        print("  5. LangChain 可以无缝集成 MCP 工具")
        print("\n下一步：")
        print("  - 探索更多 MCP 服务器示例")
        print("  - 学习构建自定义 MCP 适配器")
        print("  - 查看 [MCP 官方文档](https://modelcontextprotocol.io/)")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
