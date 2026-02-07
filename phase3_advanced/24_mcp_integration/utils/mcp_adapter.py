"""
MCP to LangChain Adapter
========================

将 MCP 工具转换为 LangChain 工具的适配器
支持在 LangChain Agent 中使用 MCP 服务器的工具
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Any, Optional
from langchain_core.tools import StructuredTool
from langchain_core.tools import BaseTool

# MCP 客户端导入
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("警告: 未安装 mcp 包，请运行: pip install mcp")


class MCPToolAdapter:
    """MCP 工具适配器 - 连接 MCP 服务器并转换为 LangChain 工具"""

    def __init__(self, server_script: str):
        """
        初始化适配器

        Args:
            server_script: MCP 服务器脚本路径
        """
        if not MCP_AVAILABLE:
            raise ImportError("请先安装 mcp 包: pip install mcp")

        self.server_script = Path(server_script)
        if not self.server_script.exists():
            raise FileNotFoundError(f"MCP 服务器脚本不存在: {server_script}")

        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)]
        )

    async def get_langchain_tools(self) -> List[BaseTool]:
        """
        获取转换为 LangChain 格式的工具列表

        Returns:
            LangChain 工具列表
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化连接
                await session.initialize()

                # 获取工具列表
                tools_response = await session.list_tools()

                # 转换为 LangChain 工具
                langchain_tools = []
                for mcp_tool in tools_response.tools:
                    langchain_tool = self._convert_tool(
                        mcp_tool,
                        session
                    )
                    langchain_tools.append(langchain_tool)

                return langchain_tools

    def _convert_tool(self, mcp_tool, session: ClientSession) -> BaseTool:
        """将单个 MCP 工具转换为 LangChain 工具"""

        async def tool_function(**kwargs):
            # 调用 MCP 工具
            result = await session.call_tool(
                mcp_tool.name,
                arguments=kwargs
            )

            # 提取文本内容
            if result.content:
                return result.content[0].text
            return str(result)

        # 创建 LangChain 工具
        return StructuredTool.from_function(
            func=tool_function,
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP 工具: {mcp_tool.name}",
        )

    async def get_tools_sync(self) -> List[BaseTool]:
        """同步方法获取工具（使用 asyncio.run）"""
        return await self.get_langchain_tools()


def create_mcp_tools(server_script: str) -> List[BaseTool]:
    """
    便捷函数：创建 MCP 工具列表

    Args:
        server_script: MCP 服务器脚本路径

    Returns:
        LangChain 工具列表

    示例:
        >>> tools = create_mcp_tools("servers/filesystem_server.py")
        >>> agent = create_agent(model=model, tools=tools)
    """
    adapter = MCPToolAdapter(server_script)

    # 运行异步方法
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(adapter.get_langchain_tools())


async def create_mcp_tools_async(server_script: str) -> List[BaseTool]:
    """
    异步版本：创建 MCP 工具列表

    Args:
        server_script: MCP 服务器脚本路径

    Returns:
        LangChain 工具列表

    示例:
        >>> tools = await create_mcp_tools_async("servers/filesystem_server.py")
    """
    adapter = MCPToolAdapter(server_script)
    return await adapter.get_langchain_tools()


# 便捷函数：同时获取多个 MCP 服务器的工具
def create_multi_mcp_tools(server_scripts: List[str]) -> List[BaseTool]:
    """
    从多个 MCP 服务器创建工具列表

    Args:
        server_scripts: MCP 服务器脚本路径列表

    Returns:
        所有服务器的 LangChain 工具列表

    示例:
        >>> server_scripts = [
        ...     "servers/filesystem_server.py",
        ...     "servers/search_server.py"
        ... ]
        >>> tools = create_multi_mcp_tools(server_scripts)
    """
    all_tools = []
    for script in server_scripts:
        try:
            tools = create_mcp_tools(script)
            all_tools.extend(tools)
        except Exception as e:
            print(f"警告: 无法加载 {script}: {e}")
    return all_tools
