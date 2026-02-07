"""
Filesystem MCP Server
====================

提供文件系统访问能力的 MCP 服务器
支持读取文件、列出目录、写入文件等操作
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os

# 初始化 MCP 服务器
mcp = FastMCP("FilesystemServer")


@mcp.tool()
def read_file(file_path: str) -> str:
    """读取文件内容

    Args:
        file_path: 文件路径（相对于当前工作目录或绝对路径）

    Returns:
        文件内容字符串
    """
    try:
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            return f"错误：文件不存在 - {file_path}"

        # 检查是否为文件
        if not path.is_file():
            return f"错误：不是文件 - {file_path}"

        # 安全检查：限制文件大小（1MB）
        file_size = path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB
            return f"错误：文件过大 ({file_size} bytes)，超过 1MB 限制"

        # 读取文件内容
        content = path.read_text(encoding='utf-8')
        return f"文件内容 ({file_path}):\n{content}"

    except Exception as e:
        return f"读取文件失败: {str(e)}"


@mcp.tool()
def list_files(directory: str = ".") -> str:
    """列出目录中的文件和子目录

    Args:
        directory: 目录路径（默认当前目录）

    Returns:
        文件和子目录列表
    """
    try:
        path = Path(directory)

        # 检查目录是否存在
        if not path.exists():
            return f"错误：目录不存在 - {directory}"

        # 检查是否为目录
        if not path.is_dir():
            return f"错误：不是目录 - {directory}"

        # 列出目录内容
        items = []
        for item in path.iterdir():
            item_type = "📁 目录" if item.is_dir() else "📄 文件"
            size = item.stat().st_size if item.is_file() else 0
            items.append(f"  [{item_type}] {item.name} ({size} bytes)")

        result = f"目录内容 ({directory}):\n" + "\n".join(items)
        return result

    except Exception as e:
        return f"列出文件失败: {str(e)}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """写入文件内容

    Args:
        file_path: 文件路径
        content: 要写入的内容

    Returns:
        操作结果
    """
    try:
        path = Path(file_path)

        # 创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        path.write_text(content, encoding='utf-8')

        return f"成功：写入文件 {file_path} ({len(content)} 字符)"

    except Exception as e:
        return f"写入文件失败: {str(e)}"


@mcp.resource("file://{path:path}")
def get_file_resource(path: str) -> str:
    """获取文件资源（只读）

    Args:
        path: 文件路径

    Returns:
        文件内容
    """
    return read_file(path)


if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()
