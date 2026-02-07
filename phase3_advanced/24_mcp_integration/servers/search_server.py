"""
Web Search MCP Server
====================

提供网络搜索能力的 MCP 服务器
使用 DuckDuckGo（无需 API Key，免费使用）
"""

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

# 初始化 MCP 服务器
mcp = FastMCP("SearchServer")


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """在网络上搜索信息

    Args:
        query: 搜索查询关键词
        max_results: 最大结果数量（默认 5，最大 10）

    Returns:
        搜索结果摘要
    """
    try:
        # 限制最大结果数
        max_results = min(max_results, 10)

        # 使用 DuckDuckGo 搜索
        ddgs = DDGS()
        results = list(ddgs.text(
            query,
            max_results=max_results
        ))

        if not results:
            return f"未找到 '{query}' 的搜索结果"

        # 格式化结果
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'N/A')
            url = result.get('link', 'N/A')
            body = result.get('body', 'N/A')[:200]

            formatted.append(f"""
{i}. {title}
   URL: {url}
   摘要: {body}...
""")

        return f"🔍 搜索结果 ({query}):\n" + "\n".join(formatted)

    except Exception as e:
        return f"搜索失败: {str(e)}"


@mcp.tool()
def search_news(query: str, max_results: int = 5) -> str:
    """搜索新闻

    Args:
        query: 新闻查询关键词
        max_results: 最大结果数量（默认 5，最大 10）

    Returns:
        新闻结果摘要
    """
    try:
        # 限制最大结果数
        max_results = min(max_results, 10)

        # 使用 DuckDuckGo 新闻搜索
        ddgs = DDGS()
        results = list(ddgs.news(
            query,
            max_results=max_results
        ))

        if not results:
            return f"未找到 '{query}' 的新闻"

        # 格式化结果
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'N/A')
            url = result.get('url', 'N/A')
            date = result.get('date', 'N/A')
            source = result.get('source', 'N/A')
            body = result.get('body', 'N/A')[:200]

            formatted.append(f"""
{i}. {title}
   来源: {source} | 日期: {date}
   URL: {url}
   摘要: {body}...
""")

        return f"📰 新闻结果 ({query}):\n" + "\n".join(formatted)

    except Exception as e:
        return f"新闻搜索失败: {str(e)}"


if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run()
