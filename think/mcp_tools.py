import asyncio
import os

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")
_SERVER_URL = os.getenv("SUNSTONE_MCP_URL")


def get_sunstone_client():
    """Return a new FastMCP Client for Sunstone tools."""
    from fastmcp import Client

    if _SERVER_URL:
        server_source = _SERVER_URL
    else:
        # Use the absolute path to the server script
        server_source = _SERVER_PATH
    return Client(server_source)


async def close_client():
    """Close the MCP client. Note: With the new FastMCP API, connections are managed by async context managers."""
    # No explicit close is needed with the new FastMCP API
    # Connections are managed by async context managers
    pass
