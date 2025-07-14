import os

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")
_SERVER_URL = os.getenv("SUNSTONE_MCP_URL")


def get_sunstone_client():
    """Return a new FastMCP Client for Sunstone tools."""
    from fastmcp import Client
    from fastmcp.client.transports import PythonStdioTransport

    if _SERVER_URL:
        server_source = _SERVER_URL
        return Client(server_source)

    env = os.environ.copy()
    transport = PythonStdioTransport(_SERVER_PATH, env=env)
    return Client(transport)


async def close_client():
    """Close the MCP client. Note: With the new FastMCP API, connections are managed by async context managers."""
    # No explicit close is needed with the new FastMCP API
    # Connections are managed by async context managers
    pass
