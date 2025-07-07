import asyncio
import os

_client = None


_SERVER_PATH = os.path.join(os.path.dirname(__file__), "sunstone_server.py")
_SERVER_URL = os.getenv("SUNSTONE_MCP_URL")


def _get_client():
    global _client
    if _client is None:
        from fastmcp import Client

        _client = Client(_SERVER_URL or _SERVER_PATH, keep_alive=True)
    return _client


async def sunstone_toolset():
    """Return a Tool object ready for Gemini/Chat."""
    client = _get_client()
    await client.initialize()
    return client.as_toolset(label="Sunstone")


async def close_client():
    if _client is not None:
        await _client.close()
