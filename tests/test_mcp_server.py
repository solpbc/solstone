import asyncio
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


async def run_client(script: Path, env: dict[str, str]):
    transport = PythonStdioTransport(str(script), args=[], env=env, keep_alive=False)
    async with Client(transport) as client:
        result1 = await client.call_tool("search_topic", {"query": "hello"})
        result2 = await client.call_tool("search_occurrence", {"query": "hi"})
    return result1.data, result2.data


def test_mcp_server_via_stdio(tmp_path):
    calls_file = tmp_path / "calls.txt"
    env = {"JOURNAL_PATH": str(tmp_path), "CALLS_FILE": str(calls_file)}
    script = Path(__file__).with_name("run_mcp_stub.py")
    data1, data2 = asyncio.run(run_client(script, env))

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert "topics:hello:5:0" in calls
    assert "occurrences:hi:5" in calls
    assert data1 == {
        "total": 1,
        "results": [{"day": "20240101", "filename": "foo", "text": "hello"}],
    }
    assert data2 == "20240101 note: occurred"
