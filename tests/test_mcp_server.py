import asyncio
import json
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


async def run_client(script: Path, env: dict[str, str]):
    transport = PythonStdioTransport(str(script), args=[], env=env, keep_alive=False)
    async with Client(transport) as client:
        result1 = await client.call_tool("search_topic", {"query": "hello"})
        result2 = await client.call_tool(
            "search_raw", {"query": "hi", "day": "20240101"}
        )
        resource = await client.read_resource("journal://summary/20240101/foo")
    return result1.data, result2.data, resource[0].text


def test_mcp_server_via_stdio(tmp_path):
    calls_file = tmp_path / "calls.txt"
    env = {"JOURNAL_PATH": str(tmp_path), "CALLS_FILE": str(calls_file)}

    day_dir = tmp_path / "20240101" / "topics"
    day_dir.mkdir(parents=True)
    (day_dir / "foo.md").write_text("first paragraph\n\nsecond", encoding="utf-8")

    script = Path(__file__).with_name("run_mcp_stub.py")
    data1, data2, text = asyncio.run(run_client(script, env))
    summary = json.loads(text)

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert "topics:hello:5:0" in calls
    assert "raws:hi:20240101:5:0" in calls
    assert data1 == {
        "total": 1,
        "limit": 5,
        "offset": 0,
        "results": [{"day": "20240101", "filename": "foo", "text": "hello"}],
    }
    assert data2 == {
        "total": 1,
        "limit": 5,
        "offset": 0,
        "results": [
            {
                "day": "20240101",
                "time": "123000",
                "type": "audio",
                "text": "occurred",
            }
        ],
    }
    assert summary["day"] == "20240101"
    assert summary["topic"] == "foo"
    assert summary["summary"] == "first paragraph\n\nsecond"
