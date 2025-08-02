import asyncio
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


async def run_client(script: Path, env: dict[str, str]):
    transport = PythonStdioTransport(str(script), args=[], env=env, keep_alive=False)
    async with Client(
        transport, roots=[Path(env["JOURNAL_PATH"]).as_uri()]
    ) as client:
        result1 = await client.call_tool("search_summaries", {"query": "hello"})
        result2 = await client.call_tool(
            "search_transcripts", {"query": "hi", "day": "20240101"}
        )
        result3 = await client.call_tool("search_events", {"query": "meet"})
        resource = await client.call_tool(
            "get_resource", {"uri": "journal://summary/20240101/foo"}
        )
        media = await client.call_tool(
            "get_resource", {"uri": "journal://media/20240101/090000_audio.json"}
        )
    return (
        result1.data,
        result2.data,
        result3.data,
        resource.data,
        media.data,
    )


def test_mcp_tools_via_stdio(tmp_path):
    calls_file = tmp_path / "calls.txt"
    env = {"JOURNAL_PATH": str(tmp_path), "CALLS_FILE": str(calls_file)}

    day_dir = tmp_path / "20240101"
    topics_dir = day_dir / "topics"
    topics_dir.mkdir(parents=True)
    (topics_dir / "foo.md").write_text("first paragraph\n\nsecond", encoding="utf-8")

    heard = day_dir / "heard"
    heard.mkdir()
    (heard / "090000_audio.flac").write_bytes(b"data")
    (day_dir / "090000_audio.json").write_text("[]", encoding="utf-8")

    script = Path(__file__).with_name("run_mcp_stub.py")
    data1, data2, data3, text, blob = asyncio.run(run_client(script, env))
    # The resource returns raw text content, not JSON
    summary_text = text

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert "topics:hello:5:0" in calls
    assert "raws:hi:20240101:5:0" in calls
    assert any(c.startswith("events:meet") for c in calls)
    assert data1 == {
        "total": 1,
        "limit": 5,
        "offset": 0,
        "results": [{"day": "20240101", "topic": "foo.md", "text": "hello"}],
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
    assert summary_text == "first paragraph\n\nsecond"
    import base64

    assert base64.b64decode(blob) == b"data"
    assert data3 == {
        "total": 1,
        "limit": 5,
        "offset": 0,
        "results": [
            {
                "day": "20240101",
                "topic": "meetings",
                "start": "09:00",
                "end": "09:30",
                "title": "Standup",
                "summary": "Daily sync",
            }
        ],
    }
