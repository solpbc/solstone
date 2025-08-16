import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


async def run_main(mod, argv):
    sys.argv = argv
    await mod.main_async()


def _setup_openai_mocks(monkeypatch, tmp_path):
    """Setup common mocks for OpenAI tests."""
    agents_stub = types.ModuleType("agents")
    agents_mcp_stub = types.ModuleType("agents.mcp")
    agents_mcp_server_stub = types.ModuleType("agents.mcp.server")
    agents_items_stub = types.ModuleType("agents.items")
    agents_run_stub = types.ModuleType("agents.run")
    agents_model_settings_stub = types.ModuleType("agents.model_settings")

    last_kwargs = {}

    class DummyAgent:
        def __init__(self, *a, **k):
            last_kwargs.update(k)

    class StreamResult:
        def __init__(self, events=None):
            self.events = events or []
            self.final_output = "ok"

        async def stream_events(self):
            for event in self.events:
                yield event

    class DummyRunner:
        called = False
        events_to_stream = []

        @staticmethod
        def run_streamed(agent, input, session=None, run_config=None, max_turns=None):
            DummyRunner.called = True
            return StreamResult(DummyRunner.events_to_stream)

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_run_stub.RunConfig = lambda **k: SimpleNamespace(**k)
    agents_model_settings_stub.ModelSettings = lambda **k: SimpleNamespace(**k)
    agents_stub.RunConfig = agents_run_stub.RunConfig
    agents_stub.ModelSettings = agents_model_settings_stub.ModelSettings
    agents_stub.set_default_openai_key = lambda k: None  # Add missing mock

    class DummySession:
        def __init__(self, *a, **k):
            pass

    agents_stub.SQLiteSession = DummySession

    class DummyMCPServer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_server_stub.MCPServerStreamableHttp = lambda **k: DummyMCPServer()

    # Item classes for streaming events
    agents_items_stub.MessageOutputItem = SimpleNamespace
    agents_items_stub.ToolCallItem = SimpleNamespace
    agents_items_stub.ToolCallOutputItem = SimpleNamespace
    agents_items_stub.ReasoningItem = SimpleNamespace

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules["agents.mcp.server"] = agents_mcp_server_stub
    sys.modules["agents.items"] = agents_items_stub
    sys.modules["agents.run"] = agents_run_stub
    sys.modules["agents.model_settings"] = agents_model_settings_stub
    sys.modules.pop("think.openai", None)

    return last_kwargs, DummyRunner


def test_openai_main(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)
    DummyRunner.events_to_stream = []  # Reset for this test

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agents", str(task), "--backend", "openai"]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == "gpt-5"
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called
    assert last_kwargs.get("mcp_servers") is not None

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_openai_thinking_events(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)

    # Create reasoning event with summary in raw_item
    raw_item = SimpleNamespace(
        summary=[SimpleNamespace(text="I need to think about this step by step.")]
    )
    reasoning_event = SimpleNamespace(
        type="run_item_stream_event",
        name="reasoning_item_created",
        item=SimpleNamespace(raw_item=raw_item),
    )

    DummyRunner.events_to_stream = [reasoning_event]

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agents", str(task), "--backend", "openai"]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "gpt-5"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


def test_openai_outfile(monkeypatch, tmp_path):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)
    DummyRunner.events_to_stream = []  # Reset for this test

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(
        run_main(
            mod,
            ["think-agents", str(task), "-o", str(out_file), "--backend", "openai"],
        )
    )

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == "gpt-5"
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_openai_thinking_events_stdout(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)

    # Create reasoning event with summary in raw_item
    raw_item = SimpleNamespace(
        summary=[SimpleNamespace(text="I need to think about this step by step.")]
    )
    reasoning_event = SimpleNamespace(
        type="run_item_stream_event",
        name="reasoning_item_created",
        item=SimpleNamespace(raw_item=raw_item),
    )

    DummyRunner.events_to_stream = [reasoning_event]

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agents", str(task), "--backend", "openai"]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "gpt-5"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


def test_openai_outfile_error(monkeypatch, tmp_path):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)

    # Make the stream_events raise an error
    class ErrorStreamResult:
        final_output = "ok"

        async def stream_events(self):
            # Yield nothing then raise error to simulate streaming error
            if False:
                yield
            raise RuntimeError("boom")

    # Override run_streamed to return error stream
    DummyRunner.run_streamed = lambda *a, **k: ErrorStreamResult()

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    with pytest.raises(RuntimeError):
        asyncio.run(
            run_main(
                mod,
                ["think-agents", str(task), "-o", str(out_file), "--backend", "openai"],
            )
        )

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[-1]["event"] == "error"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["error"] == "boom"
    assert "trace" in events[-1]

    logged = list((journal / "agents").glob("*.jsonl"))
    # Two log files may be created (one for the start and one for the error)
    assert len(logged) >= 1
    # Combine all logged events from all files
    all_logged_events = []
    for log_file in logged:
        all_logged_events.extend(
            [json.loads(line) for line in log_file.read_text().splitlines()]
        )
    # Check that our events are in the logged events
    assert any(
        e["event"] == "error" and e["error"] == "boom" for e in all_logged_events
    )


def test_openai_thinking_events_error(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)

    # Create reasoning event with summary in raw_item
    raw_item = SimpleNamespace(
        summary=[SimpleNamespace(text="I need to think about this step by step.")]
    )
    reasoning_event = SimpleNamespace(
        type="run_item_stream_event",
        name="reasoning_item_created",
        item=SimpleNamespace(raw_item=raw_item),
    )

    DummyRunner.events_to_stream = [reasoning_event]

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agents", str(task), "--backend", "openai"]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "gpt-5"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


def test_openai_tool_call_events(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)

    # Create tool call and output events
    tool_call_raw = SimpleNamespace(
        name="search_web", id="call_123", arguments='{"query": "weather today"}'
    )
    tool_call_event = SimpleNamespace(
        type="run_item_stream_event",
        name="tool_called",
        item=SimpleNamespace(raw_item=tool_call_raw),
    )

    tool_output_raw = SimpleNamespace(tool_call_id="call_123")
    tool_output_event = SimpleNamespace(
        type="run_item_stream_event",
        name="tool_output",
        item=SimpleNamespace(raw_item=tool_output_raw, output="Sunny, 75°F"),
    )

    DummyRunner.events_to_stream = [tool_call_event, tool_output_event]

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory and MCP URI file
    agents_dir = journal / "agents"
    agents_dir.mkdir()
    mcp_uri_file = agents_dir / "mcp.uri"
    mcp_uri_file.write_text("http://localhost:5173/mcp")

    task = tmp_path / "task.txt"
    task.write_text("what's the weather?")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-agents", str(task), "--backend", "openai"]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check start event
    assert events[0]["event"] == "start"
    assert events[0]["prompt"] == "what's the weather?"

    # Look for tool_start event
    tool_start_events = [e for e in events if e["event"] == "tool_start"]
    assert len(tool_start_events) == 1
    assert tool_start_events[0]["tool"] == "search_web"
    assert tool_start_events[0]["args"] == {"query": "weather today"}
    assert tool_start_events[0]["call_id"] == "call_123"
    assert isinstance(tool_start_events[0]["ts"], int)

    # Look for tool_end event
    tool_end_events = [e for e in events if e["event"] == "tool_end"]
    assert len(tool_end_events) == 1
    assert tool_end_events[0]["tool"] == "search_web"
    assert tool_end_events[0]["args"] == {"query": "weather today"}
    assert tool_end_events[0]["result"] == "Sunny, 75°F"
    assert tool_end_events[0]["call_id"] == "call_123"
    assert isinstance(tool_end_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called
