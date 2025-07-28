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


class DummyMessages:
    async def create(self, **kwargs):
        DummyMessages.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])


class DummyClient:
    def __init__(self, *a, **k):
        self.messages = DummyMessages()


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def call_tool(self, name: str, arguments=None, **kwargs):
        return {"ok": True}

    def list_tools(self):
        return []


class DummyMCPClient:
    def __init__(self, *a, **k):
        self.session = DummySession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


def _setup_anthropic_stub(monkeypatch):
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.AsyncAnthropic = DummyClient
    types_mod = types.ModuleType("anthropic.types")
    types_mod.MessageParam = dict
    types_mod.ToolParam = dict
    types_mod.ToolUseBlock = SimpleNamespace
    anthropic_mod.types = types_mod
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)
    monkeypatch.setitem(sys.modules, "anthropic.types", types_mod)


def _setup_fastmcp_stub(monkeypatch):
    fastmcp_mod = types.ModuleType("fastmcp")
    fastmcp_mod.Client = lambda *a, **k: DummyMCPClient()
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)
    transports_mod = types.ModuleType("fastmcp.client.transports")
    transports_mod.PythonStdioTransport = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fastmcp.client.transports", transports_mod)


def test_claude_main(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    sys.modules.pop("think.anthropic", None)
    mod = importlib.reload(importlib.import_module("think.anthropic"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-claude", str(task)]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0] == {
        "event": "start",
        "prompt": "hello",
        "persona": "default",
        "model": "claude-opus-4-20250514",
    }
    assert events[-1] == {"event": "finish", "result": "ok"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_claude_outfile(monkeypatch, tmp_path):
    _setup_anthropic_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    sys.modules.pop("think.anthropic", None)
    mod = importlib.reload(importlib.import_module("think.anthropic"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    asyncio.run(run_main(mod, ["think-claude", str(task), "-o", str(out_file)]))

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[0] == {
        "event": "start",
        "prompt": "hello",
        "persona": "default",
        "model": "claude-opus-4-20250514",
    }
    assert events[-1] == {"event": "finish", "result": "ok"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_claude_outfile_error(monkeypatch, tmp_path):
    _setup_anthropic_stub(monkeypatch)

    class ErrorClient(DummyMCPClient):
        async def __aenter__(self):
            raise RuntimeError("boom")

    fastmcp_mod = types.ModuleType("fastmcp")
    fastmcp_mod.Client = lambda *a, **k: ErrorClient()
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)
    transports_mod = types.ModuleType("fastmcp.client.transports")
    transports_mod.PythonStdioTransport = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fastmcp.client.transports", transports_mod)

    sys.modules.pop("think.anthropic", None)
    mod = importlib.reload(importlib.import_module("think.anthropic"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    with pytest.raises(RuntimeError):
        asyncio.run(run_main(mod, ["think-claude", str(task), "-o", str(out_file)]))

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[-1] == {"event": "error", "error": "boom"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events
