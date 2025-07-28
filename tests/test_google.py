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


def _setup_genai_stub(monkeypatch):
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")

    class DummyChat:
        def __init__(self, model, history=None, config=None):
            self.model = model
            self.history = list(history or [])
            self.config = config

        def get_history(self):
            return list(self.history)

        def record_history(self, content):
            self.history.append(content)

        def send_message(self, message, config=None):
            DummyChat.kwargs = {
                "message": message,
                "config": config,
                "model": self.model,
            }
            return SimpleNamespace(text="ok")

    class DummyChats:
        def create(self, *, model, config=None, history=None):
            return DummyChat(model, history=history, config=config)

    class DummyClient:
        def __init__(self, *a, **k):
            self.chats = DummyChats()

    genai_mod.Client = DummyClient
    genai_mod.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **k: SimpleNamespace(**k),
        ToolConfig=lambda **k: SimpleNamespace(**k),
        FunctionCallingConfig=lambda **k: SimpleNamespace(**k),
        Content=lambda **k: SimpleNamespace(**k),
        Part=lambda **k: SimpleNamespace(**k),
    )
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def call_tool(self, name: str, arguments=None, **kwargs):
        return {"ok": True}


class DummyMCPClient:
    def __init__(self, *a, **k):
        self.session = DummySession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


def _setup_fastmcp_stub(monkeypatch):
    fastmcp_mod = types.ModuleType("fastmcp")
    fastmcp_mod.Client = lambda *a, **k: DummyMCPClient()
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)
    transports_mod = types.ModuleType("fastmcp.client.transports")
    transports_mod.PythonStdioTransport = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fastmcp.client.transports", transports_mod)


def test_genai_main(monkeypatch, tmp_path, capsys):
    _setup_genai_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    sys.modules.pop("think.google", None)
    mod = importlib.reload(importlib.import_module("think.google"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    asyncio.run(run_main(mod, ["think.google", str(task)]))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0] == {"event": "start", "prompt": "hello"}
    assert events[-1] == {"event": "finish", "result": "ok"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_genai_outfile(monkeypatch, tmp_path):
    _setup_genai_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    sys.modules.pop("think.google", None)
    mod = importlib.reload(importlib.import_module("think.google"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    asyncio.run(run_main(mod, ["think.google", str(task), "-o", str(out_file)]))

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[0] == {"event": "start", "prompt": "hello"}
    assert events[-1] == {"event": "finish", "result": "ok"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_genai_outfile_error(monkeypatch, tmp_path):
    _setup_genai_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)

    class ErrorClient(DummyMCPClient):
        async def __aenter__(self):
            raise RuntimeError("boom")

    fastmcp_mod = types.ModuleType("fastmcp")
    fastmcp_mod.Client = lambda *a, **k: ErrorClient()
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)
    transports_mod = types.ModuleType("fastmcp.client.transports")
    transports_mod.PythonStdioTransport = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fastmcp.client.transports", transports_mod)

    sys.modules.pop("think.google", None)
    mod = importlib.reload(importlib.import_module("think.google"))

    journal = tmp_path / "journal"
    journal.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("hello")
    out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    with pytest.raises(RuntimeError):
        asyncio.run(run_main(mod, ["think.google", str(task), "-o", str(out_file)]))

    events = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert events[-1] == {"event": "error", "error": "boom"}

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events
