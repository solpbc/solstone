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


def test_openai_main(monkeypatch, tmp_path, capsys):
    agents_stub = types.ModuleType("agents")

    last_kwargs = {}

    class DummyAgent:
        def __init__(self, *a, **k):
            last_kwargs.update(k)

    class DummyRunner:
        called = False

        @staticmethod
        async def run(agent, prompt, session=None, run_config=None):
            DummyRunner.called = True
            return SimpleNamespace(final_output="ok")

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.RunConfig = lambda **k: SimpleNamespace()
    agents_stub.ModelSettings = lambda **k: SimpleNamespace()
    agents_stub.set_default_openai_key = lambda k: None

    class DummySession:
        def __init__(self, *a, **k):
            pass

    agents_stub.SQLiteSession = DummySession
    agents_stub.AgentHooks = object
    agents_stub.enable_verbose_stdout_logging = lambda: None

    agents_mcp_stub = types.ModuleType("agents.mcp")

    class DummyMCP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_stub.MCPServerStdio = lambda **k: DummyMCP()

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules.pop("think.openai", None)

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
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
    assert events[0]["model"] == "o4-mini"
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called
    assert last_kwargs.get("mcp_servers") is not None

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_openai_outfile(monkeypatch, tmp_path):
    agents_stub = types.ModuleType("agents")

    class DummyAgent:
        def __init__(self, *a, **k):
            pass

    class DummyRunner:
        @staticmethod
        async def run(agent, prompt, session=None, run_config=None):
            return SimpleNamespace(final_output="ok")

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.RunConfig = lambda **k: SimpleNamespace()
    agents_stub.ModelSettings = lambda **k: SimpleNamespace()
    agents_stub.set_default_openai_key = lambda k: None

    class DummySession:
        def __init__(self, *a, **k):
            pass

    agents_stub.SQLiteSession = DummySession
    agents_stub.AgentHooks = object
    agents_stub.enable_verbose_stdout_logging = lambda: None

    agents_mcp_stub = types.ModuleType("agents.mcp")

    class DummyMCP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_stub.MCPServerStdio = lambda **k: DummyMCP()

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules.pop("think.openai", None)

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
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
    assert events[0]["model"] == "o4-mini"
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_openai_outfile_error(monkeypatch, tmp_path):
    agents_stub = types.ModuleType("agents")

    class DummyAgent:
        def __init__(self, *a, **k):
            pass

    class DummyRunner:
        @staticmethod
        async def run(agent, prompt, session=None, run_config=None):
            raise RuntimeError("boom")

    agents_stub.Agent = DummyAgent
    agents_stub.Runner = DummyRunner
    agents_stub.RunConfig = lambda **k: SimpleNamespace()
    agents_stub.ModelSettings = lambda **k: SimpleNamespace()
    agents_stub.set_default_openai_key = lambda k: None

    class DummySession:
        def __init__(self, *a, **k):
            pass

    agents_stub.SQLiteSession = DummySession
    agents_stub.AgentHooks = object
    agents_stub.enable_verbose_stdout_logging = lambda: None

    agents_mcp_stub = types.ModuleType("agents.mcp")

    class DummyMCP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    agents_mcp_stub.MCPServerStdio = lambda **k: DummyMCP()

    for key in list(sys.modules.keys()):
        if key == "agents" or key.startswith("agents."):
            sys.modules.pop(key)
    sys.modules["agents"] = agents_stub
    sys.modules["agents.mcp"] = agents_mcp_stub
    sys.modules.pop("think.openai", None)

    importlib.reload(importlib.import_module("think.openai"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
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

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events
