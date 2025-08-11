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


def _setup_openai_mocks(monkeypatch):
    """Setup common mocks for OpenAI tests."""
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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

    return last_kwargs, DummyRunner


def test_openai_main(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch)

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


def test_openai_thinking_events(monkeypatch, tmp_path, capsys):
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
            # Mock result with reasoning data
            reasoning_item = SimpleNamespace(
                reasoning=SimpleNamespace(
                    summary="I need to think about this step by step.",
                    content="This is my reasoning process.",
                )
            )
            return SimpleNamespace(final_output="ok", new_items=[reasoning_item])

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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    class DummyMCP:  # noqa: F811
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

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

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "o4-mini-2025-04-16"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    class DummyMCP:  # noqa: F811
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

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


def test_openai_thinking_events_stdout(monkeypatch, tmp_path, capsys):
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
            # Mock result with reasoning data
            reasoning_item = SimpleNamespace(
                reasoning=SimpleNamespace(
                    summary="I need to think about this step by step.",
                    content="This is my reasoning process.",
                )
            )
            return SimpleNamespace(final_output="ok", new_items=[reasoning_item])

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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    class DummyMCP:  # noqa: F811
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

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

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "o4-mini-2025-04-16"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    class DummyMCP:  # noqa: F811
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

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
    assert "trace" in events[-1]

    logged = list((journal / "agents").glob("*.jsonl"))
    assert len(logged) == 1
    logged_events = [json.loads(line) for line in logged[0].read_text().splitlines()]
    assert logged_events == events


def test_openai_thinking_events_error(monkeypatch, tmp_path, capsys):
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
            # Mock result with reasoning data
            reasoning_item = SimpleNamespace(
                reasoning=SimpleNamespace(
                    summary="I need to think about this step by step.",
                    content="This is my reasoning process.",
                )
            )
            return SimpleNamespace(final_output="ok", new_items=[reasoning_item])

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

    class DummyMCP:  # noqa: F811
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

    # Mock create_mcp_client to avoid reading URI file
    class DummyMCP:  # noqa: F811
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def mock_create_mcp_client():
        return DummyMCP()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)

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

    # Check that we have start, thinking, and finish events
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"

    # Look for thinking event
    thinking_events = [e for e in events if e["event"] == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "I need to think about this step by step."
    assert thinking_events[0]["model"] == "o4-mini-2025-04-16"
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called
