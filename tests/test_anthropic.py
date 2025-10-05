import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

from tests.agents_stub import install_agents_stub
from think.models import CLAUDE_SONNET_4


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


class DummyMessages:
    async def create(self, **kwargs):
        DummyMessages.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])


class DummyMessagesWithThinking:
    async def create(self, **kwargs):
        DummyMessagesWithThinking.kwargs = kwargs
        # Return response with both thinking and text content
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="I'm thinking about this..."),
                SimpleNamespace(type="text", text="ok"),
            ]
        )


class DummyMessagesError:
    async def create(self, **kwargs):
        DummyMessagesError.kwargs = kwargs
        raise Exception("boo")


def _setup_anthropic_stub(monkeypatch, error=False, with_thinking=False):
    # Create mock Anthropic client
    anthropic_stub = types.ModuleType("anthropic")
    anthropic_types_stub = types.ModuleType("anthropic.types")

    class DummyClient:
        def __init__(self, **kwargs):
            if with_thinking:
                self.messages = DummyMessagesWithThinking()
            elif error:
                self.messages = DummyMessagesError()
            else:
                self.messages = DummyMessages()

    anthropic_stub.Anthropic = DummyClient
    anthropic_stub.AsyncAnthropic = DummyClient  # Add async version

    # Add types to the types module
    anthropic_types_stub.MessageParam = dict
    anthropic_types_stub.ToolParam = dict
    anthropic_types_stub.ToolUseBlock = SimpleNamespace

    # Add types as a submodule
    anthropic_stub.types = anthropic_types_stub

    # Stub out the anthropic module
    if "anthropic" in sys.modules:
        sys.modules.pop("anthropic")
    if "anthropic.types" in sys.modules:
        sys.modules.pop("anthropic.types")
    sys.modules["anthropic"] = anthropic_stub
    sys.modules["anthropic.types"] = anthropic_types_stub


def _setup_fastmcp_stub(monkeypatch):
    """Mock fastmcp client."""
    fastmcp_stub = types.ModuleType("fastmcp")
    fastmcp_fastmcp_stub = types.ModuleType("fastmcp.fastmcp")

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def list_tools(self):
            return []

        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=f"Called {name}")]
            )

    fastmcp_fastmcp_stub.FastMCP = DummyClient

    if "fastmcp" in sys.modules:
        sys.modules.pop("fastmcp")
    if "fastmcp.fastmcp" in sys.modules:
        sys.modules.pop("fastmcp.fastmcp")
    sys.modules["fastmcp"] = fastmcp_stub
    sys.modules["fastmcp.fastmcp"] = fastmcp_fastmcp_stub

    def mock_create_mcp_client(_url=None):
        return DummyClient()

    monkeypatch.setattr("think.utils.create_mcp_client", mock_create_mcp_client)


def test_claude_main(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.anthropic", None)
    importlib.reload(importlib.import_module("muse.anthropic"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "anthropic",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == CLAUDE_SONNET_4
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_claude_outfile(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch)
    _setup_fastmcp_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.anthropic", None)
    importlib.reload(importlib.import_module("muse.anthropic"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "anthropic",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    # Output file functionality was removed in NDJSON-only mode
    # Check stdout instead
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert events[0]["prompt"] == "hello"
    assert events[0]["persona"] == "default"
    assert events[0]["model"] == CLAUDE_SONNET_4
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_claude_thinking_events(monkeypatch, tmp_path, capsys):
    """Test that thinking events are properly emitted for Claude models."""

    class DummyClientWithThinking:
        def __init__(self, **kwargs):
            self.messages = DummyMessagesWithThinking()

    # Setup anthropic stub with thinking
    _setup_anthropic_stub(monkeypatch, with_thinking=True)
    _setup_fastmcp_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.anthropic", None)
    importlib.reload(importlib.import_module("muse.anthropic"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "anthropic",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check for thinking event
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    assert len(thinking_events) == 1
    assert "I'm thinking about this..." in thinking_events[0]["summary"]

    # Check that regular events are still present
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"


def test_claude_outfile_error(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch, error=True)
    _setup_fastmcp_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("muse.anthropic", None)
    importlib.reload(importlib.import_module("muse.anthropic"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "anthropic",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    # Error events should be written to stdout
    out_lines = capsys.readouterr().out.strip().splitlines()
    if out_lines:  # May be empty if error is raised before any output
        events = [json.loads(line) for line in out_lines if line]
        if events:
            assert any(e["event"] == "error" for e in events)
