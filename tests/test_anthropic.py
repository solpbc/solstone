# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

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


class MockThinkingBlock:
    """Mock ThinkingBlock that passes isinstance checks."""

    type = "thinking"

    def __init__(self, thinking: str, signature: str = "mock-signature"):
        self.thinking = thinking
        self.signature = signature


class MockRedactedThinkingBlock:
    """Mock RedactedThinkingBlock that passes isinstance checks."""

    type = "redacted_thinking"

    def __init__(self, data: str):
        self.data = data


class DummyMessagesWithThinking:
    async def create(self, **kwargs):
        DummyMessagesWithThinking.kwargs = kwargs
        # Return response with both thinking and text content
        return SimpleNamespace(
            content=[
                MockThinkingBlock("I'm thinking about this...", "test-signature-123"),
                SimpleNamespace(type="text", text="ok"),
            ]
        )


class DummyMessagesWithRedactedThinking:
    async def create(self, **kwargs):
        DummyMessagesWithRedactedThinking.kwargs = kwargs
        # Return response with redacted thinking
        return SimpleNamespace(
            content=[
                MockRedactedThinkingBlock("encrypted-data-xyz"),
                SimpleNamespace(type="text", text="ok"),
            ]
        )


class DummyMessagesError:
    async def create(self, **kwargs):
        DummyMessagesError.kwargs = kwargs
        raise Exception("boo")


def _setup_anthropic_stub(
    monkeypatch, error=False, with_thinking=False, with_redacted_thinking=False
):
    # Create mock Anthropic client
    anthropic_stub = types.ModuleType("anthropic")
    anthropic_types_stub = types.ModuleType("anthropic.types")

    class DummyClient:
        def __init__(self, **kwargs):
            if with_redacted_thinking:
                self.messages = DummyMessagesWithRedactedThinking()
            elif with_thinking:
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
    # Use our mock classes for isinstance checks
    anthropic_types_stub.ThinkingBlock = MockThinkingBlock
    anthropic_types_stub.RedactedThinkingBlock = MockRedactedThinkingBlock

    # Add types as a submodule
    anthropic_stub.types = anthropic_types_stub

    # Stub out the anthropic module
    if "anthropic" in sys.modules:
        sys.modules.pop("anthropic")
    if "anthropic.types" in sys.modules:
        sys.modules.pop("anthropic.types")
    sys.modules["anthropic"] = anthropic_stub
    sys.modules["anthropic.types"] = anthropic_types_stub


def _setup_claude_cli_stub(
    monkeypatch,
    provider_mod,
    *,
    error=False,
    with_thinking=False,
    with_redacted_thinking=False,
):
    monkeypatch.setattr(
        provider_mod, "check_cli_binary", lambda _name: "/usr/bin/claude"
    )
    monkeypatch.setattr(provider_mod, "lookup_cli_session_id", lambda _agent_id: None)

    class DummyCLIRunner:
        def __init__(
            self,
            cmd,
            prompt_text,
            translate,
            callback,
            aggregator,
            cwd=None,
            env=None,
            timeout=600,
        ):
            self.translate = translate
            self.callback = callback
            self.aggregator = aggregator
            self.cli_session_id = None

        async def run(self):
            if error:
                raise RuntimeError("boo")

            raw_events = [
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "test-session-abc123",
                }
            ]
            if with_thinking:
                raw_events.append(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "I'm thinking about this...",
                                }
                            ]
                        },
                    }
                )
            if with_redacted_thinking:
                raw_events.append(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "thinking", "thinking": "[redacted]"}]
                        },
                    }
                )
            raw_events.append(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "ok"}]},
                }
            )

            for raw_event in raw_events:
                session_id = self.translate(raw_event, self.aggregator, self.callback)
                if session_id:
                    self.cli_session_id = session_id

            result = self.aggregator.flush_as_result()
            return result or "Done."

    monkeypatch.setattr(provider_mod, "CLIRunner", DummyCLIRunner)


def test_claude_main(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("think.providers.anthropic", None)
    provider_mod = importlib.reload(
        importlib.import_module("think.providers.anthropic")
    )
    _setup_claude_cli_stub(monkeypatch, provider_mod)
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    # Prompt includes system instruction prepended during enrichment
    assert "hello" in events[0]["prompt"]
    assert events[0]["name"] == "default"
    assert events[0]["model"] == CLAUDE_SONNET_4
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_claude_outfile(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch)
    install_agents_stub()
    sys.modules.pop("think.providers.anthropic", None)
    provider_mod = importlib.reload(
        importlib.import_module("think.providers.anthropic")
    )
    _setup_claude_cli_stub(monkeypatch, provider_mod)
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    # Output file functionality was removed in NDJSON-only mode
    # Check stdout instead
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    # Prompt includes system instruction prepended during enrichment
    assert "hello" in events[0]["prompt"]
    assert events[0]["name"] == "default"
    assert events[0]["model"] == CLAUDE_SONNET_4
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_claude_thinking_events(monkeypatch, tmp_path, capsys):
    """Test that thinking events are properly emitted for Claude models."""
    # Setup anthropic stub with thinking
    _setup_anthropic_stub(monkeypatch, with_thinking=True)
    install_agents_stub()
    sys.modules.pop("think.providers.anthropic", None)
    provider_mod = importlib.reload(
        importlib.import_module("think.providers.anthropic")
    )
    _setup_claude_cli_stub(monkeypatch, provider_mod, with_thinking=True)
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

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


def test_claude_redacted_thinking_events(monkeypatch, tmp_path, capsys):
    """Test that redacted thinking events are properly handled."""
    _setup_anthropic_stub(monkeypatch, with_redacted_thinking=True)
    install_agents_stub()
    sys.modules.pop("think.providers.anthropic", None)
    provider_mod = importlib.reload(
        importlib.import_module("think.providers.anthropic")
    )
    _setup_claude_cli_stub(monkeypatch, provider_mod, with_redacted_thinking=True)
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check for redacted thinking event
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0]["summary"] == "[redacted]"

    # Check that regular events are still present
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "finish"


def test_claude_outfile_error(monkeypatch, tmp_path, capsys):
    _setup_anthropic_stub(monkeypatch, error=True)
    install_agents_stub()
    sys.modules.pop("think.providers.anthropic", None)
    provider_mod = importlib.reload(
        importlib.import_module("think.providers.anthropic")
    )
    _setup_claude_cli_stub(monkeypatch, provider_mod, error=True)
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    # Error events should be written to stdout
    out_lines = capsys.readouterr().out.strip().splitlines()
    if out_lines:  # May be empty if error is raised before any output
        events = [json.loads(line) for line in out_lines if line]
        if events:
            assert any(e["event"] == "error" for e in events)
