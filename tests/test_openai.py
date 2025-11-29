import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

from tests.agents_stub import install_agents_stub
from think.models import GPT_5


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def _setup_openai_mocks(monkeypatch, tmp_path):
    """Setup common mocks for OpenAI tests."""
    del monkeypatch, tmp_path  # unused
    return install_agents_stub()


def test_openai_main(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)
    DummyRunner.events_to_stream = []  # Reset for this test

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
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
    assert events[0]["model"] == GPT_5
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called
    assert last_kwargs.get("mcp_servers") is not None

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


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

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    # Create agents directory
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

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
    assert thinking_events[0]["model"] == GPT_5
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


def test_openai_mcp_headers(monkeypatch, tmp_path):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)
    DummyRunner.events_to_stream = []
    last_kwargs.clear()

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "agents").mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "audit headers",
            "backend": "openai",
            "persona": "investigator",
            "agent_id": "999",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    mcp_kwargs = last_kwargs.get("mcp_server")
    assert mcp_kwargs is not None
    headers = mcp_kwargs["params"].get("headers", {})
    assert headers["X-Agent-Id"] == "999"
    assert headers["X-Agent-Persona"] == "investigator"


def test_openai_outfile(monkeypatch, tmp_path, capsys):
    last_kwargs, DummyRunner = _setup_openai_mocks(monkeypatch, tmp_path)
    DummyRunner.events_to_stream = []  # Reset for this test

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    # out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
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
    assert events[0]["model"] == GPT_5
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


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

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

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
    assert thinking_events[0]["model"] == GPT_5
    assert isinstance(thinking_events[0]["ts"], int)

    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "ok"
    assert DummyRunner.called


def test_openai_outfile_error(monkeypatch, tmp_path, capsys):
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

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    # out_file = tmp_path / "out.txt"

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    # Output file functionality was removed in NDJSON-only mode
    # Check stdout instead
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[-1]["event"] == "error"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["error"] == "boom"
    assert "trace" in events[-1]

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


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

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

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
    assert thinking_events[0]["model"] == GPT_5
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

    importlib.reload(importlib.import_module("muse.openai"))
    mod = importlib.reload(importlib.import_module("muse.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()
    agents_dir = journal / "agents"
    agents_dir.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5173/mcp",
        }
    )
    asyncio.run(run_main(mod, ["muse-agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]

    # Check start event
    assert events[0]["event"] == "start"
    assert events[0]["prompt"] == "hello"

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


def test_convert_turns_to_items_format():
    """Test that _convert_turns_to_items uses correct Responses API content types.

    Verifies the fix for the bug where content type was 'text' (Chat Completions API)
    instead of 'input_text'/'output_text' (Responses API).
    """
    from muse.openai import _convert_turns_to_items

    # Test user message uses input_text
    user_turns = [{"role": "user", "content": "Hello, world!"}]
    user_items = _convert_turns_to_items(user_turns)

    assert len(user_items) == 1
    assert user_items[0]["type"] == "message"
    assert user_items[0]["role"] == "user"
    assert user_items[0]["content"][0]["type"] == "input_text"
    assert user_items[0]["content"][0]["text"] == "Hello, world!"

    # Test assistant message uses output_text
    assistant_turns = [{"role": "assistant", "content": "Hi there!"}]
    assistant_items = _convert_turns_to_items(assistant_turns)

    assert len(assistant_items) == 1
    assert assistant_items[0]["type"] == "message"
    assert assistant_items[0]["role"] == "assistant"
    assert assistant_items[0]["content"][0]["type"] == "output_text"
    assert assistant_items[0]["content"][0]["text"] == "Hi there!"

    # Test mixed conversation
    mixed_turns = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Thanks!"},
    ]
    mixed_items = _convert_turns_to_items(mixed_turns)

    assert len(mixed_items) == 3
    assert mixed_items[0]["content"][0]["type"] == "input_text"
    assert mixed_items[1]["content"][0]["type"] == "output_text"
    assert mixed_items[2]["content"][0]["type"] == "input_text"

    # Test empty content is skipped
    empty_turns = [{"role": "user", "content": ""}]
    empty_items = _convert_turns_to_items(empty_turns)
    assert len(empty_items) == 0
