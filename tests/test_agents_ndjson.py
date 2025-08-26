"""Tests for NDJSON-only input in think.agents."""

import asyncio
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_journal(tmp_path, monkeypatch):
    """Set up a temporary journal directory."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    agents_path = journal_path / "agents"
    agents_path.mkdir()

    # Create mock MCP URI file
    mcp_uri = agents_path / "mcp.uri"
    mcp_uri.write_text("http://localhost:5175/mcp")

    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))
    return journal_path


@pytest.fixture
def mock_run_agent(monkeypatch):
    """Mock the run_agent function."""

    async def mock_agent(prompt, **kwargs):
        # Emit events through the callback if provided
        on_event = kwargs.get("on_event")
        if on_event:
            # Extract model from config if present
            config = kwargs.get("config", {})
            model = config.get("model", kwargs.get("model", ""))

            on_event(
                {
                    "event": "start",
                    "prompt": prompt,
                    "backend": kwargs.get("backend", "openai"),
                    "model": model,
                    "persona": kwargs.get("persona", "default"),
                    "ts": 1234567890,
                }
            )
            on_event(
                {
                    "event": "finish",
                    "result": f"Response to: {prompt}",
                    "ts": 1234567891,
                }
            )
        return f"Response to: {prompt}"

    monkeypatch.setattr("think.agents.run_agent", mock_agent)
    return mock_agent


def test_ndjson_single_request(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test processing a single NDJSON request from stdin."""
    from think.agents import main_async

    # Mock stdin with NDJSON data
    ndjson_input = json.dumps(
        {
            "prompt": "What is 2+2?",
            "backend": "openai",
            "persona": "default",
            "config": {
                "model": "gpt-4o",
                "max_tokens": 100,
            },
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    # Mock argparse results
    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    # Check output events
    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    # Should have start and finish events
    assert len(lines) >= 2

    start_event = json.loads(lines[0])
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "What is 2+2?"
    assert start_event["backend"] == "openai"
    assert start_event["model"] == "gpt-4o"  # Model comes from config

    finish_event = json.loads(lines[1])
    assert finish_event["event"] == "finish"
    assert "Response to: What is 2+2?" in finish_event["result"]


def test_ndjson_multiple_requests(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test processing multiple NDJSON requests from stdin."""
    from think.agents import main_async

    # Multiple NDJSON lines
    requests = [
        {"prompt": "First question", "backend": "openai"},
        {
            "prompt": "Second question",
            "backend": "anthropic",
            "config": {"model": "claude-3"},
        },
        {"prompt": "Third question", "persona": "technical"},
    ]

    ndjson_input = "\n".join(json.dumps(r) for r in requests)

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    # Should have 2 events per request (start + finish)
    assert len(lines) >= 6

    # Verify each request was processed
    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    assert len(start_events) == 3
    assert start_events[0]["prompt"] == "First question"
    assert start_events[1]["prompt"] == "Second question"
    assert start_events[1]["backend"] == "anthropic"
    assert start_events[2]["prompt"] == "Third question"
    assert start_events[2]["persona"] == "technical"


def test_ndjson_invalid_json(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test handling of invalid JSON in NDJSON input."""
    from think.agents import main_async

    # Mix of valid and invalid JSON
    ndjson_input = """{"prompt": "Valid request"}
not valid json
{"prompt": "Another valid request"}"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]

    # Should have processed valid requests and reported error for invalid
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert "Invalid JSON" in error_events[0]["error"]

    # Valid requests should still be processed
    start_events = [e for e in events if e["event"] == "start"]
    assert len(start_events) == 2


def test_ndjson_missing_prompt(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test handling of NDJSON request without required 'prompt' field."""
    from think.agents import main_async

    ndjson_input = json.dumps(
        {
            "backend": "openai",
            "config": {"model": "gpt-4o"},
            # Missing 'prompt' field
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    # Should have an error event for missing prompt
    assert len(lines) >= 1
    error_event = json.loads(lines[0])
    assert error_event["event"] == "error"
    assert "Missing 'prompt'" in error_event["error"]


def test_ndjson_empty_lines(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test that empty lines in NDJSON input are ignored."""
    from think.agents import main_async

    ndjson_input = """{"prompt": "First"}

{"prompt": "Second"}

"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    # Should process both requests, ignoring empty lines
    assert len(start_events) == 2


def test_default_values(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test that default values are applied when not specified."""
    from think.agents import main_async

    # Minimal request with only prompt
    ndjson_input = json.dumps({"prompt": "Test prompt"})

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    start_event = json.loads(lines[0])
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "Test prompt"
    assert start_event["backend"] == "openai"  # Default backend
    assert start_event["persona"] == "default"  # Default persona


def test_openai_key_setting(mock_journal, mock_run_agent, monkeypatch, capsys):
    """Test that OpenAI API key is set when backend is openai."""
    from think.agents import main_async

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    ndjson_input = json.dumps({"prompt": "Test", "backend": "openai"})

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_set_key = MagicMock()

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key", mock_set_key):
            asyncio.run(main_async())

    # Verify set_default_openai_key was called with the API key
    mock_set_key.assert_called_with("test-key-123")
