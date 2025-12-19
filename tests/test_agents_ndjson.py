"""Tests for NDJSON-only input in muse.agents."""

import asyncio
import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from think.models import GPT_5


@pytest.fixture
def mock_journal(tmp_path, monkeypatch):
    """Set up a temporary journal directory."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    agents_path = journal_path / "agents"
    agents_path.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))
    return journal_path


async def mock_run_agent(config, on_event=None):
    """Mock run_agent function for testing."""
    # Extract values from config
    prompt = config.get("prompt", "")
    backend = config.get("backend", "openai")
    model = config.get("model", "")
    persona = config.get("persona", "default")

    # Emit events through the callback if provided
    if on_event:
        on_event(
            {
                "event": "start",
                "prompt": prompt,
                "backend": backend,
                "model": model,
                "persona": persona,
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


def test_ndjson_single_request(mock_journal, monkeypatch, capsys):
    """Test processing a single NDJSON request from stdin."""
    # Mock stdin with NDJSON data
    ndjson_input = json.dumps(
        {
            "prompt": "What is 2+2?",
            "backend": "openai",
            "persona": "default",
            "model": GPT_5,
            "max_tokens": 100,
            "mcp_server_url": "http://localhost:5175/mcp",
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    # Mock argparse results
    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())

    # Mock agents module to prevent actual imports
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    # Now import after mocks are in place
    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    # Check output events
    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    events = [json.loads(line) for line in lines if line]

    # Should have start and finish events
    assert events

    start_event = events[0]
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "What is 2+2?"
    assert start_event["backend"] == "openai"
    assert start_event["model"] == GPT_5  # Model comes from config

    finish_events = [e for e in events if e["event"] == "finish"]
    assert finish_events


def test_ndjson_multiple_requests(mock_journal, monkeypatch, capsys):
    """Test processing multiple NDJSON requests from stdin."""
    # Multiple NDJSON lines
    requests = [
        {
            "prompt": "First question",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
        {
            "prompt": "Second question",
            "backend": "anthropic",
            "model": "claude-3",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
        {
            "prompt": "Third question",
            "persona": "technical",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
    ]

    ndjson_input = "\n".join(json.dumps(r) for r in requests)

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent
    mock_anthropic = MagicMock()
    mock_anthropic.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", mock_anthropic)
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
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


def test_ndjson_invalid_json(mock_journal, monkeypatch, capsys):
    """Test handling of invalid JSON in NDJSON input."""
    # Mix of valid and invalid JSON
    ndjson_input = """{"prompt": "Valid request", "backend": "openai", "mcp_server_url": "http://localhost:5175/mcp"}
not valid json
{"prompt": "Another valid request", "backend": "openai", "mcp_server_url": "http://localhost:5175/mcp"}"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
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


def test_ndjson_missing_prompt(mock_journal, monkeypatch, capsys):
    """Test handling of NDJSON request without required 'prompt' field."""
    ndjson_input = json.dumps(
        {
            "backend": "openai",
            "model": GPT_5,
            # Missing 'prompt' field
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    # Should have an error event for missing prompt
    assert len(lines) >= 1
    error_event = json.loads(lines[0])
    assert error_event["event"] == "error"
    assert "Missing 'prompt'" in error_event["error"]


def test_ndjson_empty_lines(mock_journal, monkeypatch, capsys):
    """Test that empty lines in NDJSON input are ignored."""
    ndjson_input = """{"prompt": "First"}

{"prompt": "Second"}

"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    # Should process both requests, ignoring empty lines
    assert len(start_events) == 2


def test_default_values(mock_journal, monkeypatch, capsys):
    """Test that default values are applied when not specified."""
    # Minimal request with only prompt
    ndjson_input = json.dumps({"prompt": "Test prompt"})

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    start_event = json.loads(lines[0])
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "Test prompt"
    assert start_event["backend"] == "openai"  # Default backend
    assert start_event["persona"] == "default"  # Default persona


def test_openai_key_setting(mock_journal, monkeypatch, capsys):
    """Test that OpenAI API key is set when backend is openai."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    ndjson_input = json.dumps(
        {
            "prompt": "Test",
            "backend": "openai",
            "mcp_server_url": "http://localhost:5175/mcp",
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    # Create mock backend modules
    mock_openai = MagicMock()
    mock_openai.run_agent = mock_run_agent

    # Mock the modules in sys.modules before import
    monkeypatch.setitem(sys.modules, "muse.openai", mock_openai)
    monkeypatch.setitem(sys.modules, "muse.anthropic", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.google", MagicMock())
    monkeypatch.setitem(sys.modules, "muse.claude", MagicMock())
    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    from muse.agents import main_async

    mock_set_key = MagicMock()

    with patch("muse.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key", mock_set_key):
            asyncio.run(main_async())

    # Verify set_default_openai_key was called with the API key
    mock_set_key.assert_called_with("test-key-123")
