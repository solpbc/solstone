# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for NDJSON-only input in think.agents."""

import asyncio
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

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
    prompt = config.get("prompt", "")
    provider = config.get("provider", "")
    model = config.get("model", "")
    persona = config.get("persona", "default")

    if on_event:
        on_event(
            {
                "event": "start",
                "prompt": prompt,
                "provider": provider,
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


def mock_all_providers(monkeypatch):
    """Mock all provider modules uniformly with mock_run_agent.

    This ensures tests are not fragile to changes in default provider.
    """
    # Mock providers in think.providers (google, openai, anthropic)
    for provider_name in ("openai", "anthropic", "google"):
        mock_module = MagicMock()
        mock_module.run_agent = mock_run_agent
        monkeypatch.setitem(
            sys.modules, f"think.providers.{provider_name}", mock_module
        )

    monkeypatch.setitem(sys.modules, "agents", MagicMock())


def test_ndjson_single_request(mock_journal, monkeypatch, capsys):
    """Test processing a single NDJSON request from stdin."""
    ndjson_input = json.dumps(
        {
            "prompt": "What is 2+2?",
            "provider": "openai",
            "persona": "default",
            "model": GPT_5,
            "max_output_tokens": 100,
            "mcp_server_url": "http://localhost:5175/mcp",
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    events = [json.loads(line) for line in lines if line]

    assert events

    start_event = events[0]
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "What is 2+2?"
    assert start_event["provider"] == "openai"
    assert start_event["model"] == GPT_5

    finish_events = [e for e in events if e["event"] == "finish"]
    assert finish_events


def test_ndjson_multiple_requests(mock_journal, monkeypatch, capsys):
    """Test processing multiple NDJSON requests from stdin."""
    requests = [
        {
            "prompt": "First question",
            "provider": "openai",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
        {
            "prompt": "Second question",
            "provider": "anthropic",
            "model": "claude-3",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
        {
            "prompt": "Third question",
            "provider": "google",
            "persona": "technical",
            "mcp_server_url": "http://localhost:5175/mcp",
        },
    ]

    ndjson_input = "\n".join(json.dumps(r) for r in requests)

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    assert len(lines) >= 6

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    assert len(start_events) == 3
    assert start_events[0]["prompt"] == "First question"
    assert start_events[1]["prompt"] == "Second question"
    assert start_events[1]["provider"] == "anthropic"
    assert start_events[2]["prompt"] == "Third question"
    assert start_events[2]["persona"] == "technical"


def test_ndjson_invalid_json(mock_journal, monkeypatch, capsys):
    """Test handling of invalid JSON in NDJSON input."""
    ndjson_input = """{"prompt": "Valid request", "provider": "openai", "mcp_server_url": "http://localhost:5175/mcp"}
not valid json
{"prompt": "Another valid request", "provider": "openai", "mcp_server_url": "http://localhost:5175/mcp"}"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]

    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert "Invalid JSON" in error_events[0]["error"]

    start_events = [e for e in events if e["event"] == "start"]
    assert len(start_events) == 2


def test_ndjson_missing_prompt(mock_journal, monkeypatch, capsys):
    """Test handling of NDJSON request without required 'prompt' field."""
    ndjson_input = json.dumps(
        {
            "provider": "openai",
            "model": GPT_5,
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    assert len(lines) >= 1
    error_event = json.loads(lines[0])
    assert error_event["event"] == "error"
    assert "Missing 'prompt'" in error_event["error"]


def test_ndjson_empty_lines(mock_journal, monkeypatch, capsys):
    """Test that empty lines in NDJSON input are ignored."""
    ndjson_input = """{"prompt": "First", "provider": "openai"}

{"prompt": "Second", "provider": "openai"}

"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        with patch("agents.set_default_openai_key"):
            asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    assert len(start_events) == 2
