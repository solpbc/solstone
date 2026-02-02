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


async def mock_run_tools(config, on_event=None):
    """Mock run_tools function for testing."""
    prompt = config.get("prompt", "")
    provider = config.get("provider", "")
    model = config.get("model", "")
    name = config.get("name", "default")

    if on_event:
        on_event(
            {
                "event": "start",
                "prompt": prompt,
                "provider": provider,
                "model": model,
                "name": name,
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


def mock_hydrate_config(request: dict) -> dict:
    """Mock hydrate_config that passes through request with minimal additions."""
    config = dict(request)
    # Add required fields if not present
    if "name" not in config:
        config["name"] = "default"
    if "provider" not in config:
        config["provider"] = "google"
    if "model" not in config:
        config["model"] = "gpt-5-mini"
    return config


def mock_all_providers(monkeypatch):
    """Mock all provider modules uniformly with mock_run_tools.

    This ensures tests are not fragile to changes in default provider.
    """
    # Mock providers in think.providers (google, openai, anthropic)
    for provider_name in ("openai", "anthropic", "google"):
        mock_module = MagicMock()
        mock_module.run_tools = mock_run_tools
        monkeypatch.setitem(
            sys.modules, f"think.providers.{provider_name}", mock_module
        )

    monkeypatch.setitem(sys.modules, "agents", MagicMock())

    # Mock hydrate_config to avoid needing real agent configs
    monkeypatch.setattr("think.agents.hydrate_config", mock_hydrate_config)


def test_ndjson_single_request(mock_journal, monkeypatch, capsys):
    """Test processing a single NDJSON request from stdin."""
    ndjson_input = json.dumps(
        {
            "prompt": "What is 2+2?",
            "provider": "openai",
            "name": "default",
            "model": GPT_5,
            "max_output_tokens": 100,
            "mcp_server_url": "http://localhost:5175/mcp",
            "tools": ["search_insights"],
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    events = [json.loads(line) for line in lines if line]

    assert events

    start_event = events[0]
    assert start_event["event"] == "start"
    # Prompt includes system instruction prepended during enrichment
    assert "What is 2+2?" in start_event["prompt"]
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
            "tools": ["search_insights"],
        },
        {
            "prompt": "Second question",
            "provider": "anthropic",
            "model": "claude-3",
            "mcp_server_url": "http://localhost:5175/mcp",
            "tools": ["search_insights"],
        },
        {
            "prompt": "Third question",
            "provider": "google",
            "name": "technical",
            "mcp_server_url": "http://localhost:5175/mcp",
            "tools": ["search_insights"],
        },
    ]

    ndjson_input = "\n".join(json.dumps(r) for r in requests)

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    assert len(lines) >= 6

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    assert len(start_events) == 3
    # Prompts include system instruction prepended during enrichment
    assert "First question" in start_events[0]["prompt"]
    assert "Second question" in start_events[1]["prompt"]
    assert start_events[1]["provider"] == "anthropic"
    assert "Third question" in start_events[2]["prompt"]
    assert start_events[2]["name"] == "technical"


def test_ndjson_invalid_json(mock_journal, monkeypatch, capsys):
    """Test handling of invalid JSON in NDJSON input."""
    ndjson_input = """{"prompt": "Valid request", "provider": "openai", "mcp_server_url": "http://localhost:5175/mcp", "tools": ["search_insights"]}
not valid json
{"prompt": "Another valid request", "provider": "openai", "mcp_server_url": "http://localhost:5175/mcp", "tools": ["search_insights"]}"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
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
            "tools": ["search_insights"],  # Has tools, so needs prompt
        }
    )

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    assert len(lines) >= 1
    error_event = json.loads(lines[0])
    assert error_event["event"] == "error"
    assert "prompt" in error_event["error"].lower()  # Error mentions prompt


def test_ndjson_empty_lines(mock_journal, monkeypatch, capsys):
    """Test that empty lines in NDJSON input are ignored."""
    ndjson_input = """{"prompt": "First", "provider": "openai", "tools": ["search_insights"]}

{"prompt": "Second", "provider": "openai", "tools": ["search_insights"]}

"""

    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False

    mock_all_providers(monkeypatch)

    from think.agents import main_async

    with patch("think.agents.setup_cli", return_value=mock_args):
        asyncio.run(main_async())

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]

    events = [json.loads(line) for line in lines]
    start_events = [e for e in events if e["event"] == "start"]

    assert len(start_events) == 2
