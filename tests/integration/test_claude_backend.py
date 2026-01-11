# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for Claude provider with SDK integration."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import CLAUDE_SONNET_4

# --- Shared Test Helpers ---


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None

    load_dotenv(fixtures_env, override=True)
    journal_path = os.getenv("JOURNAL_PATH")
    return fixtures_env, journal_path


def skip_if_claude_cli_unavailable():
    """Check if Claude Code CLI is available, skip test if not."""
    claude_path = Path.home() / ".claude" / "local" / "node_modules" / ".bin" / "claude"
    if not claude_path.exists():
        pytest.skip(f"Claude Code CLI not found at {claude_path}")

    try:
        result = subprocess.run(
            [str(claude_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            pytest.skip("Claude Code CLI not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Claude Code CLI not found")


def setup_test_journal(journal_path: str) -> Path:
    """Create journal structure for testing. Returns journal_dir path."""
    journal_dir = Path(journal_path)
    journal_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = journal_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    health_dir = journal_dir / "health"
    health_dir.mkdir(exist_ok=True)

    return journal_dir


def prepare_test_env(journal_path: str, max_tokens: int = 100) -> dict:
    """Prepare environment variables for test subprocess."""
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    claude_bin_dir = str(Path.home() / ".claude" / "local" / "node_modules" / ".bin")
    env["PATH"] = claude_bin_dir + ":" + env.get("PATH", "")
    env["CLAUDE_AGENT_MODEL"] = CLAUDE_SONNET_4
    env["CLAUDE_AGENT_MAX_TOKENS"] = str(max_tokens)
    return env


def parse_events(stdout: str, strict: bool = True) -> list:
    """Parse JSONL events from stdout."""
    events = []
    for line in stdout.strip().split("\n"):
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                if strict:
                    pytest.fail(f"Failed to parse JSON line: {line}\nError: {e}")
                # Non-strict: skip non-JSON lines (verbose output)
    return events


def handle_error_event(finish_event: dict):
    """Handle error events, skipping for known intermittent issues."""
    if finish_event.get("event") == "error":
        error_msg = finish_event.get("error", "Unknown error")
        if "CLI not found" in error_msg:
            pytest.skip(f"Claude Code CLI issue: {error_msg}")
        elif "rate" in error_msg.lower() or "retry" in error_msg.lower():
            pytest.skip(f"Intermittent Claude API error: {error_msg}")
        else:
            pytest.fail(f"Unexpected error: {finish_event}")


# --- Tests ---


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_provider_real_sdk():
    """Test Claude provider with real SDK call if Claude Code CLI is available."""
    fixtures_env, journal_path = get_fixtures_env()
    if not fixtures_env:
        pytest.skip("fixtures/.env not found")
    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    skip_if_claude_cli_unavailable()
    setup_test_journal(journal_path)
    env = prepare_test_env(journal_path, max_tokens=100)

    ndjson_input = json.dumps(
        {
            "prompt": "what is 2+2? Just give me the number.",
            "provider": "claude",
            "persona": "default",
            "model": CLAUDE_SONNET_4,
            "max_tokens": 100,
        }
    )

    result = subprocess.run(
        ["muse-agents"],
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    events = parse_events(result.stdout, strict=True)
    assert (
        len(events) >= 2
    ), f"Expected at least start and finish events, got {len(events)}"

    # Check start event
    start_event = events[0]
    assert start_event["event"] == "start"
    assert start_event["prompt"] == "what is 2+2? Just give me the number."
    assert start_event["model"] == CLAUDE_SONNET_4
    assert start_event["persona"] == "default"
    assert start_event["provider"] == "claude"
    # Claude provider now emits journal_path instead of facet
    assert "journal_path" in start_event
    if "ts" in start_event:
        assert isinstance(start_event["ts"], int)

    # Check finish event
    finish_event = events[-1]
    handle_error_event(finish_event)

    assert (
        finish_event["event"] == "finish"
    ), f"Expected finish event, got: {finish_event}"
    if "ts" in finish_event:
        assert isinstance(finish_event["ts"], int)
    assert "result" in finish_event

    result_text = finish_event["result"].lower()
    assert (
        "4" in result_text or "four" in result_text
    ), f"Expected '4' in response, got: {finish_event['result']}"

    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 0, f"Found error events: {error_events}"
    assert result.stderr == "", f"Expected empty stderr, got: {result.stderr}"


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_provider_with_tool_calls():
    """Test Claude provider with tool calls (read-only file access)."""
    fixtures_env, journal_path = get_fixtures_env()
    if not fixtures_env:
        pytest.skip("fixtures/.env not found")
    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    skip_if_claude_cli_unavailable()
    journal_dir = setup_test_journal(journal_path)
    env = prepare_test_env(journal_path, max_tokens=200)

    # Create a test file in the journal (not in a facet)
    test_file = journal_dir / "test_file.txt"
    test_file.write_text("Hello from test file!")

    try:
        ndjson_input = json.dumps(
            {
                "prompt": f"Read the file at {test_file} and tell me what it says.",
                "provider": "claude",
                "persona": "default",
                "model": CLAUDE_SONNET_4,
                "max_tokens": 200,
            }
        )

        result = subprocess.run(
            ["muse-agents", "-v"],
            env=env,
            input=ndjson_input,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        test_file.unlink(missing_ok=True)

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    events = parse_events(result.stdout, strict=False)

    finish_event = events[-1]
    handle_error_event(finish_event)

    assert (
        finish_event["event"] == "finish"
    ), f"Expected finish event, got: {finish_event}"
    result_text = finish_event["result"].lower()
    assert (
        "hello" in result_text
    ), f"Expected 'hello' in response, got: {finish_event['result']}"


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_provider_with_thinking():
    """Test Claude provider thinking/reasoning events."""
    fixtures_env, journal_path = get_fixtures_env()
    if not fixtures_env:
        pytest.skip("fixtures/.env not found")
    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    skip_if_claude_cli_unavailable()
    setup_test_journal(journal_path)
    env = prepare_test_env(journal_path, max_tokens=200)

    ndjson_input = json.dumps(
        {
            "prompt": "Think step by step: If I have 3 apples and give away 1, how many do I have left? Just give the number.",
            "provider": "claude",
            "persona": "default",
            "model": CLAUDE_SONNET_4,
            "max_tokens": 200,
        }
    )

    result = subprocess.run(
        ["muse-agents"],
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    events = parse_events(result.stdout, strict=False)

    finish_event = events[-1]
    handle_error_event(finish_event)

    assert (
        finish_event["event"] == "finish"
    ), f"Expected finish event, got: {finish_event}"
    result_text = finish_event["result"].lower()
    assert (
        "2" in result_text or "two" in result_text
    ), f"Expected '2' in response, got: {finish_event['result']}"
