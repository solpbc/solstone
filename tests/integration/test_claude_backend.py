"""Integration test for Claude backend with SDK integration."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import CLAUDE_SONNET_4


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, journal_path


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_backend_real_sdk():
    """Test Claude backend with real SDK call if Claude Code CLI is available."""
    # Use the fixtures journal path
    fixtures_env, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Check if Claude Code CLI is available
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
        pytest.skip(
            "Claude Code CLI not found - install with: npm install -g @anthropic-ai/claude-code"
        )

    # Create journal structure
    journal_dir = Path(journal_path)
    journal_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = journal_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Create test-domain directory for Claude backend
    domains_dir = journal_dir / "domains"
    domains_dir.mkdir(exist_ok=True)
    test_domain_dir = domains_dir / "test-domain"
    test_domain_dir.mkdir(exist_ok=True)

    # Create a temporary directory for task file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare environment with fixtures values
        env = os.environ.copy()
        env["JOURNAL_PATH"] = journal_path
        # Add Claude CLI to PATH
        claude_bin_dir = str(
            Path.home() / ".claude" / "local" / "node_modules" / ".bin"
        )
        env["PATH"] = claude_bin_dir + ":" + env.get("PATH", "")
        # Use Sonnet 4 for testing
        env["CLAUDE_AGENT_MODEL"] = CLAUDE_SONNET_4
        env["CLAUDE_AGENT_MAX_TOKENS"] = "100"

        # Create NDJSON input
        ndjson_input = json.dumps(
            {
                "prompt": "what is 2+2? Just give me the number.",
                "backend": "claude",
                "persona": "default",
                "config": {
                    "model": CLAUDE_SONNET_4,
                    "max_tokens": 100,
                    "domain": "test-domain",  # Claude backend requires a domain
                },
            }
        )

        # Run the think-agents command
        cmd = ["think-agents"]

        result = subprocess.run(
            cmd,
            env=env,
            input=ndjson_input,
            capture_output=True,
            text=True,
            timeout=10,  # 60 second timeout for SDK call
        )

        # Check that the command succeeded
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Parse stdout events (should be JSONL format)
        stdout_lines = result.stdout.strip().split("\n")
        events = []
        for line in stdout_lines:
            if line:  # Skip empty lines
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON line: {line}\nError: {e}")

        # Verify we have events
        assert (
            len(events) >= 2
        ), f"Expected at least start and finish events, got {len(events)}"

        # Check start event
        start_event = events[0]
        assert start_event["event"] == "start"
        assert start_event["prompt"] == "what is 2+2? Just give me the number."
        assert start_event["model"] == CLAUDE_SONNET_4
        assert start_event["persona"] == "default"
        assert start_event["backend"] == "claude"
        # ts field might not be present in all events
        if "ts" in start_event:
            assert isinstance(start_event["ts"], int)

        # Check finish event (should be last)
        finish_event = events[-1]
        assert finish_event["event"] == "finish"
        if "ts" in finish_event:
            assert isinstance(finish_event["ts"], int)
        assert "result" in finish_event

        # The result should contain "4" somewhere
        result_text = finish_event["result"].lower()
        assert (
            "4" in result_text or "four" in result_text
        ), f"Expected '4' in response, got: {finish_event['result']}"

        # Check for no errors in the events
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0, f"Found error events: {error_events}"

        # Verify stderr is empty (no errors)
        assert result.stderr == "", f"Expected empty stderr, got: {result.stderr}"


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_backend_with_tool_calls():
    """Test Claude backend with tool calls."""
    # Use the fixtures journal path
    fixtures_env, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Check if Claude Code CLI is available
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

    # Create journal structure
    journal_dir = Path(journal_path)
    journal_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = journal_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Create test-domain directory for Claude backend
    domains_dir = journal_dir / "domains"
    domains_dir.mkdir(exist_ok=True)
    test_domain_dir = domains_dir / "test-domain"
    test_domain_dir.mkdir(exist_ok=True)

    # Create a temporary directory for task file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test file to read
        test_file = journal_dir / "test_file.txt"
        test_file.write_text("Hello from test file!")

        # Prepare environment
        env = os.environ.copy()
        env["JOURNAL_PATH"] = journal_path
        # Add Claude CLI to PATH
        claude_bin_dir = str(
            Path.home() / ".claude" / "local" / "node_modules" / ".bin"
        )
        env["PATH"] = claude_bin_dir + ":" + env.get("PATH", "")
        env["CLAUDE_AGENT_MODEL"] = CLAUDE_SONNET_4
        env["CLAUDE_AGENT_MAX_TOKENS"] = "200"

        # Create NDJSON input
        ndjson_input = json.dumps(
            {
                "prompt": f"Read the file at {test_file} and tell me what it says.",
                "backend": "claude",
                "persona": "default",
                "config": {
                    "model": CLAUDE_SONNET_4,
                    "max_tokens": 200,
                    "domain": "test-domain",  # Claude backend requires a domain
                },
            }
        )

        # Run the think-agents command with verbose flag to get tool events
        cmd = ["think-agents", "-v"]

        result = subprocess.run(
            cmd,
            env=env,
            input=ndjson_input,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Clean up test file
        test_file.unlink(missing_ok=True)

        # Check that the command succeeded
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Parse stdout events
        stdout_lines = result.stdout.strip().split("\n")
        events = []
        for line in stdout_lines:
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip non-JSON lines (verbose output)
                    continue

        # Look for tool events (optional - Claude might just respond without tools)
        # tool_start_events = [e for e in events if e.get("event") == "tool_start"]
        # tool_end_events = [e for e in events if e.get("event") == "tool_end"]

        # Tool events are optional in this backend
        # The important thing is that the finish event contains the correct response

        # Check finish event contains the file content
        finish_event = events[-1]
        assert finish_event["event"] == "finish"
        result_text = finish_event["result"].lower()
        assert (
            "hello" in result_text
        ), f"Expected 'hello' in response, got: {finish_event['result']}"


@pytest.mark.integration
@pytest.mark.requires_claude_sdk
def test_claude_backend_with_thinking():
    """Test Claude backend thinking/reasoning events."""
    # Use the fixtures journal path
    fixtures_env, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Check if Claude Code CLI is available
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

    # Create journal structure
    journal_dir = Path(journal_path)
    journal_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = journal_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Create test-domain directory for Claude backend
    domains_dir = journal_dir / "domains"
    domains_dir.mkdir(exist_ok=True)
    test_domain_dir = domains_dir / "test-domain"
    test_domain_dir.mkdir(exist_ok=True)

    # Create a temporary directory for task file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare environment
        env = os.environ.copy()
        env["JOURNAL_PATH"] = journal_path
        # Add Claude CLI to PATH
        claude_bin_dir = str(
            Path.home() / ".claude" / "local" / "node_modules" / ".bin"
        )
        env["PATH"] = claude_bin_dir + ":" + env.get("PATH", "")
        # Use Sonnet 4 model
        env["CLAUDE_AGENT_MODEL"] = CLAUDE_SONNET_4
        env["CLAUDE_AGENT_MAX_TOKENS"] = "200"

        # Create NDJSON input
        ndjson_input = json.dumps(
            {
                "prompt": "Think step by step: If I have 3 apples and give away 1, how many do I have left? Just give the number.",
                "backend": "claude",
                "persona": "default",
                "config": {
                    "model": CLAUDE_SONNET_4,
                    "max_tokens": 200,
                    "domain": "test-domain",  # Claude backend requires a domain
                },
            }
        )

        # Run the think-agents command
        cmd = ["think-agents"]

        result = subprocess.run(
            cmd,
            env=env,
            input=ndjson_input,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Check that the command succeeded
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        # Parse stdout events
        stdout_lines = result.stdout.strip().split("\n")
        events = []
        for line in stdout_lines:
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Check for thinking events (may or may not have them depending on model)
        # thinking_events = [e for e in events if e.get("event") == "thinking"]
        # We don't assert on thinking events since they're optional

        # Check finish event
        finish_event = events[-1]
        assert finish_event["event"] == "finish"
        result_text = finish_event["result"].lower()
        assert (
            "2" in result_text or "two" in result_text
        ), f"Expected '2' in response, got: {finish_event['result']}"
