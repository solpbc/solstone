"""Integration test for Anthropic backend with real API calls."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import CLAUDE_HAIKU_3_5, CLAUDE_SONNET_4


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_anthropic_backend_real_api():
    """Test Anthropic backend with real API call if key is available."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Verify journal structure exists
    journal_dir = Path(journal_path)
    if not journal_dir.exists():
        pytest.skip(f"Journal directory {journal_dir} does not exist")

    agents_dir = journal_dir / "agents"
    if not agents_dir.exists():
        pytest.skip(f"Agents directory {agents_dir} does not exist")

    # Start MCP server in the background
    mcp_env = os.environ.copy()
    mcp_env["JOURNAL_PATH"] = journal_path
    mcp_server = subprocess.Popen(
        ["think-mcp-tools", "--transport", "http", "--port", "5176"],
        env=mcp_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the server time to start
    time.sleep(2)

    try:
        # Update the MCP URI file to point to our test server
        mcp_uri_file = agents_dir / "mcp.uri"
        mcp_uri_file.write_text("http://localhost:5176/mcp")

        # Create a temporary directory for task file only
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create task file with simple math question
            task_file = tmpdir / "task.txt"
            task_file.write_text("what is 1+1? Just give me the number.")

            # Prepare environment with fixtures values
            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal_path
            env["ANTHROPIC_API_KEY"] = api_key
            # Use the default model (or override with haiku for testing)
            # Using default model
            env["ANTHROPIC_AGENT_MAX_TOKENS"] = "100"
            env["ANTHROPIC_AGENT_MAX_TURNS"] = "1"

            # Run the think-agents command
            cmd = ["think-agents", str(task_file), "--backend", "anthropic"]

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for API call
            )

            # Check that the command succeeded
            assert (
                result.returncode == 0
            ), f"Command failed with stderr: {result.stderr}"

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
            assert start_event["prompt"] == "what is 1+1? Just give me the number."
            # Check the model - should be either the default or the one we set
            expected_model = env.get("ANTHROPIC_AGENT_MODEL", CLAUDE_SONNET_4)
            assert start_event["model"] == expected_model
            assert start_event["persona"] == "default"
            assert isinstance(start_event["ts"], int)

            # Check finish event (should be last)
            finish_event = events[-1]
            assert finish_event["event"] == "finish"
            assert isinstance(finish_event["ts"], int)
            assert "result" in finish_event

            # The result should contain "2" somewhere
            result_text = finish_event["result"].lower()
            assert (
                "2" in result_text or "two" in result_text
            ), f"Expected '2' in response, got: {finish_event['result']}"

            # Check for no errors in the events
            error_events = [e for e in events if e.get("event") == "error"]
            assert len(error_events) == 0, f"Found error events: {error_events}"

            # Verify stderr is empty (no errors)
            assert result.stderr == "", f"Expected empty stderr, got: {result.stderr}"

            # Verify that a log file was created in the journal
            log_files = list(agents_dir.glob("*.jsonl"))
            # There should be at least one log file (there may be more from previous runs)
            assert (
                len(log_files) >= 1
            ), f"Expected at least 1 log file, found {len(log_files)}"
    finally:
        # Cleanup: terminate the MCP server
        mcp_server.terminate()
        mcp_server.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.requires_api
def test_anthropic_backend_with_verbose():
    """Test Anthropic backend with verbose flag to check debug output."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Verify journal structure exists
    journal_dir = Path(journal_path)
    agents_dir = journal_dir / "agents"

    # Start MCP server in the background
    mcp_env = os.environ.copy()
    mcp_env["JOURNAL_PATH"] = journal_path
    mcp_server = subprocess.Popen(
        [
            "think-mcp-tools",
            "--transport",
            "http",
            "--port",
            "5177",
        ],  # Different port to avoid conflicts
        env=mcp_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the server time to start
    time.sleep(2)

    try:
        # Update the MCP URI file to point to our test server
        mcp_uri_file = agents_dir / "mcp.uri"
        mcp_uri_file.write_text("http://localhost:5177/mcp")

        # Create a temporary directory for task file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create task file
            task_file = tmpdir / "task.txt"
            task_file.write_text("what is 2+2? Just give me the number.")

            # Prepare environment
            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal_path
            env["ANTHROPIC_API_KEY"] = api_key
            # Use default model
            env["ANTHROPIC_AGENT_MAX_TOKENS"] = "100"
            env["ANTHROPIC_AGENT_MAX_TURNS"] = "1"

            # Run with verbose flag
            cmd = [
                "think-agents",
                str(task_file),
                "--backend",
                "anthropic",
                "-v",  # Verbose flag
            ]

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=30
            )

            # With verbose, we might have debug output in stdout mixed with JSON
            # The JSON events should still be parseable
            stdout_lines = result.stdout.strip().split("\n")
            events = []
            for line in stdout_lines:
                if line and line.startswith("{"):  # JSON lines start with {
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip non-JSON lines (debug output)

            # Basic checks
            assert len(events) >= 2, "Should have start and finish events"
            assert events[0]["event"] == "start"
            assert events[-1]["event"] == "finish"

            # Result should contain 4
            result_text = events[-1]["result"].lower()
            assert (
                "4" in result_text or "four" in result_text
            ), f"Expected '4' in response, got: {events[-1]['result']}"
    finally:
        # Cleanup: terminate the MCP server
        mcp_server.terminate()
        mcp_server.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.requires_api
def test_anthropic_backend_with_max_turns():
    """Test Anthropic backend with multiple turns enabled."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Verify journal structure exists
    journal_dir = Path(journal_path)
    agents_dir = journal_dir / "agents"

    # Start MCP server in the background
    mcp_env = os.environ.copy()
    mcp_env["JOURNAL_PATH"] = journal_path
    mcp_server = subprocess.Popen(
        [
            "think-mcp-tools",
            "--transport",
            "http",
            "--port",
            "5178",
        ],  # Different port to avoid conflicts
        env=mcp_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the server time to start
    time.sleep(2)

    try:
        # Update the MCP URI file to point to our test server
        mcp_uri_file = agents_dir / "mcp.uri"
        mcp_uri_file.write_text("http://localhost:5178/mcp")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a task file
            task_file = tmpdir / "task.txt"
            task_file.write_text(
                "What is the square root of 16? Just the number please."
            )

            # Prepare environment
            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal_path
            env["ANTHROPIC_API_KEY"] = api_key
            # Use default model with more tokens for complex reasoning
            env["ANTHROPIC_AGENT_MAX_TOKENS"] = "200"
            env["ANTHROPIC_AGENT_MAX_TURNS"] = "1"

            # Run the command
            cmd = ["think-agents", str(task_file), "--backend", "anthropic"]

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=30
            )

            assert result.returncode == 0

            # Parse events
            stdout_lines = result.stdout.strip().split("\n")
            events = [json.loads(line) for line in stdout_lines if line]

            # Verify the answer is correct
            finish_event = events[-1]
            assert finish_event["event"] == "finish"
            result_text = finish_event["result"].lower()
            assert (
                "4" in result_text or "four" in result_text
            ), f"Expected '4' in response, got: {finish_event['result']}"
    finally:
        # Cleanup: terminate the MCP server
        mcp_server.terminate()
        mcp_server.wait(timeout=5)
