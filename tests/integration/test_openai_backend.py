"""Integration test for OpenAI backend with real API calls."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import GPT_5_MINI


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_openai_backend_basic():
    """Test OpenAI backend with basic prompt, no MCP."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Create NDJSON input with disable_mcp
    ndjson_input = json.dumps(
        {
            "prompt": "what is 1+1? Just give me the number.",
            "backend": "openai",
            "persona": "default",
            "model": GPT_5_MINI,  # Use cheap model for testing
            "max_tokens": 100,
            "disable_mcp": True,
        }
    )

    # Run the muse-agents command
    cmd = ["muse-agents"]
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

    # Parse stdout events (should be JSONL format)
    stdout_lines = result.stdout.strip().split("\n")
    events = []
    for line in stdout_lines:
        if line:
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
    assert start_event["model"] == GPT_5_MINI
    assert start_event["persona"] == "default"
    assert isinstance(start_event["ts"], int)

    # Check finish event
    finish_event = events[-1]
    assert finish_event["event"] == "finish"
    assert isinstance(finish_event["ts"], int)
    assert "result" in finish_event

    # The result should contain "2"
    result_text = finish_event["result"].lower()
    assert (
        "2" in result_text or "two" in result_text
    ), f"Expected '2' in response, got: {finish_event['result']}"

    # Check for no errors
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 0, f"Found error events: {error_events}"

    # Verify stderr is empty
    assert result.stderr == "", f"Expected empty stderr, got: {result.stderr}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_openai_backend_with_reasoning():
    """Test OpenAI backend with reasoning model (o1-mini if available)."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Use standard model since o1 models are not supported with Responses API
    ndjson_input = json.dumps(
        {
            "prompt": "What is the square root of 16? Just the number please.",
            "backend": "openai",
            "persona": "default",
            "model": GPT_5_MINI,
            "max_tokens": 200,
            "disable_mcp": True,
        }
    )

    # Run the muse-agents command
    cmd = ["muse-agents"]
    result = subprocess.run(
        cmd,
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Parse events
    stdout_lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in stdout_lines if line]

    # Check for thinking events (may be present with o1 models)
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    # May or may not have thinking events depending on the model

    # Verify the answer is correct
    finish_event = events[-1]
    assert finish_event["event"] == "finish"
    result_text = finish_event["result"].lower()
    assert (
        "4" in result_text or "four" in result_text
    ), f"Expected '4' in response, got: {finish_event['result']}"
