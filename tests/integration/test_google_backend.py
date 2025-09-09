"""Integration test for Google backend with real API calls."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import GEMINI_FLASH, GEMINI_PRO


def get_fixtures_env():
    """Load the fixtures/.env file and return the environment."""
    fixtures_env = Path(__file__).parent.parent.parent / "fixtures" / ".env"
    if not fixtures_env.exists():
        return None, None, None

    # Load the env file
    load_dotenv(fixtures_env, override=True)

    api_key = os.getenv("GOOGLE_API_KEY")
    journal_path = os.getenv("JOURNAL_PATH")

    return fixtures_env, api_key, journal_path


@pytest.mark.integration
@pytest.mark.requires_api
def test_google_backend_basic():
    """Test Google backend with basic prompt, no MCP."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["GOOGLE_API_KEY"] = api_key

    # Create NDJSON input with disable_mcp
    ndjson_input = json.dumps(
        {
            "prompt": "what is 1+1? Just give me the number.",
            "backend": "google",
            "persona": "default",
            "config": {"max_tokens": 100, "disable_mcp": True},
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
    assert start_event["model"] == GEMINI_FLASH
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

    # Verify stderr has no errors (warnings about thought_signature are OK)
    if result.stderr:
        assert (
            "error" not in result.stderr.lower() or "thought_signature" in result.stderr
        ), f"Unexpected stderr content: {result.stderr}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_google_backend_with_thinking():
    """Test Google backend with thinking enabled."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["GOOGLE_API_KEY"] = api_key

    # Create NDJSON input with thinking model (if available)
    ndjson_input = json.dumps(
        {
            "prompt": "What is the square root of 16? Just the number please.",
            "backend": "google",
            "persona": "default",
            "config": {
                "model": GEMINI_PRO,  # Pro model for thinking
                "max_tokens": 200,
                "disable_mcp": True,
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

    # Allow for model unavailability
    if result.returncode != 0:
        if (
            "model not found" in result.stderr.lower()
            or "invalid model" in result.stderr.lower()
        ):
            pytest.skip("Thinking model not available")
        assert False, f"Command failed with stderr: {result.stderr}"

    # Parse events
    stdout_lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in stdout_lines if line]

    # Check for thinking events (may be present with thinking models)
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    # With thinking models, we might get thinking events

    # Verify the answer is correct
    finish_event = events[-1]
    assert finish_event["event"] == "finish"
    assert "result" in finish_event, f"No result in finish event: {finish_event}"
    if finish_event["result"]:
        result_text = finish_event["result"].lower()
        assert (
            "4" in result_text or "four" in result_text
        ), f"Expected '4' in response, got: {finish_event['result']}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_google_backend_with_verbose():
    """Test Google backend with verbose flag."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["GOOGLE_API_KEY"] = api_key

    # Create NDJSON input
    ndjson_input = json.dumps(
        {
            "prompt": "what is 2+2? Just give me the number.",
            "backend": "google",
            "persona": "default",
            "config": {"max_tokens": 100, "disable_mcp": True},
        }
    )

    # Run with verbose flag
    cmd = ["think-agents", "-v"]
    result = subprocess.run(
        cmd,
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Parse JSON events from stdout
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


@pytest.mark.integration
@pytest.mark.requires_api
def test_google_backend_custom_model():
    """Test Google backend with custom model configuration."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["GOOGLE_API_KEY"] = api_key

    # Use a specific model
    ndjson_input = json.dumps(
        {
            "prompt": "What is 3*3? Just give me the number.",
            "backend": "google",
            "persona": "default",
            "config": {
                "model": GEMINI_FLASH,  # Flash model
                "max_tokens": 50,
                "disable_mcp": True,
            },
        }
    )

    # Run the command
    cmd = ["think-agents"]
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

    # Verify model in start event
    start_event = events[0]
    assert start_event["model"] == GEMINI_FLASH

    # Verify the answer
    finish_event = events[-1]
    result_text = finish_event["result"].lower()
    assert (
        "9" in result_text or "nine" in result_text
    ), f"Expected '9' in response, got: {finish_event['result']}"
