# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for OpenAI provider with real API calls."""

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
def test_openai_provider_basic():
    """Test OpenAI provider with basic prompt, no MCP."""
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
            "provider": "openai",
            "name": "default",
            "model": GPT_5_MINI,  # Use cheap model for testing
            "max_output_tokens": 100,
            "disable_mcp": True,
        }
    )

    # Run the sol agents command
    cmd = ["sol", "agents"]
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
    assert start_event["name"] == "default"
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
def test_openai_provider_with_reasoning():
    """Test OpenAI provider with reasoning model to verify thinking summaries.

    Uses GPT-5-mini which supports reasoning with summary="detailed" config.
    The key test is that:
    1. The request succeeds (reasoning config is valid)
    2. We may receive thinking events with summaries (model-dependent)
    3. If thinking events are present, they have the expected structure
    """
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

    # Use GPT-5-mini which supports reasoning summaries
    # Use a prompt that encourages step-by-step reasoning
    ndjson_input = json.dumps(
        {
            "prompt": "If I have 3 apples and buy 5 more, then give away 2, how many do I have? Think through this step by step.",
            "provider": "openai",
            "name": "default",
            "model": GPT_5_MINI,
            "max_output_tokens": 500,
            "disable_mcp": True,
        }
    )

    # Run the sol agents command
    cmd = ["sol", "agents"]
    result = subprocess.run(
        cmd,
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=30,  # Increased for reasoning
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Parse events
    stdout_lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in stdout_lines if line]

    # Verify no errors
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 0, f"Found error events: {error_events}"

    # Check for thinking events - GPT-5 series should produce these
    # when reasoning config is properly set
    thinking_events = [e for e in events if e.get("event") == "thinking"]

    # If we have thinking events, verify their structure
    for thinking in thinking_events:
        assert "summary" in thinking, f"Thinking event missing 'summary': {thinking}"
        assert isinstance(
            thinking["summary"], str
        ), f"Thinking summary should be string: {thinking}"
        assert len(thinking["summary"]) > 0, "Thinking summary should not be empty"
        assert "model" in thinking, f"Thinking event missing 'model': {thinking}"
        assert "ts" in thinking, f"Thinking event missing 'ts': {thinking}"
        assert isinstance(thinking["ts"], int), "Timestamp should be int"

    # Verify the answer is correct (6 apples: 3 + 5 - 2 = 6)
    finish_event = events[-1]
    assert finish_event["event"] == "finish"
    result_text = finish_event["result"].lower()
    assert (
        "6" in result_text or "six" in result_text
    ), f"Expected '6' in response, got: {finish_event['result']}"

    # Log whether we got thinking events for debugging
    print(f"Received {len(thinking_events)} thinking events")


@pytest.mark.integration
@pytest.mark.requires_api
def test_openai_provider_with_extra_context():
    """Test OpenAI provider with extra_context to verify Responses API format.

    This exercises the session.add_items() path that was broken when content type
    was 'text' instead of 'input_text'. The key assertion is that we don't get
    the 400 error about invalid content type.
    """
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

    # Include extra_context like get_agent() does in production
    # This exercises the _convert_turns_to_items() code path
    ndjson_input = json.dumps(
        {
            "prompt": "What project was mentioned in the context above? Just the name.",
            "provider": "openai",
            "name": "default",
            "model": GPT_5_MINI,
            "max_output_tokens": 50,
            "disable_mcp": True,
            "extra_context": "## Project Context\nYou are working on Project Moonshot.",
        }
    )

    # Run the sol agents command
    cmd = ["sol", "agents"]
    result = subprocess.run(
        cmd,
        env=env,
        input=ndjson_input,
        capture_output=True,
        text=True,
        timeout=15,
    )

    # Parse stdout events
    stdout_lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in stdout_lines if line]

    # The critical check: no 400 error about invalid content type
    # This was the original bug - using 'text' instead of 'input_text'
    error_events = [e for e in events if e.get("event") == "error"]
    for err in error_events:
        error_msg = err.get("error", "")
        assert (
            "Invalid value: 'text'" not in error_msg
        ), f"Got content type format error - regression! Error: {error_msg}"
        assert (
            "input_text" not in error_msg or "Supported values" not in error_msg
        ), f"Got content type format error - regression! Error: {error_msg}"

    # Verify we got past the format validation (start event was emitted)
    start_events = [e for e in events if e.get("event") == "start"]
    assert len(start_events) == 1, "Should have start event"

    # If we get a finish event, verify the response references the context
    finish_events = [e for e in events if e.get("event") == "finish"]
    if finish_events:
        result_text = finish_events[0].get("result", "").lower()
        assert (
            "moonshot" in result_text
        ), f"Expected 'moonshot' in response, got: {finish_events[0].get('result')}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_openai_json_truncation_error():
    """Test that OpenAI provider raises IncompleteJSONError when JSON response is truncated.

    Uses a small max_output_tokens to force truncation, verifying that
    the provider correctly detects non-stop finish reasons and raises an error
    with the partial text available for debugging.
    """
    fixtures_env, api_key, _ = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    # Import provider directly for this test
    from think.models import IncompleteJSONError
    from think.providers import openai as openai_provider

    # Request JSON output with small token limit to force truncation
    with pytest.raises(IncompleteJSONError) as exc_info:
        openai_provider.generate(
            contents="Return a JSON array of the first 50 prime numbers.",
            model=GPT_5_MINI,
            json_output=True,
            max_output_tokens=50,  # Too small to complete the response
        )

    # Verify error message and partial_text attribute
    assert "JSON response incomplete" in str(exc_info.value)
    assert exc_info.value.reason is not None
    assert isinstance(exc_info.value.partial_text, str)
