# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration test for Anthropic provider with real API calls."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

from think.models import CLAUDE_SONNET_4


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
def test_anthropic_provider_basic():
    """Test Anthropic provider with basic prompt, no MCP."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["ANTHROPIC_API_KEY"] = api_key

    # Create NDJSON input (no mcp_server_url = no MCP tools)
    ndjson_input = json.dumps(
        {
            "prompt": "what is 1+1? Just give me the number.",
            "provider": "anthropic",
            "model": CLAUDE_SONNET_4,
            "name": "default",
            "max_output_tokens": 100,
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
    assert start_event["model"] == CLAUDE_SONNET_4
    assert start_event["name"] == "default"
    assert isinstance(start_event["ts"], int)

    # Check finish event
    finish_event = events[-1]

    # Check if this was an API error (intermittent failures)
    if finish_event.get("event") == "error":
        error_msg = finish_event.get("error", "Unknown error")
        if (
            "rate" in error_msg.lower()
            or "retry" in error_msg.lower()
            or "quota" in error_msg.lower()
        ):
            pytest.skip(f"Intermittent Anthropic API error: {error_msg}")
        else:
            pytest.fail(f"Unexpected error: {finish_event}")

    assert (
        finish_event["event"] == "finish"
    ), f"Expected finish event, got: {finish_event}"
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

    # Verify stderr has no errors (deprecation warnings from third-party libs are OK)
    if result.stderr:
        assert (
            "error" not in result.stderr.lower()
            or "deprecationwarning" in result.stderr.lower()
        ), f"Unexpected stderr content: {result.stderr}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_anthropic_provider_with_thinking():
    """Test Anthropic provider with thinking enabled."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    if not journal_path:
        pytest.skip("JOURNAL_PATH not found in fixtures/.env file")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["ANTHROPIC_API_KEY"] = api_key

    # Create NDJSON input with thinking config
    ndjson_input = json.dumps(
        {
            "prompt": "What is the square root of 16? Just the number please.",
            "provider": "anthropic",
            "name": "default",
            "model": CLAUDE_SONNET_4,
            "max_output_tokens": 2048,
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

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Parse events
    stdout_lines = result.stdout.strip().split("\n")
    events = [json.loads(line) for line in stdout_lines if line]

    # Check for thinking events (if the model supports it)
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    # If thinking events are present, verify they have the expected fields
    for te in thinking_events:
        assert "summary" in te, "Thinking event should have summary"
        assert "ts" in te, "Thinking event should have timestamp"
        # Signature should be present for normal thinking (not redacted)
        if te.get("summary") != "[redacted]":
            assert "signature" in te, "Thinking event should have signature"

    # Verify the answer is correct
    finish_event = events[-1]

    # Check if this was an API error (intermittent failures)
    if finish_event.get("event") == "error":
        error_msg = finish_event.get("error", "Unknown error")
        if (
            "rate" in error_msg.lower()
            or "retry" in error_msg.lower()
            or "quota" in error_msg.lower()
        ):
            pytest.skip(f"Intermittent Anthropic API error: {error_msg}")
        else:
            pytest.fail(f"Unexpected error: {finish_event}")

    assert (
        finish_event["event"] == "finish"
    ), f"Expected finish event, got: {finish_event}"
    result_text = finish_event["result"].lower()
    assert (
        "4" in result_text or "four" in result_text
    ), f"Expected '4' in response, got: {finish_event['result']}"


@pytest.mark.integration
@pytest.mark.requires_api
def test_anthropic_json_truncation_detection():
    """Test that Anthropic provider detects JSON response truncation via finish_reason.

    Uses a very small max_output_tokens to force truncation, verifying that
    the provider returns finish_reason='max_tokens' which callers can use
    to detect incomplete responses.
    """
    fixtures_env, api_key, _ = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in fixtures/.env file")

    # Import provider directly for this test
    from think.providers import anthropic as anthropic_provider

    # Request JSON output with very small token limit to force truncation
    # Use run_generate which returns GenerateResult, then check finish_reason
    result = anthropic_provider.run_generate(
        contents="Return a JSON array of the first 50 prime numbers.",
        model=CLAUDE_SONNET_4,
        json_output=True,
        max_output_tokens=10,  # Too small to complete the response
    )

    # Verify truncation was detected via finish_reason
    assert (
        result["finish_reason"] == "max_tokens"
    ), f"Expected max_tokens finish_reason, got: {result['finish_reason']}"
    # Partial text should be present
    assert isinstance(result["text"], str)
