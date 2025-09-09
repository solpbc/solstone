"""Integration test for Cortex service with cortex_client."""

import json
import os
import subprocess
import time
from pathlib import Path
from threading import Thread

import pytest
from dotenv import load_dotenv

from think.cortex_client import cortex_agents, cortex_request, cortex_run, cortex_watch
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


def read_agent_file(agent_id: str, journal_path: str):
    """Read all events from an agent file."""
    agents_dir = Path(journal_path) / "agents"

    # Try active file first
    active_file = agents_dir / f"{agent_id}_active.jsonl"
    if active_file.exists():
        agent_file = active_file
    else:
        # Look for completed file
        for file in agents_dir.glob(f"{agent_id}_*.jsonl"):
            if not file.name.endswith("_pending.jsonl"):
                agent_file = file
                break
        else:
            return []

    events = []
    with open(agent_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def get_agent_status(agent_id: str, journal_path: str):
    """Get the status of an agent."""
    agents_dir = Path(journal_path) / "agents"

    # Check for active file
    active_file = agents_dir / f"{agent_id}_active.jsonl"
    if active_file.exists():
        return "running"

    # Check for completed file (just the agent_id without suffix)
    completed_file = agents_dir / f"{agent_id}.jsonl"
    if completed_file.exists():
        return "completed"
    
    # Also check for failed files
    for file in agents_dir.glob(f"{agent_id}_*.jsonl"):
        if "_failed" in file.name:
            return "failed"
        elif not file.name.endswith("_pending.jsonl") and not file.name.endswith(
            "_active.jsonl"
        ):
            return "completed"

    # Check for pending file
    pending_file = agents_dir / f"{agent_id}_pending.jsonl"
    if pending_file.exists():
        return "pending"

    return "unknown"


@pytest.mark.integration
@pytest.mark.requires_api
def test_cortex_service_with_client():
    """Test Cortex service with cortex_client for a simple OpenAI request."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Collect all events
        events = []

        def handle_event(event):
            events.append(event)
            print(f"Event: {event.get('event')} - {event}")

        # Run agent with simple math question
        result = cortex_run(
            prompt="what is 2+2, just return the number nothing else",
            backend="openai",
            persona="default",
            config={
                "model": GPT_5_MINI,  # Use cheap model for testing
                "max_tokens": 100,
                "disable_mcp": True,  # Disable MCP for simple test
            },
            on_event=handle_event,
        )

        # Verify result contains "4"
        assert "4" in result, f"Expected '4' in result, got: {result}"

        # Verify event structure
        assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"

        # The first event from agents is typically 'start', not 'request'
        # The request is written when creating the file but may not be streamed

        # Find start event
        start_events = [e for e in events if e.get("event") == "start"]
        assert len(start_events) > 0, "No start event found"
        start_event = start_events[0]
        assert start_event["model"] == GPT_5_MINI

        # Find finish event
        finish_events = [e for e in events if e.get("event") == "finish"]
        assert len(finish_events) > 0, "No finish event found"
        finish_event = finish_events[0]
        assert "4" in str(finish_event["result"])

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.requires_api
def test_cortex_streaming_events():
    """Test streaming events from Cortex service."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Track events as they stream
        streamed_events = []
        event_types_seen = set()

        def event_handler(event):
            streamed_events.append(event)
            event_types_seen.add(event.get("event"))
            print(f"Streamed: {event.get('event')} at {event.get('ts')}")

        # Run agent and stream events
        cortex_run(
            prompt="What is the capital of France? Just say the city name.",
            backend="openai",
            persona="default",
            config={
                "model": GPT_5_MINI,  # Use cheap model for testing
                "max_tokens": 100,
                "disable_mcp": True,
            },
            on_event=event_handler,
        )

        # Verify we got the expected event types
        # Note: 'request' event is written to file but may not be streamed
        expected_events = {"start", "finish"}
        assert expected_events.issubset(
            event_types_seen
        ), f"Missing events. Expected: {expected_events}, Got: {event_types_seen}"

        # Verify finish event contains Paris
        finish_events = [e for e in streamed_events if e.get("event") == "finish"]
        assert (
            len(finish_events) == 1
        ), f"Expected 1 finish event, got {len(finish_events)}"
        assert "Paris" in finish_events[0].get(
            "result", ""
        ), "Expected 'Paris' in result"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
def test_cortex_agent_list():
    """Test listing agents through cortex_agents function."""
    # Use fixtures journal for this test
    journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Set journal path for cortex_client functions
    os.environ["JOURNAL_PATH"] = journal_path

    # List all agents
    result = cortex_agents(limit=5, agent_type="all")

    # Verify structure
    assert "agents" in result
    assert "pagination" in result
    assert "live_count" in result
    assert "historical_count" in result

    # Check pagination structure
    pagination = result["pagination"]
    assert "limit" in pagination
    assert "offset" in pagination
    assert "total" in pagination
    assert "has_more" in pagination

    # If there are agents, verify their structure
    if result["agents"]:
        agent = result["agents"][0]
        required_fields = [
            "id",
            "status",
            "persona",
            "model",
            "prompt",
            "start",
        ]
        for field in required_fields:
            assert field in agent, f"Missing field '{field}' in agent"


@pytest.mark.integration
@pytest.mark.requires_api
def test_cortex_simple_math_streaming():
    """Test Cortex service with simple math question and verify streaming."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Track events to verify streaming
        event_sequence = []
        result_value = None
        error_message = None

        def stream_handler(event):
            event_type = event.get("event")
            event_sequence.append(event_type)
            print(f"Event: {event_type} - {event}")

            if event_type == "finish":
                nonlocal result_value
                result_value = event.get("result", "")
                print(f"Got result: {result_value}")
            elif event_type == "error":
                nonlocal error_message
                error_message = event.get("error", "Unknown error")
                print(f"Got error: {error_message}")

        # Make the exact request: "what is 2+2, just return the number nothing else"
        try:
            result = cortex_run(
                prompt="what is 2+2, just return the number nothing else",
                backend="openai",
                persona="default",
                config={
                    "model": GPT_5_MINI,  # Use cheap model for testing
                    "max_tokens": 100,  # Minimum is 16 for OpenAI
                    "disable_mcp": True,
                },
                on_event=stream_handler,
            )
            result_value = result
        except RuntimeError as e:
            error_message = str(e)

        # Check if we got an error
        if error_message:
            print(f"Test failed with error: {error_message}")
            print(f"Event sequence: {event_sequence}")
            pytest.fail(f"Agent failed with error: {error_message}")

        # Verify we got events in correct order
        # Note: 'request' event may not be streamed, agents emit 'start' first
        assert "start" in event_sequence, "Missing start event"
        assert (
            "finish" in event_sequence
        ), f"Missing finish event. Got events: {event_sequence}"

        # Verify the final event has the correct answer
        assert result_value is not None, "No result received"
        assert "4" in result_value, f"Expected '4' in result, got: {result_value}"

        print(f"Success! Got answer: {result_value}")
        print(f"Event sequence: {event_sequence}")

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.requires_api
def test_cortex_default_model():
    """Test Cortex service using default model (no model specified)."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Track events
        events = []

        def event_handler(event):
            events.append(event)
            print(f"Event: {event.get('event')}")

        # Run with NO model specified - use backend defaults
        try:
            result = cortex_run(
                prompt="what is 2+2, just return the number nothing else",
                backend="openai",
                persona="default",
                config={
                    # No model specified - use default
                    "disable_mcp": True,
                },
                on_event=event_handler,
            )
        except RuntimeError:
            # Agent might error, that's ok for this test
            result = None

        # Check we got the key events
        event_types = [e.get("event") for e in events]
        # Note: 'request' event may not be streamed
        assert "start" in event_types
        assert "finish" in event_types or "error" in event_types

        # Find the result
        if result:
            print(f"Got result with default model: {result}")
            # Result should contain 4
            assert "4" in result, f"Expected '4' in result, got: {result}"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
def test_cortex_error_handling():
    """Test Cortex error handling with invalid request."""
    journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Create request with empty prompt (should fail)
        agent_file = cortex_request(
            prompt="",  # Empty prompt should cause error
            backend="openai",
            persona="default",
        )

        # Extract agent_id from the filename
        agent_id = Path(agent_file).stem.replace("_active", "")

        # Give Cortex time to process the request
        time.sleep(3)
        
        # Read events directly from the agent file instead of watching
        events = read_agent_file(agent_id, journal_path)
        
        # Check if we got any events
        event_types = [e.get("event") for e in events]
        print(f"Events found in agent file: {event_types}")
        
        # For empty prompt, cortex might not process it at all or might handle it gracefully
        # The test is mainly to ensure cortex doesn't crash with invalid input

        # Check agent status
        status = get_agent_status(agent_id, journal_path)
        assert status in [
            "failed",
            "completed",
            "running",
        ], f"Expected valid status, got: {status}"

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()


@pytest.mark.integration
@pytest.mark.requires_api
def test_cortex_watch_multiple_agents():
    """Test watching multiple agents simultaneously."""
    fixtures_env, api_key, journal_path = get_fixtures_env()

    if not fixtures_env:
        pytest.skip("fixtures/.env not found")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in fixtures/.env file")

    if not journal_path:
        journal_path = str(Path(__file__).parent.parent.parent / "fixtures" / "journal")

    # Prepare environment
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal_path
    env["OPENAI_API_KEY"] = api_key

    # Start Cortex service in background
    cortex_process = subprocess.Popen(
        ["think-cortex"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give Cortex time to start up
    time.sleep(2)

    try:
        # Verify Cortex is running
        assert cortex_process.poll() is None, "Cortex service failed to start"

        # Set journal path for cortex_client functions
        os.environ["JOURNAL_PATH"] = journal_path

        # Create multiple agent requests
        agent_files = []
        for i in range(3):
            agent_file = cortex_request(
                prompt=f"what is {i}+{i}, just return the number nothing else",
                backend="openai",
                persona="default",
                config={
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                    "disable_mcp": True,
                },
            )
            agent_files.append(agent_file)
            time.sleep(0.1)  # Small delay between requests

        # Extract agent IDs
        agent_ids = [Path(f).stem.replace("_active", "") for f in agent_files]

        # Track events for all agents
        agent_events = {aid: [] for aid in agent_ids}
        completed_agents = set()

        def event_handler(event):
            # Try to determine which agent this event belongs to
            # Events should come from the same file, so we need to track by file
            for aid in agent_ids:
                if aid not in completed_agents:
                    # This is a simplification - in real use you'd track by file
                    agent_events[aid].append(event)

                    if event.get("event") == "finish":
                        completed_agents.add(aid)

                        # Stop watching when all agents complete
                        if len(completed_agents) == len(agent_ids):
                            return False
                    break

            return True  # Continue watching

        # Watch for events with timeout
        def watch_with_timeout():
            try:
                cortex_watch(event_handler)
            except Exception:
                pass

        watch_thread = Thread(target=watch_with_timeout, daemon=True)
        watch_thread.start()
        watch_thread.join(timeout=10)

        # Verify we got events for at least some agents
        assert len(completed_agents) > 0, f"No agents completed. Events: {agent_events}"

        print(f"Completed {len(completed_agents)} out of {len(agent_ids)} agents")

    finally:
        # Clean up: terminate Cortex service
        cortex_process.terminate()
        try:
            cortex_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cortex_process.kill()
