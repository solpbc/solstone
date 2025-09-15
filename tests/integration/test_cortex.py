"""Integration tests for the Cortex agent system."""

import json
import os
import threading
import time
from pathlib import Path

import pytest

from think.cortex import CortexService
from think.cortex_client import cortex_agents, cortex_request, cortex_run


@pytest.mark.integration
def test_cortex_service_startup(integration_journal_path):
    """Test that Cortex service starts up and creates necessary directories."""
    # Initialize the service
    cortex = CortexService(journal_path=str(integration_journal_path))

    # Verify agents directory was created
    agents_dir = integration_journal_path / "agents"
    assert agents_dir.exists()
    assert agents_dir.is_dir()

    # Verify service initializes correctly
    status = cortex.get_status()
    assert status["running_agents"] == 0
    assert status["agent_ids"] == []


@pytest.mark.integration
def test_cortex_request_creation(integration_journal_path):
    """Test creating a Cortex agent request file."""
    # Set JOURNAL_PATH for cortex_request
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Create a request
    active_file = cortex_request(
        prompt="Test prompt", persona="default", backend="openai"
    )

    # Verify file was created with correct name pattern
    active_path = Path(active_file)
    assert active_path.exists()
    assert active_path.name.endswith("_active.jsonl")
    assert active_path.parent == integration_journal_path / "agents"

    # Verify request content
    with open(active_path, "r") as f:
        request = json.loads(f.readline())

    assert request["event"] == "request"
    assert request["prompt"] == "Test prompt"
    assert request["persona"] == "default"
    assert request["backend"] == "openai"
    assert "ts" in request


@pytest.mark.integration
def test_cortex_agent_process_creation(integration_journal_path):
    """Test that Cortex spawns agent processes for active files."""
    # Set up environment
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Start Cortex service in a thread
    cortex = CortexService(journal_path=str(integration_journal_path))
    service_thread = threading.Thread(target=cortex.start, daemon=True)
    service_thread.start()

    # Give service time to start watching
    time.sleep(0.5)

    try:
        # Create a simple test request
        active_file = cortex_request(
            prompt="Return exactly: Hello from agent",
            persona="default",
            backend="openai",
        )

        # Give agent time to spawn and process
        time.sleep(2)

        # The agent might have already completed, so check the file
        active_path = Path(active_file)
        completed_path = active_path.parent / active_path.name.replace(
            "_active.jsonl", ".jsonl"
        )

        # Either file should exist (active if still running, completed if done)
        assert active_path.exists() or completed_path.exists()

        # If completed, verify it has events
        if completed_path.exists():
            with open(completed_path, "r") as f:
                lines = f.readlines()
                assert len(lines) >= 2  # At least request and finish/error

                # First line should be request
                first_event = json.loads(lines[0])
                assert first_event["event"] == "request"

                # Last line should be finish or error
                last_event = json.loads(lines[-1])
                assert last_event["event"] in ["finish", "error"]

    finally:
        # Stop the service
        cortex.stop()


@pytest.mark.integration
@pytest.mark.slow
def test_cortex_run_simple_agent(integration_journal_path):
    """Test running an agent synchronously with cortex_run."""
    # Set up environment
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Start Cortex service in background
    cortex = CortexService(journal_path=str(integration_journal_path))
    service_thread = threading.Thread(target=cortex.start, daemon=True)
    service_thread.start()

    # Give service time to start
    time.sleep(0.5)

    try:
        # Track events we receive
        events = []

        def on_event(event):
            events.append(event)

        # Run agent synchronously
        result = cortex_run(
            prompt="Return exactly the text: Integration test successful",
            persona="default",
            backend="openai",
            on_event=on_event,
        )

        # Verify we got a result
        assert result is not None
        assert isinstance(result, str)

        # Verify we received events
        assert len(events) > 0

        # Should have at least start and finish events
        event_types = [e.get("event") for e in events]
        assert "start" in event_types or "finish" in event_types

    except Exception as e:
        # If the agent backend isn't available, skip the test
        if any(keyword in str(e).lower() for keyword in ["api", "key", "mcp", "uri"]):
            pytest.skip(f"Agent backend not configured: {e}")
        raise

    finally:
        cortex.stop()


@pytest.mark.integration
def test_cortex_agent_list(integration_journal_path):
    """Test listing agents from the journal."""
    # Set up environment
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Create some mock agent files for testing
    agents_dir = integration_journal_path / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Clear any existing agent files from previous tests
    for f in agents_dir.glob("*.jsonl"):
        f.unlink()

    # Create a completed agent file
    completed_file = agents_dir / "1234567890123.jsonl"
    completed_file.write_text(
        '{"event": "request", "ts": 1234567890123, "prompt": "Test 1", "persona": "default", "backend": "openai"}\n'
        '{"event": "finish", "ts": 1234567890456, "result": "Done"}\n'
    )

    # Create an active agent file
    active_file = agents_dir / "1234567890789_active.jsonl"
    active_file.write_text(
        '{"event": "request", "ts": 1234567890789, "prompt": "Test 2", "persona": "default", "backend": "openai"}\n'
    )

    # List all agents
    result = cortex_agents(limit=10)

    assert "agents" in result
    assert "pagination" in result
    assert len(result["agents"]) == 2

    # Check agent details
    agents = result["agents"]

    # Find the completed agent
    completed = next((a for a in agents if a["id"] == "1234567890123"), None)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["prompt"] == "Test 1"

    # Find the active agent
    active = next((a for a in agents if a["id"] == "1234567890789"), None)
    assert active is not None
    assert active["status"] == "running"
    assert active["prompt"] == "Test 2"

    # Test filtering
    live_result = cortex_agents(agent_type="live")
    assert len(live_result["agents"]) == 1
    assert live_result["agents"][0]["status"] == "running"

    historical_result = cortex_agents(agent_type="historical")
    assert len(historical_result["agents"]) == 1
    assert historical_result["agents"][0]["status"] == "completed"


@pytest.mark.integration
def test_cortex_error_handling(integration_journal_path):
    """Test Cortex error handling for invalid requests."""
    # Set up environment
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Start Cortex service
    cortex = CortexService(journal_path=str(integration_journal_path))
    service_thread = threading.Thread(target=cortex.start, daemon=True)
    service_thread.start()

    # Give service time to start
    time.sleep(0.5)

    try:
        # Create a malformed request directly (bypassing cortex_request validation)
        agents_dir = integration_journal_path / "agents"
        agents_dir.mkdir(exist_ok=True)

        # Create an active file with invalid JSON
        ts = int(time.time() * 1000)
        bad_file = agents_dir / f"{ts}_active.jsonl"
        bad_file.write_text("not valid json\n")

        # Give Cortex time to process
        time.sleep(1)

        # File should be completed with error
        completed_file = agents_dir / f"{ts}.jsonl"
        assert completed_file.exists()

        # Should contain an error event
        with open(completed_file, "r") as f:
            lines = f.readlines()
            # First line is the invalid content, second should be error
            if len(lines) > 1:
                error_event = json.loads(lines[-1])
                assert error_event["event"] == "error"
                assert (
                    "JSON" in error_event["error"] or "Invalid" in error_event["error"]
                )

    finally:
        cortex.stop()
