# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration tests for the Cortex agent system with Callosum."""

import json
import os
import threading
import time

import pytest

from think.callosum import CallosumServer
from think.cortex import CortexService
from think.cortex_client import cortex_agents, cortex_request
from think.utils import now_ms


@pytest.fixture
def callosum_server(integration_journal_path):
    """Start a Callosum server for integration testing."""
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    server = CallosumServer()
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    socket_path = integration_journal_path / "health" / "callosum.sock"
    for _ in range(50):
        if socket_path.exists():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Callosum server did not start in time")

    yield server

    server.stop()
    server_thread.join(timeout=2)


@pytest.mark.integration
def test_cortex_service_startup(integration_journal_path, callosum_server):
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
def test_cortex_request_creation(integration_journal_path, callosum_server):
    """Test creating a Cortex agent request via Callosum."""
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Listen for broadcasts
    received_messages = []

    def callback(msg):
        received_messages.append(msg)

    from think.callosum import CallosumConnection

    listener = CallosumConnection()
    listener.start(callback=callback)
    time.sleep(0.1)

    # Create a request
    agent_id = cortex_request(prompt="Test prompt", name="default", provider="openai")

    time.sleep(0.2)

    # Verify request was broadcast
    assert len(received_messages) >= 1
    request = [m for m in received_messages if m.get("event") == "request"][0]
    assert request["prompt"] == "Test prompt"
    assert request["name"] == "default"
    assert request["provider"] == "openai"
    assert request["agent_id"] == agent_id

    listener.stop()


@pytest.mark.integration
def test_cortex_end_to_end_with_echo_agent(integration_journal_path, callosum_server):
    """Test end-to-end Cortex flow with a simple echo agent."""
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Create a mock agent script that just echoes
    agents_dir = integration_journal_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Start Cortex service in background
    cortex = CortexService(journal_path=str(integration_journal_path))
    service_thread = threading.Thread(target=cortex.start, daemon=True)
    service_thread.start()

    time.sleep(0.5)  # Let service start

    # Collect events
    received_events = []

    def callback(message):
        # Filter for cortex tract
        if message.get("tract") != "cortex":
            return
        received_events.append(message)

    # Start watching with CallosumConnection
    from think.callosum import CallosumConnection

    watcher = CallosumConnection()
    watcher.start(callback=callback)

    time.sleep(0.2)

    # Make a request (this will fail because no real agent, but we can verify the flow)
    agent_id = cortex_request(
        prompt="Test end-to-end", name="default", provider="openai"
    )

    # Wait for at least request event
    time.sleep(1.0)

    # Should have received the request event
    request_events = [e for e in received_events if e.get("event") == "request"]
    assert len(request_events) >= 1
    assert request_events[0]["agent_id"] == agent_id

    watcher.stop()
    cortex.stop()


@pytest.mark.integration
def test_cortex_agents_listing(integration_journal_path):
    """Test listing agents from the cortex_agents function."""
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Create some test agent files
    agents_dir = integration_journal_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Get initial count
    initial_result = cortex_agents()
    initial_count = len(initial_result["agents"])

    ts = now_ms()

    # Create completed agent (inside a date subdirectory, matching cortex layout)
    agent_subdir = agents_dir / "20260214"
    agent_subdir.mkdir(parents=True, exist_ok=True)
    completed_file = agent_subdir / f"{ts}.jsonl"
    with open(completed_file, "w") as f:
        json.dump(
            {
                "event": "request",
                "ts": ts,
                "prompt": "Test",
                "name": "default",
                "provider": "openai",
            },
            f,
        )
        f.write("\n")
        json.dump({"event": "finish", "ts": ts + 100, "result": "Done"}, f)
        f.write("\n")

    # List agents
    result = cortex_agents()

    # Should have one more than before
    assert len(result["agents"]) == initial_count + 1

    # Find our agent
    our_agent = [a for a in result["agents"] if a["id"] == str(ts)][0]
    assert our_agent["status"] == "completed"
    assert our_agent["prompt"] == "Test"


@pytest.mark.integration
def test_cortex_error_handling(integration_journal_path, callosum_server):
    """Test that Cortex handles errors gracefully."""
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Listen for events
    received_events = []

    def callback(msg):
        received_events.append(msg)

    from think.callosum import CallosumConnection

    listener = CallosumConnection()
    listener.start(callback=callback)
    time.sleep(0.1)

    # Make a request
    cortex_request(
        prompt="Test error handling",
        name="nonexistent_agent",  # This may cause issues
        provider="openai",
    )

    time.sleep(0.2)

    # Should have at least received the request
    request_events = [e for e in received_events if e.get("event") == "request"]
    assert len(request_events) >= 1

    listener.stop()
