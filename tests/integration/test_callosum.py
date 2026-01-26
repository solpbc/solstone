# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration tests for the Callosum message bus.

These tests use real servers, sockets, and I/O to validate core flows.
"""

import os
import threading
import time
from pathlib import Path

import pytest

from think.callosum import CallosumConnection, CallosumServer


@pytest.fixture
def journal_path(tmp_path):
    """Set up a temporary journal path."""
    journal = tmp_path / "journal"
    journal.mkdir()
    os.environ["JOURNAL_PATH"] = str(journal)
    yield journal
    # Cleanup
    if "JOURNAL_PATH" in os.environ:
        del os.environ["JOURNAL_PATH"]


@pytest.fixture
def callosum_server(journal_path):
    """Start a Callosum server in a background thread."""
    server = CallosumServer()

    # Start server in background thread
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    socket_path = journal_path / "health" / "callosum.sock"
    for _ in range(50):  # 5 seconds max
        if socket_path.exists():
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Server did not start in time")

    yield server

    # Stop server
    server.stop()
    server_thread.join(timeout=2)


def test_server_creates_socket(journal_path):
    """Test that server creates socket file in health directory."""
    server = CallosumServer()

    # Start server in background
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for socket to be created
    socket_path = journal_path / "health" / "callosum.sock"
    for _ in range(50):
        if socket_path.exists():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Socket file not created")

    assert socket_path.exists()

    # Stop server
    server.stop()
    server_thread.join(timeout=2)

    # Socket should be cleaned up
    assert not socket_path.exists()


def test_single_client_emit_and_listen(callosum_server):
    """Test single client emitting and listening."""
    received_messages = []

    def callback(message):
        received_messages.append(message)

    # Create connection with callback (listener)
    listener = CallosumConnection()
    listener.start(callback=callback)

    # Give listener time to connect
    time.sleep(0.2)

    # Create connection and emit
    client = CallosumConnection()
    client.start()

    # Give client time to connect
    time.sleep(0.2)

    client.emit("test", "hello", data="world")

    # Wait for message to be sent and received
    time.sleep(0.2)

    # Verify message received
    assert len(received_messages) == 1
    msg = received_messages[0]
    assert msg["tract"] == "test"
    assert msg["event"] == "hello"
    assert msg["data"] == "world"
    assert "ts" in msg  # Server should add timestamp

    # Cleanup
    listener.stop()
    client.stop()


def test_multiple_clients_broadcast(callosum_server):
    """Test that messages are broadcast to all listeners."""
    received_by_listener1 = []
    received_by_listener2 = []
    received_by_listener3 = []

    def callback1(msg):
        received_by_listener1.append(msg)

    def callback2(msg):
        received_by_listener2.append(msg)

    def callback3(msg):
        received_by_listener3.append(msg)

    # Create multiple listeners
    listener1 = CallosumConnection()
    listener1.start(callback=callback1)

    listener2 = CallosumConnection()
    listener2.start(callback=callback2)

    listener3 = CallosumConnection()
    listener3.start(callback=callback3)

    time.sleep(0.2)

    # Emit message from client
    client = CallosumConnection()
    client.start()

    # Wait for client to connect
    time.sleep(0.2)

    client.emit("cortex", "agent_start", agent_id="123", name="analyst")

    # Wait for broadcast
    time.sleep(0.2)

    # All listeners should receive the message
    assert len(received_by_listener1) == 1
    assert len(received_by_listener2) == 1
    assert len(received_by_listener3) == 1

    # Verify content
    for received in [
        received_by_listener1,
        received_by_listener2,
        received_by_listener3,
    ]:
        msg = received[0]
        assert msg["tract"] == "cortex"
        assert msg["event"] == "agent_start"
        assert msg["agent_id"] == "123"
        assert msg["name"] == "analyst"

    # Cleanup
    client.stop()
    listener1.stop()
    listener2.stop()
    listener3.stop()


def test_callosum_send_integration(callosum_server):
    """Test that callosum_send() works with real server."""
    from think.callosum import callosum_send

    received_messages = []

    def callback(msg):
        received_messages.append(msg)

    # Create listener
    listener = CallosumConnection()
    listener.start(callback=callback)

    time.sleep(0.2)

    # Send message using callosum_send
    result = callosum_send("test", "fire_and_forget", data="ephemeral")

    # Should succeed
    assert result is True

    # Wait for message to be broadcast
    time.sleep(0.2)

    # Verify message was received
    assert len(received_messages) == 1
    msg = received_messages[0]
    assert msg["tract"] == "test"
    assert msg["event"] == "fire_and_forget"
    assert msg["data"] == "ephemeral"

    # Cleanup
    listener.stop()
