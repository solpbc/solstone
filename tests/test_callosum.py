"""Tests for the Callosum message bus."""

import json
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
    listener = CallosumConnection(callback=callback)
    listener.connect()

    # Give listener time to start
    time.sleep(0.1)

    # Create connection and emit
    client = CallosumConnection()
    client.emit("test", "hello", data="world")

    # Wait for message
    time.sleep(0.1)

    # Verify message received
    assert len(received_messages) == 1
    msg = received_messages[0]
    assert msg["tract"] == "test"
    assert msg["event"] == "hello"
    assert msg["data"] == "world"
    assert "ts" in msg  # Server should add timestamp

    # Cleanup
    listener.close()
    client.close()


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
    listener1 = CallosumConnection(callback=callback1)
    listener1.connect()

    listener2 = CallosumConnection(callback=callback2)
    listener2.connect()

    listener3 = CallosumConnection(callback=callback3)
    listener3.connect()

    time.sleep(0.1)

    # Emit message from client
    client = CallosumConnection()
    client.emit("cortex", "agent_start", agent_id="123", persona="analyst")

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
        assert msg["persona"] == "analyst"

    # Cleanup
    client.close()
    listener1.close()
    listener2.close()
    listener3.close()


def test_multiple_emitters_to_multiple_listeners(callosum_server):
    """Test multiple clients emitting to multiple listeners."""
    received_by_listener1 = []
    received_by_listener2 = []

    # Create listeners
    listener1 = CallosumConnection(
        callback=lambda msg: received_by_listener1.append(msg)
    )
    listener1.connect()

    listener2 = CallosumConnection(
        callback=lambda msg: received_by_listener2.append(msg)
    )
    listener2.connect()

    time.sleep(0.1)

    # Create multiple clients and emit
    client1 = CallosumConnection()
    client2 = CallosumConnection()
    client3 = CallosumConnection()

    client1.emit("indexer", "scan_start", index_type="transcripts")
    client2.emit("cortex", "agent_finish", agent_id="456")
    client3.emit("supervisor", "process_exit", process_name="observer")

    # Wait for all messages
    time.sleep(0.2)

    # Both listeners should receive all 3 messages
    assert len(received_by_listener1) == 3
    assert len(received_by_listener2) == 3

    # Verify messages (order should be preserved)
    tracts1 = [msg["tract"] for msg in received_by_listener1]
    tracts2 = [msg["tract"] for msg in received_by_listener2]

    assert tracts1 == ["indexer", "cortex", "supervisor"]
    assert tracts2 == ["indexer", "cortex", "supervisor"]

    # Cleanup
    for client in [client1, client2, client3]:
        client.close()
    for listener in [listener1, listener2]:
        listener.close()


def test_client_reconnect_on_failure(callosum_server):
    """Test that client reconnects after connection failure."""
    client = CallosumConnection()

    # First emit should work
    client.emit("test", "first")

    # Simulate connection failure by closing socket
    if client.sock:
        client.sock.close()
        client.sock = None

    # Create listener BEFORE the reconnect to ensure clean state
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Next emit should reconnect and work
    client.emit("test", "reconnected")

    time.sleep(0.1)

    # Should only receive the reconnected message
    assert len(received) == 1
    assert received[0]["event"] == "reconnected"

    # Cleanup
    client.close()
    listener.close()


def test_invalid_message_without_tract(callosum_server):
    """Test that messages without tract field are rejected."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Send invalid message directly to socket (bypassing emit validation)
    client = CallosumConnection()
    client.connect()

    # Send message without tract
    invalid_msg = {"event": "test"}
    line = json.dumps(invalid_msg) + "\n"
    client.sock.sendall(line.encode("utf-8"))

    time.sleep(0.1)

    # Should not be broadcast
    assert len(received) == 0

    # Cleanup
    client.close()
    listener.close()


def test_invalid_message_without_event(callosum_server):
    """Test that messages without event field are rejected."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Send invalid message directly to socket
    client = CallosumConnection()
    client.connect()

    # Send message without event
    invalid_msg = {"tract": "test"}
    line = json.dumps(invalid_msg) + "\n"
    client.sock.sendall(line.encode("utf-8"))

    time.sleep(0.1)

    # Should not be broadcast
    assert len(received) == 0

    # Cleanup
    client.close()
    listener.close()


def test_client_emit_when_server_not_running(journal_path):
    """Test that client fails gracefully when server is not running."""
    # No server started
    client = CallosumConnection()

    # Should not raise exception, just fail silently
    client.emit("test", "no_server")

    # Connection should be None
    assert client.sock is None


def test_custom_timestamp_preserved(callosum_server):
    """Test that custom timestamp in message is preserved."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Send message with custom timestamp via raw socket
    client = CallosumConnection()
    client.connect()

    custom_ts = 1234567890
    msg = {"tract": "test", "event": "custom_ts", "ts": custom_ts}
    line = json.dumps(msg) + "\n"
    client.sock.sendall(line.encode("utf-8"))

    time.sleep(0.1)

    assert len(received) == 1
    assert received[0]["ts"] == custom_ts

    # Cleanup
    client.close()
    listener.close()


def test_multiple_sequential_messages(callosum_server):
    """Test sending multiple messages in sequence."""
    received = []
    listener = CallosumConnection(callback=lambda msg: received.append(msg))
    listener.connect()

    time.sleep(0.1)

    # Send multiple messages
    client = CallosumConnection()
    for i in range(10):
        client.emit("test", "message", seq=i)

    time.sleep(0.2)

    # All messages should be received in order
    assert len(received) == 10
    for i, msg in enumerate(received):
        assert msg["seq"] == i

    # Cleanup
    client.close()
    listener.close()


def test_listener_disconnect_doesnt_affect_others(callosum_server):
    """Test that one listener disconnecting doesn't affect others."""
    received_by_listener1 = []
    received_by_listener2 = []

    # Create two listeners
    listener1 = CallosumConnection(
        callback=lambda msg: received_by_listener1.append(msg)
    )
    listener1.connect()

    listener2 = CallosumConnection(
        callback=lambda msg: received_by_listener2.append(msg)
    )
    listener2.connect()

    time.sleep(0.1)

    # Send first message
    client = CallosumConnection()
    client.emit("test", "first")

    time.sleep(0.1)

    # Both should receive
    assert len(received_by_listener1) == 1
    assert len(received_by_listener2) == 1

    # Disconnect listener1
    listener1.close()

    # Give server time to clean up the disconnected client
    time.sleep(0.3)

    # Send second message
    client.emit("test", "second")

    time.sleep(0.1)

    # Only listener2 should receive the second message
    assert len(received_by_listener1) == 1  # Still 1
    assert len(received_by_listener2) == 2  # Now 2

    # Cleanup
    client.close()
    listener2.close()


def test_with_fixtures_journal():
    """Test using the actual fixtures journal directory."""
    # Use fixtures journal
    fixtures_journal = Path(__file__).parent.parent / "fixtures" / "journal"
    os.environ["JOURNAL_PATH"] = str(fixtures_journal)

    try:
        # Start server
        server = CallosumServer()
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()

        # Wait for socket
        socket_path = fixtures_journal / "health" / "callosum.sock"
        for _ in range(50):
            if socket_path.exists():
                break
            time.sleep(0.1)
        else:
            pytest.fail("Socket not created")

        # Test basic emit/listen
        received = []
        listener = CallosumConnection(callback=lambda msg: received.append(msg))
        listener.connect()

        time.sleep(0.1)

        client = CallosumConnection()
        client.emit("cortex", "test", message="using fixtures journal")

        time.sleep(0.1)

        assert len(received) == 1
        assert received[0]["message"] == "using fixtures journal"

        # Cleanup
        client.close()
        listener.close()

        server.stop()
        server_thread.join(timeout=2)

        # Socket should be cleaned up
        assert not socket_path.exists()

    finally:
        if "JOURNAL_PATH" in os.environ:
            del os.environ["JOURNAL_PATH"]
