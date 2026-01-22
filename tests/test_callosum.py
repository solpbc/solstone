# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for the Callosum message bus.

These tests use mocks to test logic in isolation without real I/O.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

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


def test_server_broadcast_validates_tract_field():
    """Test that messages without tract field are rejected."""
    server = CallosumServer()

    # Message without tract should be rejected and return False
    invalid_msg = {"event": "test"}
    result = server.broadcast(invalid_msg)

    assert result is False
    # Should not be queued
    assert server.broadcast_queue.qsize() == 0


def test_server_broadcast_validates_event_field():
    """Test that messages without event field are rejected."""
    server = CallosumServer()

    # Message without event should be rejected and return False
    invalid_msg = {"tract": "test"}
    result = server.broadcast(invalid_msg)

    assert result is False
    # Should not be queued
    assert server.broadcast_queue.qsize() == 0


def test_server_broadcast_adds_timestamp():
    """Test that server adds timestamp if not present."""
    server = CallosumServer()

    # Valid message without timestamp
    msg = {"tract": "test", "event": "hello"}

    with patch("think.callosum.time.time", return_value=1234567.890):
        result = server.broadcast(msg)

    assert result is True
    # Message should be queued with timestamp added
    queued_msg = server.broadcast_queue.get_nowait()
    assert queued_msg["tract"] == "test"
    assert queued_msg["event"] == "hello"
    assert queued_msg["ts"] == 1234567890  # milliseconds


def test_server_broadcast_preserves_custom_timestamp():
    """Test that custom timestamp in message is preserved."""
    server = CallosumServer()

    custom_ts = 9999999999
    msg = {"tract": "test", "event": "hello", "ts": custom_ts}

    result = server.broadcast(msg)

    assert result is True
    # Should preserve custom timestamp
    queued_msg = server.broadcast_queue.get_nowait()
    assert queued_msg["ts"] == custom_ts


def test_server_broadcast_removes_dead_clients():
    """Test that _send_to_clients removes clients that fail to receive."""
    server = CallosumServer()

    # Create mock clients - one working, one dead
    working_client = Mock()
    dead_client = Mock()
    dead_client.sendall.side_effect = Exception("Connection broken")
    dead_client.settimeout = Mock()
    working_client.settimeout = Mock()

    server.clients = [working_client, dead_client]

    # Call _send_to_clients directly (the method used by _writer_loop)
    msg = {"tract": "test", "event": "hello", "ts": 12345}
    server._send_to_clients(msg)

    # Dead client should be removed
    assert working_client in server.clients
    assert dead_client not in server.clients
    assert len(server.clients) == 1

    # Dead client socket should be closed
    dead_client.close.assert_called_once()


def test_client_emit_returns_false_when_not_started():
    """Test that emit() returns False and logs warning if start() not called yet."""
    client = CallosumConnection()

    # emit() should return False and log when thread not started
    with patch("think.callosum.logger") as mock_logger:
        result = client.emit("test", "hello")
        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Thread not running" in mock_logger.warning.call_args[0][0]


def test_client_emit_queues_message():
    """Test that emit() queues message when thread is running."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    client.thread = mock_thread

    result = client.emit("test", "hello", data="world", count=42)

    assert result is True
    # Message should be in queue
    assert client.send_queue.qsize() == 1
    msg = client.send_queue.get_nowait()
    assert msg["tract"] == "test"
    assert msg["event"] == "hello"
    assert msg["data"] == "world"
    assert msg["count"] == 42


def test_client_emit_returns_false_when_queue_full():
    """Test that emit() returns False when queue is full."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    client.thread = mock_thread

    # Fill the queue
    for i in range(1000):
        client.send_queue.put({"tract": "test", "event": f"msg{i}"})

    # Next emit should fail
    with patch("think.callosum.logger") as mock_logger:
        result = client.emit("test", "overflow")
        assert result is False
        mock_logger.warning.assert_called()
        assert "Queue full" in mock_logger.warning.call_args[0][0]


def test_client_start_creates_thread():
    """Test that start() creates and starts background thread."""
    client = CallosumConnection()

    def callback(msg):
        pass

    client.start(callback=callback)

    assert client.thread is not None
    assert client.thread.is_alive()
    assert client.callback is callback

    # Cleanup
    client.stop()


def test_client_start_idempotent():
    """Test that calling start() multiple times is safe."""
    client = CallosumConnection()

    client.start()
    first_thread = client.thread

    # Call start again
    client.start()

    # Should still have same thread (not restarted)
    assert client.thread is first_thread

    # Cleanup
    client.stop()


def test_client_stop_stops_thread():
    """Test that stop() stops the background thread."""
    client = CallosumConnection()

    # Setup running thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = False
    client.thread = mock_thread

    client.stop()

    # Should set stop event and join thread
    assert client.stop_event.is_set()
    mock_thread.join.assert_called_once_with(timeout=0.5)


def test_server_socket_path_from_env(journal_path):
    """Test that server uses JOURNAL_PATH env var for socket path."""
    server = CallosumServer()

    expected_path = journal_path / "health" / "callosum.sock"
    assert server.socket_path == expected_path


def test_server_socket_path_custom():
    """Test that server accepts custom socket path."""
    custom_path = Path("/tmp/custom.sock")
    server = CallosumServer(socket_path=custom_path)

    assert server.socket_path == custom_path


def test_client_socket_path_from_env(journal_path):
    """Test that client uses JOURNAL_PATH env var for socket path."""
    client = CallosumConnection()

    expected_path = journal_path / "health" / "callosum.sock"
    assert client.socket_path == expected_path


def test_client_socket_path_custom():
    """Test that client accepts custom socket path."""
    custom_path = Path("/tmp/custom.sock")
    client = CallosumConnection(socket_path=custom_path)

    assert client.socket_path == custom_path


def test_callosum_send_uses_default_path_when_journal_path_unset():
    """Test that callosum_send() uses default path when JOURNAL_PATH unset."""
    from think.callosum import callosum_send

    # Temporarily remove JOURNAL_PATH
    old_path = os.environ.get("JOURNAL_PATH")
    if "JOURNAL_PATH" in os.environ:
        del os.environ["JOURNAL_PATH"]

    try:
        # callosum_send will use the platform default path via get_journal()
        # Result depends on whether a server is listening at that path
        result = callosum_send("test", "event", data="value")
        # Just verify it returns a boolean (doesn't raise due to missing path)
        assert isinstance(result, bool)
    finally:
        # Restore JOURNAL_PATH
        if old_path:
            os.environ["JOURNAL_PATH"] = old_path


def test_callosum_send_with_custom_path():
    """Test that callosum_send() accepts custom socket path."""
    from think.callosum import callosum_send

    # Use non-existent socket - should return False but not crash
    custom_path = Path("/tmp/nonexistent_callosum.sock")
    result = callosum_send("test", "event", socket_path=custom_path, data="value")

    # Should fail gracefully (no server listening)
    assert result is False
