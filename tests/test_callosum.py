"""Unit tests for the Callosum message bus.

These tests use mocks to test logic in isolation without real I/O.
"""

import json
import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
    server.clients = [Mock()]

    # Message without tract should be rejected
    invalid_msg = {"event": "test"}
    server.broadcast(invalid_msg)

    # Client should not receive anything
    server.clients[0].sendall.assert_not_called()


def test_server_broadcast_validates_event_field():
    """Test that messages without event field are rejected."""
    server = CallosumServer()
    server.clients = [Mock()]

    # Message without event should be rejected
    invalid_msg = {"tract": "test"}
    server.broadcast(invalid_msg)

    # Client should not receive anything
    server.clients[0].sendall.assert_not_called()


def test_server_broadcast_adds_timestamp():
    """Test that server adds timestamp if not present."""
    server = CallosumServer()
    mock_client = Mock()
    server.clients = [mock_client]

    # Valid message without timestamp
    msg = {"tract": "test", "event": "hello"}

    with patch("time.time", return_value=1234567.890):
        server.broadcast(msg)

    # Should have called sendall with message including timestamp
    mock_client.sendall.assert_called_once()
    sent_data = mock_client.sendall.call_args[0][0]
    sent_msg = json.loads(sent_data.decode("utf-8"))

    assert sent_msg["tract"] == "test"
    assert sent_msg["event"] == "hello"
    assert sent_msg["ts"] == 1234567890  # milliseconds


def test_server_broadcast_preserves_custom_timestamp():
    """Test that custom timestamp in message is preserved."""
    server = CallosumServer()
    mock_client = Mock()
    server.clients = [mock_client]

    custom_ts = 9999999999
    msg = {"tract": "test", "event": "hello", "ts": custom_ts}

    server.broadcast(msg)

    # Should preserve custom timestamp
    sent_data = mock_client.sendall.call_args[0][0]
    sent_msg = json.loads(sent_data.decode("utf-8"))
    assert sent_msg["ts"] == custom_ts


def test_server_broadcast_removes_dead_clients():
    """Test that broadcast removes clients that fail to receive."""
    server = CallosumServer()

    # Create mock clients - one working, one dead
    working_client = Mock()
    dead_client = Mock()
    dead_client.sendall.side_effect = Exception("Connection broken")

    server.clients = [working_client, dead_client]

    msg = {"tract": "test", "event": "hello"}
    server.broadcast(msg)

    # Dead client should be removed
    assert working_client in server.clients
    assert dead_client not in server.clients
    assert len(server.clients) == 1

    # Dead client socket should be closed
    dead_client.close.assert_called_once()


def test_client_emit_requires_connect_called():
    """Test that emit() requires connect() to be called first."""
    client = CallosumConnection()

    # emit() should raise RuntimeError if connect() was never called
    with pytest.raises(RuntimeError, match="Must call connect\\(\\) before emit\\(\\)"):
        client.emit("test", "hello")


def test_client_emit_graceful_when_disconnected():
    """Test that emit() logs and returns silently if connection drops."""
    client = CallosumConnection()

    # Simulate that connect() was called (set receive_thread)
    client.receive_thread = Mock()
    client.sock = None  # But connection is now dead

    # Should not raise, just log
    with patch("think.callosum.logger") as mock_logger:
        client.emit("test", "hello")
        mock_logger.info.assert_called_once()
        assert "Not connected" in mock_logger.info.call_args[0][0]


def test_client_emit_handles_socket_errors():
    """Test that emit() handles socket errors gracefully."""
    client = CallosumConnection()

    # Simulate connected state
    client.receive_thread = Mock()
    mock_sock = Mock()
    mock_sock.sendall.side_effect = Exception("Broken pipe")
    client.sock = mock_sock

    # Should not raise, just mark connection as dead
    with patch("think.callosum.logger") as mock_logger:
        client.emit("test", "hello")
        mock_logger.info.assert_called()
        assert "Failed to emit" in mock_logger.info.call_args[0][0]

    # Connection should be marked dead
    assert client.sock is None


def test_client_emit_sends_valid_message():
    """Test that emit() sends properly formatted JSON message."""
    client = CallosumConnection()

    # Setup connected state
    client.receive_thread = Mock()
    mock_sock = Mock()
    client.sock = mock_sock

    client.emit("test", "hello", data="world", count=42)

    # Verify sendall was called with correct JSON
    mock_sock.sendall.assert_called_once()
    sent_data = mock_sock.sendall.call_args[0][0]
    sent_msg = json.loads(sent_data.decode("utf-8").strip())

    assert sent_msg["tract"] == "test"
    assert sent_msg["event"] == "hello"
    assert sent_msg["data"] == "world"
    assert sent_msg["count"] == 42


def test_client_connect_retries_on_failure(journal_path):
    """Test that connect() retries when socket connection fails."""
    client = CallosumConnection()

    attempt_count = [0]

    def mock_connect_side_effect(*args):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionRefusedError("Not ready yet")
        # Succeed on 3rd attempt
        return None

    with patch("socket.socket") as mock_socket_class:
        mock_sock = Mock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = mock_connect_side_effect

        # Speed up retry for testing
        with patch("time.sleep"):
            client.connect(retry_delay=0.01)

    # Should have succeeded after 3 attempts
    assert attempt_count[0] == 3
    assert client.sock is not None
    assert client.receive_thread is not None


def test_client_connect_idempotent(journal_path):
    """Test that calling connect() multiple times is safe."""
    with patch("socket.socket") as mock_socket_class:
        mock_sock = Mock()
        mock_socket_class.return_value = mock_sock

        client = CallosumConnection()
        client.connect()

        first_sock = client.sock
        first_thread = client.receive_thread

        # Call connect again
        client.connect()

        # Should still have same connection (not reconnected)
        assert client.sock is first_sock
        assert client.receive_thread is first_thread


def test_client_close_stops_receive_thread(journal_path):
    """Test that close() stops the receive thread."""
    client = CallosumConnection()

    # Setup connected state
    mock_thread = Mock()
    mock_thread.is_alive.return_value = False
    client.receive_thread = mock_thread
    client.sock = Mock()

    client.close()

    # Should join the thread
    mock_thread.join.assert_called_once_with(timeout=2)

    # Should close socket
    assert client.sock is None


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
