import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages = []
        self.received = []
        self.connected = True
        self.closed = False
        
    def send(self, message):
        if self.closed:
            from simple_websocket import ConnectionClosed
            raise ConnectionClosed()
        self.messages.append(message)
        
    def receive(self, timeout=None):
        if self.received:
            return self.received.pop(0)
        return None
        
    def close(self):
        self.closed = True
        self.connected = False
        
    def add_received(self, message):
        self.received.append(message)


@pytest.fixture
def mock_journal(tmp_path, monkeypatch):
    """Set up a temporary journal directory."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    agents_path = journal_path / "agents"
    agents_path.mkdir()
    
    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))
    return journal_path


@pytest.fixture
def cortex_server():
    """Create a CortexServer instance for testing."""
    from think.cortex import CortexServer
    return CortexServer("/test/ws")


def test_running_agent_creation():
    """Test RunningAgent class initialization and methods."""
    from think.cortex import RunningAgent
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Running
    mock_process.pid = 12345
    
    log_path = Path("/tmp/test.jsonl")
    agent = RunningAgent("123456789", mock_process, log_path)
    
    assert agent.agent_id == "123456789"
    assert agent.process == mock_process
    assert agent.log_path == log_path
    assert agent.status == "running"
    assert agent.is_running() is True
    
    # Test to_dict
    agent_dict = agent.to_dict()
    assert agent_dict["id"] == "123456789"
    assert agent_dict["status"] == "running"
    assert agent_dict["pid"] == 12345
    
    # Test stop
    agent.stop()
    mock_process.terminate.assert_called_once()
    assert agent.status == "stopped"


def test_cortex_server_initialization(cortex_server):
    """Test CortexServer initialization."""
    assert cortex_server.path == "/test/ws"
    assert cortex_server.running_agents == {}


def test_handle_list_empty(cortex_server):
    """Test listing agents when none are running."""
    ws = MockWebSocket()
    
    cortex_server._handle_list(ws)
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_list"
    assert response["agents"] == []


def test_handle_list_with_agents(cortex_server):
    """Test listing agents when some are running."""
    from think.cortex import RunningAgent
    
    # Add mock running agent
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 12345
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    ws = MockWebSocket()
    cortex_server._handle_list(ws)
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_list"
    assert len(response["agents"]) == 1
    assert response["agents"][0]["id"] == "123456789"


def test_handle_attach_nonexistent_agent(cortex_server):
    """Test attaching to a non-existent agent."""
    ws = MockWebSocket()
    
    result = cortex_server._handle_attach(ws, "999999999", None)
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "error"
    assert "not found" in response["message"]
    assert result == ""


def test_handle_attach_successful(cortex_server, tmp_path):
    """Test successful agent attachment."""
    from think.cortex import RunningAgent
    
    # Create mock agent log file with sample events
    log_path = tmp_path / "test.jsonl"
    log_path.write_text(
        '{"event": "start", "ts": 1703123456789, "prompt": "test"}\n'
        '{"event": "finish", "ts": 1703123456790, "result": "done"}\n'
    )
    
    # Add mock running agent
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, log_path)
    cortex_server.running_agents["123456789"] = agent
    
    ws = MockWebSocket()
    result = cortex_server._handle_attach(ws, "123456789", None)
    
    # Should have attachment confirmation + 2 history events
    assert len(ws.messages) == 3
    
    # Check attachment confirmation
    attach_response = json.loads(ws.messages[0])
    assert attach_response["type"] == "attached"
    assert attach_response["agent_id"] == "123456789"
    
    # Check history events
    event1 = json.loads(ws.messages[1])
    assert event1["type"] == "agent_event"
    assert event1["event"]["event"] == "start"
    
    event2 = json.loads(ws.messages[2])
    assert event2["type"] == "agent_event"
    assert event2["event"]["event"] == "finish"
    
    assert result == "123456789"
    assert ws in agent.watchers


def test_handle_detach(cortex_server):
    """Test detaching from an agent."""
    from think.cortex import RunningAgent
    
    # Add mock running agent with watcher
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    ws = MockWebSocket()
    agent.watchers.add(ws)
    
    result = cortex_server._handle_detach(ws, "123456789")
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "detached"
    assert ws not in agent.watchers
    assert result is None


@patch('think.cortex.subprocess.Popen')
@patch('think.cortex.threading.Thread')
def test_handle_spawn(mock_thread, mock_popen, cortex_server, mock_journal):
    """Test spawning a new agent."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process
    
    ws = MockWebSocket()
    spawn_request = {
        "prompt": "Test prompt",
        "backend": "openai", 
        "model": "gpt-4",
        "persona": "default",
        "max_tokens": 1000
    }
    
    cortex_server._handle_spawn(ws, spawn_request)
    
    # Check subprocess was called correctly
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    
    assert "think.agents" in cmd
    assert "--backend" in cmd
    assert "openai" in cmd
    assert "--model" in cmd
    assert "gpt-4" in cmd
    assert "-q" in cmd
    assert "Test prompt" in cmd
    
    # Check response
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_spawned"
    assert "agent_id" in response
    
    # Check agent was added to running agents
    assert len(cortex_server.running_agents) == 1
    
    # Check monitoring thread was started
    mock_thread.assert_called_once()


def test_handle_spawn_missing_prompt(cortex_server, mock_journal):
    """Test spawning agent without prompt."""
    ws = MockWebSocket()
    spawn_request = {"backend": "openai"}
    
    cortex_server._handle_spawn(ws, spawn_request)
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "error"
    assert "prompt is required" in response["message"]


def test_handle_spawn_no_journal(cortex_server, monkeypatch):
    """Test spawning agent without JOURNAL_PATH."""
    monkeypatch.delenv("JOURNAL_PATH", raising=False)
    
    ws = MockWebSocket()
    spawn_request = {"prompt": "test"}
    
    cortex_server._handle_spawn(ws, spawn_request)
    
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "error"
    assert "JOURNAL_PATH not configured" in response["message"]


def test_broadcast_agent_event(cortex_server):
    """Test broadcasting events to agent watchers."""
    from think.cortex import RunningAgent
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    agent.watchers.add(ws1)
    agent.watchers.add(ws2)
    
    event_data = {"event": "thinking", "ts": 1703123456789, "summary": "test"}
    cortex_server._broadcast_agent_event("123456789", event_data)
    
    # Both watchers should receive the event
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 1
    
    response1 = json.loads(ws1.messages[0])
    assert response1["type"] == "agent_event"
    assert response1["agent_id"] == "123456789"
    assert response1["event"] == event_data


def test_broadcast_agent_event_removes_closed_watchers(cortex_server):
    """Test that closed watchers are removed during broadcast."""
    from think.cortex import RunningAgent
    from simple_websocket import ConnectionClosed
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    ws2.close()  # Close this one
    
    agent.watchers.add(ws1)
    agent.watchers.add(ws2)
    
    event_data = {"event": "thinking", "ts": 1703123456789, "summary": "test"}
    cortex_server._broadcast_agent_event("123456789", event_data)
    
    # Only ws1 should receive the event, ws2 should be removed
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 0
    assert ws2 not in agent.watchers
    assert ws1 in agent.watchers


def test_cleanup_dead_agents(cortex_server):
    """Test cleanup of finished agents."""
    from think.cortex import RunningAgent
    
    # Add one running and one dead agent
    mock_process1 = MagicMock()
    mock_process1.poll.return_value = None  # Running
    
    mock_process2 = MagicMock()
    mock_process2.poll.return_value = 1  # Finished
    
    agent1 = RunningAgent("111111111", mock_process1, Path("/tmp/test1.jsonl"))
    agent2 = RunningAgent("222222222", mock_process2, Path("/tmp/test2.jsonl"))
    
    cortex_server.running_agents["111111111"] = agent1
    cortex_server.running_agents["222222222"] = agent2
    
    cortex_server._cleanup_dead_agents()
    
    # Only running agent should remain
    assert len(cortex_server.running_agents) == 1
    assert "111111111" in cortex_server.running_agents
    assert "222222222" not in cortex_server.running_agents


def test_stop_agent(cortex_server):
    """Test stopping a running agent."""
    from think.cortex import RunningAgent
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    # Test stopping existing agent
    result = cortex_server.stop_agent("123456789")
    assert result is True
    mock_process.terminate.assert_called_once()
    
    # Test stopping non-existent agent
    result = cortex_server.stop_agent("999999999")
    assert result is False


def test_get_agent_count(cortex_server):
    """Test getting count of running agents."""
    from think.cortex import RunningAgent
    
    assert cortex_server.get_agent_count() == 0
    
    # Add running agent
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent
    
    assert cortex_server.get_agent_count() == 1


def test_connection_handler_invalid_json(cortex_server):
    """Test connection handler with invalid JSON."""
    ws = MockWebSocket()
    ws.add_received("invalid json")
    
    # Mock the _handle_connection method to test just the JSON parsing part
    with patch.object(cortex_server, '_send_error') as mock_send_error:
        try:
            cortex_server._handle_connection(ws)
        except:
            pass  # Expected due to mocking
            
    mock_send_error.assert_called()
    args = mock_send_error.call_args[0]
    assert "Invalid JSON" in args[1]


def test_connection_handler_unknown_action(cortex_server):
    """Test connection handler with unknown action."""
    ws = MockWebSocket()
    ws.add_received(json.dumps({"action": "unknown"}))
    
    with patch.object(cortex_server, '_send_error') as mock_send_error:
        try:
            cortex_server._handle_connection(ws)
        except:
            pass  # Expected due to mocking
            
    mock_send_error.assert_called()
    args = mock_send_error.call_args[0]
    assert "Unknown action: unknown" in args[1]


def test_send_message_connection_closed(cortex_server):
    """Test sending message to closed connection."""
    from simple_websocket import ConnectionClosed
    
    ws = MockWebSocket()
    ws.close()
    
    with pytest.raises(ConnectionClosed):
        cortex_server._send_message(ws, {"type": "test"})


@patch('think.cortex.time.time')
def test_tail_agent_log(mock_time, cortex_server, tmp_path):
    """Test tailing agent log file."""
    from think.cortex import RunningAgent
    
    # Mock time to return consistent values
    mock_time.return_value = 1703123456.789
    
    # Create log file with events
    log_path = tmp_path / "agent.jsonl"
    log_path.write_text(
        '{"event": "start", "ts": 1703123456789}\n'
        '{"event": "finish", "ts": 1703123456790}\n'
    )
    
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Not running
    
    agent = RunningAgent("123456789", mock_process, log_path)
    cortex_server.running_agents["123456789"] = agent
    
    # Add a watcher
    ws = MockWebSocket()
    agent.watchers.add(ws)
    
    # This should return quickly since process is not running
    cortex_server._tail_agent_log(agent)
    
    # No broadcasts should happen since process isn't running
    assert len(ws.messages) == 0


def test_send_agent_history_nonexistent_file(cortex_server):
    """Test sending history from non-existent log file."""
    ws = MockWebSocket()
    
    cortex_server._send_agent_history(ws, "123456789", Path("/nonexistent/file.jsonl"))
    
    # Should not send any messages for non-existent file
    assert len(ws.messages) == 0


def test_send_agent_history_with_invalid_json(cortex_server, tmp_path):
    """Test sending history with invalid JSON lines."""
    log_path = tmp_path / "agent.jsonl"
    log_path.write_text(
        '{"event": "start", "ts": 1703123456789}\n'
        'invalid json line\n'
        '{"event": "finish", "ts": 1703123456790}\n'
    )
    
    ws = MockWebSocket()
    cortex_server._send_agent_history(ws, "123456789", log_path)
    
    # Should send 2 valid events, skip the invalid one
    assert len(ws.messages) == 2
    
    event1 = json.loads(ws.messages[0])
    assert event1["event"]["event"] == "start"
    
    event2 = json.loads(ws.messages[1])
    assert event2["event"]["event"] == "finish"