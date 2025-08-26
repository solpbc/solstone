import json
import subprocess
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
            msg = self.received.pop(0)
            # If we receive None, disconnect to stop the loop
            if msg is None:
                self.connected = False
            return msg
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


def test_handle_list_empty(cortex_server, mock_journal):
    """Test listing agents when none are running."""
    ws = MockWebSocket()
    req = {"limit": 10, "offset": 0}

    cortex_server._handle_list(ws, req)

    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_list"
    assert response["agents"] == []
    assert "pagination" in response
    assert response["pagination"]["total"] == 0


def test_handle_list_with_agents(cortex_server, mock_journal):
    """Test listing agents when some are running."""
    from think.cortex import RunningAgent

    # Add mock running agent
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 12345

    agent = RunningAgent("123456789", mock_process, Path("/tmp/test.jsonl"))
    cortex_server.running_agents["123456789"] = agent

    ws = MockWebSocket()
    req = {"limit": 10, "offset": 0}
    cortex_server._handle_list(ws, req)

    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_list"
    assert len(response["agents"]) == 1
    assert response["agents"][0]["id"] == "123456789"
    assert "pagination" in response
    assert response["pagination"]["total"] == 1


def test_handle_attach_nonexistent_agent(cortex_server):
    """Test attaching to a non-existent agent."""
    ws = MockWebSocket()

    result = cortex_server._handle_attach(ws, "999999999", None)

    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "error"
    assert "not found or not running" in response["message"]
    assert result == ""


def test_handle_attach_successful(cortex_server, tmp_path):
    """Test successful agent attachment."""
    from think.cortex import RunningAgent

    # Create mock agent log file
    log_path = tmp_path / "test.jsonl"

    # Add mock running agent with in-memory events
    mock_process = MagicMock()
    mock_process.poll.return_value = None

    agent = RunningAgent("123456789", mock_process, log_path)
    # Add events to agent's in-memory list
    agent.events = [
        {"event": "start", "ts": 1703123456789, "prompt": "test"},
        {"event": "finish", "ts": 1703123456790, "result": "done"},
    ]
    cortex_server.running_agents["123456789"] = agent

    ws = MockWebSocket()
    result = cortex_server._handle_attach(ws, "123456789", None)

    # Should have attachment confirmation + 2 in-memory events
    assert len(ws.messages) == 3

    # Check attachment confirmation
    attach_response = json.loads(ws.messages[0])
    assert attach_response["type"] == "attached"
    assert attach_response["agent_id"] == "123456789"

    # Check in-memory events
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


@patch("think.cortex.subprocess.Popen")
@patch("think.cortex.threading.Thread")
def test_handle_spawn(mock_thread, mock_popen, cortex_server, mock_journal):
    """Test spawning a new agent."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.poll.return_value = None
    mock_process.stdin = MagicMock()
    mock_popen.return_value = mock_process

    ws = MockWebSocket()
    spawn_request = {
        "prompt": "Test prompt",
        "backend": "openai",
        "model": "gpt-4",
        "persona": "default",
        "max_tokens": 1000,
    }

    cortex_server._handle_spawn(ws, spawn_request)

    # Check subprocess was called correctly
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    cmd = call_args[0][0]

    # Should only have the command name now
    assert cmd == ["think-agents"]

    # Check stdin was configured
    assert call_args[1]["stdin"] == subprocess.PIPE

    # Check NDJSON was written to stdin
    mock_process.stdin.write.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson_request = json.loads(written_data.strip())

    assert ndjson_request["prompt"] == "Test prompt"
    assert ndjson_request["backend"] == "openai"
    assert ndjson_request["persona"] == "default"
    assert ndjson_request["config"]["model"] == "gpt-4"
    assert ndjson_request["config"]["max_tokens"] == 1000

    # Check stdin was closed
    mock_process.stdin.close.assert_called_once()

    # Check response
    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_spawned"
    assert "agent_id" in response

    # Check agent was added to running agents
    assert len(cortex_server.running_agents) == 1

    # Check monitoring threads were started (stdout and stderr)
    assert mock_thread.call_count == 2  # Two threads: stdout and stderr


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


def test_get_agent_count(cortex_server, mock_journal):
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
    ws.add_received(None)  # This will cause the loop to break

    # Mock the _handle_connection method to test just the JSON parsing part
    with patch.object(cortex_server, "_send_error") as mock_send_error:
        try:
            cortex_server._handle_connection(ws)
        except Exception:
            pass  # Expected due to mocking

    mock_send_error.assert_called()
    args = mock_send_error.call_args[0]
    assert "Invalid JSON" in args[1]


def test_connection_handler_unknown_action(cortex_server):
    """Test connection handler with unknown action."""
    ws = MockWebSocket()
    ws.add_received(json.dumps({"action": "unknown"}))
    ws.add_received(None)  # This will cause the loop to break

    with patch.object(cortex_server, "_send_error") as mock_send_error:
        try:
            cortex_server._handle_connection(ws)
        except Exception:
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


@patch("think.cortex.time.time")
def test_monitor_stdout(mock_time, cortex_server, tmp_path):
    """Test monitoring agent stdout."""
    from io import StringIO

    from think.cortex import RunningAgent

    # Mock time to return consistent values
    mock_time.return_value = 1703123456.789

    # Create log file path
    log_path = tmp_path / "agent.jsonl"

    # Create mock process with stdout
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Still running initially
    mock_process.stdout = StringIO(
        '{"event": "start", "ts": 1703123456789}\n'
        '{"event": "finish", "ts": 1703123456790}\n'
    )

    agent = RunningAgent("123456789", mock_process, log_path)
    cortex_server.running_agents["123456789"] = agent

    # Add a watcher
    ws = MockWebSocket()
    agent.watchers.add(ws)

    # Monitor stdout (this will read all lines and write to log)
    cortex_server._monitor_stdout(agent)

    # Check log file was created with events
    assert log_path.exists()
    log_contents = log_path.read_text()
    assert '"event": "start"' in log_contents
    assert '"event": "finish"' in log_contents


# Removed tests for historical agent functionality:
# - test_send_agent_history_nonexistent_file
# - test_send_agent_history_with_invalid_json
# - test_historical_agent_creation
# - test_load_historical_agents_empty
# - test_load_historical_agents_with_files
# - test_parse_agent_file_valid
# - test_parse_agent_file_empty
# - test_determine_agent_status_finished
# - test_determine_agent_status_error
# - test_determine_agent_status_unknown


def test_get_running_agents_with_pagination(cortex_server, mock_journal):
    """Test getting running agents with pagination."""
    from think.cortex import RunningAgent

    # Add running agents
    mock_process1 = MagicMock()
    mock_process1.poll.return_value = None
    mock_process1.pid = 12345
    agent1 = RunningAgent("999999999", mock_process1, Path("/tmp/running1.jsonl"))
    agent1.started_at = 1703123456789
    cortex_server.running_agents["999999999"] = agent1

    mock_process2 = MagicMock()
    mock_process2.poll.return_value = None
    mock_process2.pid = 12346
    agent2 = RunningAgent("888888888", mock_process2, Path("/tmp/running2.jsonl"))
    agent2.started_at = 1703123456790  # Newer
    cortex_server.running_agents["888888888"] = agent2

    # Test pagination
    agents, total = cortex_server._get_running_agents_with_pagination(
        limit=10, offset=0
    )

    assert total == 2  # 2 running agents
    assert len(agents) == 2

    # Agents should be sorted by started_at (newest first)
    assert agents[0]["id"] == "888888888"  # Newer agent first
    assert agents[1]["id"] == "999999999"


def test_handle_list_with_pagination(cortex_server, mock_journal):
    """Test list request with pagination parameters."""
    from think.cortex import RunningAgent

    # Add only running agents
    for i in range(15):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345 + i
        agent = RunningAgent(
            f"99999999{i}", mock_process, Path(f"/tmp/running{i}.jsonl")
        )
        agent.started_at = 1703123456789 + i  # Different timestamps for sorting
        cortex_server.running_agents[f"99999999{i}"] = agent

    # Test first page
    ws = MockWebSocket()
    req = {"limit": 10, "offset": 0}
    cortex_server._handle_list(ws, req)

    response = json.loads(ws.messages[0])
    assert response["type"] == "agent_list"
    assert len(response["agents"]) == 10
    assert response["pagination"]["total"] == 15
    assert response["pagination"]["has_more"] is True

    # Test second page
    ws = MockWebSocket()
    req = {"limit": 10, "offset": 10}
    cortex_server._handle_list(ws, req)

    response = json.loads(ws.messages[0])
    assert len(response["agents"]) == 5
    assert response["pagination"]["has_more"] is False


def test_handle_list_invalid_pagination(cortex_server):
    """Test list request with invalid pagination parameters."""
    ws = MockWebSocket()
    req = {"limit": "invalid", "offset": "invalid"}

    cortex_server._handle_list(ws, req)

    assert len(ws.messages) == 1
    response = json.loads(ws.messages[0])
    assert response["type"] == "error"
    assert "Invalid limit or offset" in response["message"]
