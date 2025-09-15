"""Tests for the file-based Cortex agent manager."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
def cortex_service(mock_journal):
    """Create a CortexService instance for testing."""
    from think.cortex import CortexService

    return CortexService(str(mock_journal))


def test_agent_process_creation():
    """Test AgentProcess class initialization and methods."""
    from think.cortex import AgentProcess

    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Running
    mock_process.pid = 12345

    log_path = Path("/tmp/test.jsonl")
    agent = AgentProcess("123456789", mock_process, log_path)

    assert agent.agent_id == "123456789"
    assert agent.process == mock_process
    assert agent.log_path == log_path
    assert agent.is_running() is True

    # Test stop
    agent.stop()
    mock_process.terminate.assert_called_once()
    assert agent.stop_event.is_set()


def test_cortex_service_initialization(cortex_service, mock_journal):
    """Test CortexService initialization."""
    assert cortex_service.journal_path == mock_journal
    assert cortex_service.agents_dir == mock_journal / "agents"
    assert cortex_service.running_agents == {}
    assert cortex_service.agents_dir.exists()


def test_process_existing_active_files(cortex_service, mock_journal):
    """Test processing existing active files on startup."""
    # Create an existing active file
    active_file = mock_journal / "agents" / "123456789_active.jsonl"
    request = {
        "event": "request",
        "ts": 123456789,
        "prompt": "Test prompt",
        "backend": "openai",
        "persona": "default",
    }
    active_file.write_text(json.dumps(request) + "\n")

    with patch.object(cortex_service, "_write_error_and_complete") as mock_error:
        cortex_service._process_existing_active_files()
        # Now it writes error and completes for stale files
        mock_error.assert_called_once()
        assert "Cortex service shutdown" in mock_error.call_args[0][1]


def test_handle_active_file_valid_request(cortex_service, mock_journal):
    """Test handling a valid active file."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    request = {
        "event": "request",
        "ts": 123456789,
        "prompt": "Test prompt",
        "backend": "openai",
        "persona": "default",
        "model": "gpt-4",
    }
    file_path.write_text(json.dumps(request) + "\n")

    with patch.object(cortex_service, "_spawn_agent") as mock_spawn:
        with patch("think.utils.get_agent") as mock_get_agent:
            # Create a mock module for mcp_tools
            mock_mcp_tools = MagicMock()
            mock_mcp_tools.get_tools = MagicMock(return_value=[])

            with patch.dict("sys.modules", {"think.mcp_tools": mock_mcp_tools}):
                # Mock get_agent to return a config
                mock_get_agent.return_value = {
                    "instruction": "Default instruction",
                    "title": "Default",
                }

                cortex_service._handle_active_file(agent_id, file_path)
                # Now it passes the merged config
                # The config includes get_agent result merged with request
                assert mock_spawn.called
                called_config = mock_spawn.call_args[0][2]
                assert called_config["prompt"] == "Test prompt"
                assert called_config["backend"] == "openai"


def test_handle_active_file_empty_file(cortex_service, mock_journal):
    """Test handling an empty active file."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    file_path.touch()  # Empty file

    with patch.object(cortex_service, "_spawn_agent") as mock_spawn:
        cortex_service._handle_active_file(agent_id, file_path)
        mock_spawn.assert_not_called()


def test_handle_active_file_invalid_json(cortex_service, mock_journal):
    """Test handling active file with invalid JSON."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    file_path.write_text("invalid json\n")

    with patch.object(cortex_service, "_write_error_and_complete") as mock_error:
        cortex_service._handle_active_file(agent_id, file_path)
        mock_error.assert_called_once()
        assert "Invalid JSON" in mock_error.call_args[0][1]


def test_handle_active_file_missing_event_field(cortex_service, mock_journal):
    """Test handling active file without 'event' field."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    request = {"ts": 123456789, "prompt": "Test prompt"}
    file_path.write_text(json.dumps(request) + "\n")

    with patch.object(cortex_service, "_spawn_agent") as mock_spawn:
        cortex_service._handle_active_file(agent_id, file_path)
        mock_spawn.assert_not_called()


def test_handle_active_file_empty_prompt(cortex_service, mock_journal):
    """Test handling active file with empty prompt."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    request = {"event": "request", "ts": 123456789, "prompt": "", "backend": "openai"}
    file_path.write_text(json.dumps(request) + "\n")

    with patch.object(cortex_service, "_write_error_and_complete") as mock_error:
        cortex_service._handle_active_file(agent_id, file_path)
        mock_error.assert_called_once()
        assert "Empty prompt" in mock_error.call_args[0][1]


@patch("think.cortex.subprocess.Popen")
@patch("think.cortex.threading.Thread")
@patch("think.cortex.threading.Timer")
def test_spawn_agent(mock_timer, mock_thread, mock_popen, cortex_service, mock_journal):
    """Test spawning an agent subprocess."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.poll.return_value = None
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_popen.return_value = mock_process

    # Setup mock timer
    mock_timer_instance = MagicMock()
    mock_timer.return_value = mock_timer_instance

    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    request = {
        "event": "request",
        "ts": 123456789,
        "prompt": "Test prompt",
        "backend": "openai",
        "persona": "default",
        "model": "gpt-4",
    }

    cortex_service._spawn_agent(
        agent_id,
        file_path,
        request,
    )

    # Check subprocess was called
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    assert call_args[0][0] == ["think-agents"]
    assert call_args[1]["stdin"] is not None
    assert call_args[1]["stdout"] is not None
    assert call_args[1]["stderr"] is not None

    # Check NDJSON was written to stdin
    mock_process.stdin.write.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson = json.loads(written_data.strip())
    assert ndjson["event"] == "request"
    assert ndjson["prompt"] == "Test prompt"
    assert ndjson["backend"] == "openai"
    assert ndjson["persona"] == "default"
    assert ndjson["model"] == "gpt-4"

    # Check stdin was closed
    mock_process.stdin.close.assert_called_once()

    # Check agent was tracked
    assert agent_id in cortex_service.running_agents
    agent = cortex_service.running_agents[agent_id]
    assert agent.agent_id == agent_id
    assert agent.log_path == file_path

    # Check monitoring threads were started
    assert mock_thread.call_count == 2  # stdout and stderr

    # Check timer was created and started
    mock_timer.assert_called_once()
    mock_timer_instance.start.assert_called_once()


@patch("think.cortex.subprocess.Popen")
def test_spawn_agent_with_handoff_from(mock_popen, cortex_service, mock_journal):
    """Test spawning an agent with handoff_from parameter."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_popen.return_value = mock_process

    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    request = {
        "event": "request",
        "ts": 123456789,
        "prompt": "Test",
        "backend": "openai",
        "persona": "default",
        "handoff_from": "parent123",
    }

    with patch("think.cortex.threading.Thread"):
        cortex_service._spawn_agent(agent_id, file_path, request)

    # Check handoff_from was included in NDJSON
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson = json.loads(written_data.strip())
    assert ndjson["handoff_from"] == "parent123"


def test_monitor_stdout_json_events(cortex_service, mock_journal):
    """Test monitoring stdout with JSON events."""
    from io import StringIO

    from think.cortex import AgentProcess

    agent_id = "123456789"
    log_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Process exits
    mock_process.stdout = StringIO(
        '{"event": "start", "ts": 1234567890}\n'
        '{"event": "finish", "ts": 1234567891, "result": "Done"}\n'
    )

    agent = AgentProcess(agent_id, mock_process, log_path)
    cortex_service.running_agents[agent_id] = agent

    with patch.object(cortex_service, "_complete_agent_file") as mock_complete:
        cortex_service._monitor_stdout(agent)

        # Check events were written to file
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "start"
        assert json.loads(lines[1])["event"] == "finish"

        # Check file was completed
        mock_complete.assert_called_once_with(agent_id, log_path)

    # Check agent was removed
    assert agent_id not in cortex_service.running_agents


def test_monitor_stdout_non_json_output(cortex_service, mock_journal):
    """Test monitoring stdout with non-JSON output."""
    from io import StringIO

    from think.cortex import AgentProcess

    agent_id = "123456789"
    log_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.stdout = StringIO(
        "Plain text output\n" '{"event": "finish", "ts": 1234567890}\n'
    )

    agent = AgentProcess(agent_id, mock_process, log_path)
    cortex_service.running_agents[agent_id] = agent

    with patch.object(cortex_service, "_complete_agent_file"):
        cortex_service._monitor_stdout(agent)

        # Check info event was created for non-JSON
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        info_event = json.loads(lines[0])
        assert info_event["event"] == "info"
        assert info_event["message"] == "Plain text output"
        assert "ts" in info_event


def test_monitor_stdout_with_handoff(cortex_service, mock_journal):
    """Test monitoring stdout with handoff in finish event."""
    from io import StringIO

    from think.cortex import AgentProcess

    agent_id = "123456789"
    log_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.stdout = StringIO(
        '{"event": "finish", "ts": 1234567890, "result": "Create matter", '
        '"handoff": {"persona": "matter_editor", "domain": "test"}}\n'
    )

    agent = AgentProcess(agent_id, mock_process, log_path)
    cortex_service.running_agents[agent_id] = agent

    with patch.object(cortex_service, "_spawn_handoff") as mock_handoff:
        with patch.object(cortex_service, "_complete_agent_file"):
            cortex_service._monitor_stdout(agent)

            mock_handoff.assert_called_once_with(
                agent_id,
                "Create matter",
                {"persona": "matter_editor", "domain": "test"},
            )


def test_monitor_stdout_no_finish_event(cortex_service, mock_journal):
    """Test monitoring stdout when process exits without finish event."""
    from io import StringIO

    from think.cortex import AgentProcess

    agent_id = "123456789"
    log_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.wait.return_value = 1  # Non-zero exit
    mock_process.stdout = StringIO('{"event": "start", "ts": 1234567890}\n')

    agent = AgentProcess(agent_id, mock_process, log_path)
    cortex_service.running_agents[agent_id] = agent

    with patch.object(cortex_service, "_complete_agent_file"):
        cortex_service._monitor_stdout(agent)

        # Check error event was added
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        error_event = json.loads(lines[1])
        assert error_event["event"] == "error"
        assert "exit_code" in error_event
        assert error_event["exit_code"] == 1


def test_monitor_stderr(cortex_service, mock_journal):
    """Test monitoring stderr for errors."""
    from io import StringIO

    from think.cortex import AgentProcess

    agent_id = "123456789"
    log_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Error exit
    mock_process.stderr = StringIO(
        "Error: Something went wrong\n" "Stack trace line 1\n" "Stack trace line 2\n"
    )

    agent = AgentProcess(agent_id, mock_process, log_path)

    cortex_service._monitor_stderr(agent)

    # Check error event was written
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1

    error_event = json.loads(lines[0])
    assert error_event["event"] == "error"
    assert "trace" in error_event
    assert "Error: Something went wrong" in error_event["trace"]
    assert error_event["exit_code"] == 1


def test_has_finish_event(cortex_service, mock_journal):
    """Test checking for finish event in JSONL file."""
    file_path = mock_journal / "agents" / "test.jsonl"

    # File with finish event
    file_path.write_text(
        '{"event": "start", "ts": 123}\n' '{"event": "finish", "ts": 124}\n'
    )
    assert cortex_service._has_finish_event(file_path) is True

    # File with error event
    file_path.write_text(
        '{"event": "start", "ts": 123}\n' '{"event": "error", "ts": 124}\n'
    )
    assert cortex_service._has_finish_event(file_path) is True

    # File without finish/error
    file_path.write_text('{"event": "start", "ts": 123}\n')
    assert cortex_service._has_finish_event(file_path) is False

    # Empty file
    file_path.write_text("")
    assert cortex_service._has_finish_event(file_path) is False


def test_complete_agent_file(cortex_service, mock_journal):
    """Test completing an agent file (rename from active to completed)."""
    agent_id = "123456789"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    active_path.touch()

    cortex_service._complete_agent_file(agent_id, active_path)

    # Check file was renamed
    assert not active_path.exists()
    completed_path = mock_journal / "agents" / f"{agent_id}.jsonl"
    assert completed_path.exists()


def test_write_error_and_complete(cortex_service, mock_journal):
    """Test writing error and completing file."""
    agent_id = "123456789"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    file_path.touch()

    cortex_service._write_error_and_complete(file_path, "Test error message")

    # Check error was written
    completed_path = mock_journal / "agents" / f"{agent_id}.jsonl"
    assert completed_path.exists()
    assert not file_path.exists()

    content = completed_path.read_text()
    error_event = json.loads(content)
    assert error_event["event"] == "error"
    assert error_event["error"] == "Test error message"
    assert "ts" in error_event


def test_spawn_handoff(cortex_service, mock_journal):
    """Test spawning a handoff agent."""
    parent_id = "parent123"
    result = "Create a new matter for AI research"
    handoff = {
        "persona": "matter_editor",
        "backend": "claude",
        "domain": "test",
        "max_turns": 5,
    }

    with patch("think.cortex_client.cortex_request") as mock_request:
        mock_request.return_value = (
            mock_journal / "agents" / "987654321000_active.jsonl"
        )
        cortex_service._spawn_handoff(parent_id, result, handoff)

        # Check cortex_request was called with correct parameters
        mock_request.assert_called_once_with(
            prompt=result,
            persona="matter_editor",
            backend="claude",
            handoff_from=parent_id,
            config={"domain": "test", "max_turns": 5},
        )


def test_spawn_handoff_with_explicit_prompt(cortex_service, mock_journal):
    """Test spawning handoff with explicit prompt in config."""
    parent_id = "parent123"
    result = "Parent result"
    handoff = {
        "persona": "reviewer",
        "prompt": "Review this analysis",  # Explicit prompt
    }

    with patch("think.cortex_client.cortex_request") as mock_request:
        cortex_service._spawn_handoff(parent_id, result, handoff)

        # Check cortex_request was called with explicit prompt
        mock_request.assert_called_once_with(
            prompt="Review this analysis",  # Uses explicit prompt
            persona="reviewer",
            backend="openai",
            handoff_from=parent_id,
            config={},  # Empty config since only prompt and persona in handoff
        )


def test_stop_service(cortex_service):
    """Test stopping the Cortex service."""
    from think.cortex import AgentProcess

    # Add running agents
    mock_process1 = MagicMock()
    mock_process1.poll.return_value = None
    agent1 = AgentProcess("111", mock_process1, Path("/tmp/1.jsonl"))

    mock_process2 = MagicMock()
    mock_process2.poll.return_value = None
    agent2 = AgentProcess("222", mock_process2, Path("/tmp/2.jsonl"))

    cortex_service.running_agents["111"] = agent1
    cortex_service.running_agents["222"] = agent2

    # Mock observer
    mock_observer = MagicMock()
    mock_observer.is_alive.return_value = True
    cortex_service.observer = mock_observer

    cortex_service.stop()

    # Check stop event is set
    assert cortex_service.stop_event.is_set()

    # Check observer was stopped
    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called_once_with(timeout=5)

    # Check all agents were stopped
    assert agent1.stop_event.is_set()
    assert agent2.stop_event.is_set()
    mock_process1.terminate.assert_called_once()
    mock_process2.terminate.assert_called_once()


def test_get_status(cortex_service):
    """Test getting service status."""
    from think.cortex import AgentProcess

    # Empty status
    status = cortex_service.get_status()
    assert status["running_agents"] == 0
    assert status["agent_ids"] == []

    # Add running agents
    mock_process = MagicMock()
    agent1 = AgentProcess("111", mock_process, Path("/tmp/1.jsonl"))
    agent2 = AgentProcess("222", mock_process, Path("/tmp/2.jsonl"))

    cortex_service.running_agents["111"] = agent1
    cortex_service.running_agents["222"] = agent2

    status = cortex_service.get_status()
    assert status["running_agents"] == 2
    assert set(status["agent_ids"]) == {"111", "222"}


def test_agent_file_handler_on_moved(cortex_service, mock_journal):
    """Test AgentFileHandler handles moved files (pending -> active)."""
    from think.cortex import AgentFileHandler

    handler = AgentFileHandler(cortex_service)

    # Create mock event for file move
    mock_event = MagicMock()
    mock_event.is_directory = False
    mock_event.dest_path = str(mock_journal / "agents" / "123_active.jsonl")

    with patch.object(cortex_service, "_handle_active_file") as mock_handle:
        handler.on_moved(mock_event)
        mock_handle.assert_called_once_with("123", Path(mock_event.dest_path))


def test_agent_file_handler_on_created(cortex_service, mock_journal):
    """Test AgentFileHandler handles created files."""
    from think.cortex import AgentFileHandler

    handler = AgentFileHandler(cortex_service)

    # Create mock event for file creation
    mock_event = MagicMock()
    mock_event.is_directory = False
    mock_event.src_path = str(mock_journal / "agents" / "456_active.jsonl")

    with patch.object(cortex_service, "_handle_active_file") as mock_handle:
        with patch("think.cortex.time.sleep"):  # Skip delay
            handler.on_created(mock_event)
            mock_handle.assert_called_once_with("456", Path(mock_event.src_path))


def test_agent_file_handler_ignores_directories(cortex_service, mock_journal):
    """Test AgentFileHandler ignores directory events."""
    from think.cortex import AgentFileHandler

    handler = AgentFileHandler(cortex_service)

    # Create mock event for directory
    mock_event = MagicMock()
    mock_event.is_directory = True
    mock_event.dest_path = str(mock_journal / "agents" / "subdir")

    with patch.object(cortex_service, "_handle_active_file") as mock_handle:
        handler.on_moved(mock_event)
        mock_handle.assert_not_called()


@patch("think.cortex.Observer")
def test_start_with_watchdog(mock_observer_class, cortex_service, mock_journal):
    """Test starting service with watchdog Observer."""
    mock_observer = MagicMock()
    mock_observer_class.return_value = mock_observer

    # Create existing active file
    active_file = mock_journal / "agents" / "existing_active.jsonl"
    active_file.touch()

    with patch.object(cortex_service, "_process_existing_active_files") as mock_process:
        with patch.object(
            cortex_service.stop_event, "is_set", side_effect=[False, True]
        ):
            with patch("think.cortex.time.sleep"):
                cortex_service.start()

    # Check observer was configured
    mock_observer_class.assert_called_once()
    mock_observer.schedule.assert_called_once()
    mock_observer.start.assert_called_once()

    # Check existing files were processed
    mock_process.assert_called_once()


def test_save_agent_result(cortex_service, mock_journal):
    """Test saving agent result to file in day directory."""
    # Mock datetime to return a specific date
    test_date = "20240115"
    from datetime import datetime as dt

    mock_dt = dt(2024, 1, 15, 12, 0, 0)
    with patch("think.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt

        # Test saving result
        agent_id = "test_agent"
        result = "This is the agent result content"
        save_filename = "test_output.md"

        cortex_service._save_agent_result(agent_id, result, save_filename)

        # Check file was created in correct location
        expected_path = mock_journal / test_date / save_filename
        assert expected_path.exists()
        assert expected_path.read_text() == result

        # Check directory was created
        assert (mock_journal / test_date).is_dir()


def test_save_agent_result_with_error(cortex_service, mock_journal, caplog):
    """Test save agent result handles errors gracefully."""
    import logging

    # Make journal read-only to cause error
    with patch("builtins.open", side_effect=PermissionError("Cannot write")):
        with caplog.at_level(logging.ERROR):
            cortex_service._save_agent_result("agent_id", "result", "output.md")

    # Check error was logged but didn't raise
    assert "Failed to save agent agent_id result" in caplog.text


def test_save_agent_result_with_day_parameter(cortex_service, mock_journal):
    """Test saving agent result to a specific day directory."""
    # Test saving result with explicit day parameter
    agent_id = "test_agent"
    result = "This is the agent result content"
    save_filename = "test_output.md"
    specified_day = "20240201"

    cortex_service._save_agent_result(
        agent_id, result, save_filename, day=specified_day
    )

    # Check file was created in specified day directory
    expected_path = mock_journal / specified_day / save_filename
    assert expected_path.exists()
    assert expected_path.read_text() == result

    # Check directory was created
    assert (mock_journal / specified_day).is_dir()


def test_save_agent_result_with_invalid_day(cortex_service, mock_journal, caplog):
    """Test save agent result with invalid day format."""
    import logging

    # Test with invalid day format
    agent_id = "test_agent"
    result = "Test content"
    save_filename = "output.md"
    invalid_day = "2024-02-01"  # Wrong format

    with caplog.at_level(logging.ERROR):
        cortex_service._save_agent_result(
            agent_id, result, save_filename, day=invalid_day
        )

    # Check error was logged
    assert "Failed to save agent test_agent result" in caplog.text

    # File should not exist in invalid path
    assert not (mock_journal / invalid_day / save_filename).exists()


def test_monitor_stdout_with_save(cortex_service, mock_journal):
    """Test monitor_stdout saves result when save field is present."""
    from think.cortex import AgentProcess

    # Create agent with save in request
    agent_id = "save_test"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    # Store request with save field
    cortex_service.agent_requests = {
        agent_id: {"event": "request", "prompt": "test", "save": "output.md"}
    }

    # Create mock process with stdout
    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Test result"}\n',
    ]
    mock_process.stdout = iter(mock_stdout)
    mock_process.wait.return_value = 0

    agent = AgentProcess(agent_id, mock_process, active_path)

    # Mock datetime for consistent test
    test_date = "20240115"
    from datetime import datetime as dt

    mock_dt = dt(2024, 1, 15, 12, 0, 0)
    with patch("think.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt

        with patch.object(cortex_service, "_complete_agent_file"):
            with patch.object(cortex_service, "_has_finish_event", return_value=True):
                cortex_service._monitor_stdout(agent)

    # Check result was saved
    save_path = mock_journal / test_date / "output.md"
    assert save_path.exists()
    assert save_path.read_text() == "Test result"


def test_monitor_stdout_with_save_and_day(cortex_service, mock_journal):
    """Test monitor_stdout saves result to specific day when day field is present."""
    from think.cortex import AgentProcess

    # Create agent with save and day in request
    agent_id = "save_day_test"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    specified_day = "20240220"

    # Store request with save and day fields
    cortex_service.agent_requests = {
        agent_id: {
            "event": "request",
            "prompt": "test",
            "save": "report.md",
            "day": specified_day,
        }
    }

    # Create mock process with stdout
    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Daily report content"}\n',
    ]
    mock_process.stdout = iter(mock_stdout)
    mock_process.wait.return_value = 0

    agent = AgentProcess(agent_id, mock_process, active_path)

    with patch.object(cortex_service, "_complete_agent_file"):
        with patch.object(cortex_service, "_has_finish_event", return_value=True):
            cortex_service._monitor_stdout(agent)

    # Check result was saved to specified day
    save_path = mock_journal / specified_day / "report.md"
    assert save_path.exists()
    assert save_path.read_text() == "Daily report content"
