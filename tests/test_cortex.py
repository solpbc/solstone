# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the file-based Cortex agent manager."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from think.models import GPT_5


class MockPipe:
    """Mock for subprocess stdout/stderr that supports context manager protocol."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self._iter = None

    def __enter__(self):
        self._iter = iter(self._lines)
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return self._iter or iter(self._lines)

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._lines)
        return next(self._iter)


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


@patch("think.cortex.subprocess.Popen")
@patch("think.cortex.threading.Thread")
@patch("think.cortex.threading.Timer")
def test_spawn_subprocess(
    mock_timer, mock_thread, mock_popen, cortex_service, mock_journal
):
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
        "provider": "openai",
        "name": "default",
        "model": GPT_5,
    }

    cortex_service._spawn_subprocess(
        agent_id,
        file_path,
        request,
        ["sol", "agents"],
        "agent",
    )

    # Check subprocess was called
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    assert call_args[0][0] == ["sol", "agents"]
    assert call_args[1]["stdin"] is not None
    assert call_args[1]["stdout"] is not None
    assert call_args[1]["stderr"] is not None

    # Check NDJSON was written to stdin
    mock_process.stdin.write.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson = json.loads(written_data.strip())
    assert ndjson["event"] == "request"
    assert ndjson["prompt"] == "Test prompt"
    assert ndjson["provider"] == "openai"
    assert ndjson["name"] == "default"
    assert ndjson["model"] == GPT_5

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
@patch("think.cortex.threading.Thread")
@patch("think.cortex.threading.Timer")
def test_spawn_generator_via_subprocess(
    mock_timer, mock_thread, mock_popen, cortex_service, mock_journal
):
    """Test spawning a generator subprocess via _spawn_subprocess."""
    mock_process = MagicMock()
    mock_process.pid = 54321
    mock_process.poll.return_value = None
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_popen.return_value = mock_process

    # Setup mock timer
    mock_timer_instance = MagicMock()
    mock_timer.return_value = mock_timer_instance

    agent_id = "987654321"
    file_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    # Generator config has "output" instead of "tools"
    config = {
        "event": "request",
        "ts": 987654321,
        "name": "activity",
        "day": "20240101",
        "output": "md",
    }

    # Generators route through _spawn_subprocess
    cortex_service._spawn_subprocess(
        agent_id,
        file_path,
        config,
        ["sol", "agents"],
        "agent",
    )

    # Check subprocess was called with agents command (generators route through agents)
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    assert call_args[0][0] == ["sol", "agents"]
    assert call_args[1]["stdin"] is not None
    assert call_args[1]["stdout"] is not None
    assert call_args[1]["stderr"] is not None

    # Check NDJSON was written to stdin
    mock_process.stdin.write.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson = json.loads(written_data.strip())
    assert ndjson["event"] == "request"
    assert ndjson["name"] == "activity"
    assert ndjson["day"] == "20240101"
    assert ndjson["output"] == "md"

    # Check stdin was closed
    mock_process.stdin.close.assert_called_once()

    # Check generator was tracked
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
def test_spawn_subprocess_with_handoff_from(mock_popen, cortex_service, mock_journal):
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
        "provider": "openai",
        "name": "default",
        "handoff_from": "parent123",
    }

    with patch("think.cortex.threading.Thread"):
        cortex_service._spawn_subprocess(
            agent_id, file_path, request, ["sol", "agents"], "agent"
        )

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

    # Handoff config is now in the finish event itself
    finish_event = {
        "event": "finish",
        "ts": 1234567890,
        "result": "Create matter",
        "handoff": {"name": "matter_editor", "facet": "test"},
    }

    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.stdout = StringIO(json.dumps(finish_event) + "\n")

    agent = AgentProcess(agent_id, mock_process, log_path)
    cortex_service.running_agents[agent_id] = agent

    with patch.object(cortex_service, "_spawn_handoff") as mock_handoff:
        with patch.object(cortex_service, "_complete_agent_file"):
            cortex_service._monitor_stdout(agent)

            mock_handoff.assert_called_once_with(
                agent_id,
                "Create matter",
                {"name": "matter_editor", "facet": "test"},
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
    cortex_service.agent_requests[agent_id] = {"name": "default", "agent_id": agent_id}

    cortex_service._complete_agent_file(agent_id, active_path)

    # Check file was renamed
    assert not active_path.exists()
    completed_path = mock_journal / "agents" / f"{agent_id}.jsonl"
    assert completed_path.exists()
    symlink_path = mock_journal / "agents" / "default.jsonl"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"{agent_id}.jsonl"


def test_complete_agent_file_replaces_symlink(cortex_service, mock_journal):
    """Test completing agent file replaces convenience symlink for same name."""
    first_agent_id = "111"
    first_active_path = mock_journal / "agents" / f"{first_agent_id}_active.jsonl"
    first_active_path.touch()
    cortex_service.agent_requests[first_agent_id] = {"name": "default"}

    cortex_service._complete_agent_file(first_agent_id, first_active_path)

    second_agent_id = "222"
    second_active_path = mock_journal / "agents" / f"{second_agent_id}_active.jsonl"
    second_active_path.touch()
    cortex_service.agent_requests[second_agent_id] = {"name": "default"}

    cortex_service._complete_agent_file(second_agent_id, second_active_path)

    symlink_path = mock_journal / "agents" / "default.jsonl"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"{second_agent_id}.jsonl"


def test_complete_agent_file_colon_name(cortex_service, mock_journal):
    """Test completing agent file sanitizes colon in convenience symlink name."""
    agent_id = "123456789"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    active_path.touch()
    cortex_service.agent_requests[agent_id] = {"name": "entities:entity_assist"}

    cortex_service._complete_agent_file(agent_id, active_path)

    symlink_path = mock_journal / "agents" / "entities--entity_assist.jsonl"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"{agent_id}.jsonl"


def test_complete_agent_file_no_name(cortex_service, mock_journal):
    """Test completing agent file skips symlink when request name is missing."""
    agent_id = "123456789"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    active_path.touch()

    cortex_service._complete_agent_file(agent_id, active_path)

    completed_path = mock_journal / "agents" / f"{agent_id}.jsonl"
    assert completed_path.exists()
    assert not any(path.is_symlink() for path in (mock_journal / "agents").iterdir())


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
        "name": "matter_editor",
        "provider": "anthropic",
        "facet": "test",
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
            name="matter_editor",
            provider="anthropic",
            handoff_from=parent_id,
            config={"facet": "test", "max_turns": 5},
        )


def test_spawn_handoff_with_explicit_prompt(cortex_service, mock_journal):
    """Test spawning handoff with explicit prompt in config."""
    parent_id = "parent123"
    result = "Parent result"
    handoff = {
        "name": "reviewer",
        "prompt": "Review this analysis",  # Explicit prompt
    }

    with patch("think.cortex_client.cortex_request") as mock_request:
        cortex_service._spawn_handoff(parent_id, result, handoff)

        # Check cortex_request was called with explicit prompt
        # Provider is None when not explicitly set - let the agent resolve its own
        mock_request.assert_called_once_with(
            prompt="Review this analysis",  # Uses explicit prompt
            name="reviewer",
            provider=None,
            handoff_from=parent_id,
            config=None,
        )


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


def test_write_output(cortex_service, mock_journal):
    """Test writing agent output to agents directory."""
    # Mock datetime to return a specific date
    test_date = "20240115"
    from datetime import datetime as dt

    mock_dt = dt(2024, 1, 15, 12, 0, 0)
    with patch("think.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt

        # Test writing output
        agent_id = "test_agent"
        result = "This is the agent result content"
        config = {"output": "md", "name": "my_agent"}

        cortex_service._write_output(agent_id, result, config)

        # Check file was created in agents/ with name-derived filename
        expected_path = mock_journal / test_date / "agents" / "my_agent.md"
        assert expected_path.exists()
        assert expected_path.read_text() == result

        # Check directories were created
        assert (mock_journal / test_date / "agents").is_dir()


def test_write_output_with_error(cortex_service, mock_journal, caplog):
    """Test write output handles errors gracefully."""
    import logging

    # Make journal read-only to cause error
    with patch("builtins.open", side_effect=PermissionError("Cannot write")):
        with caplog.at_level(logging.ERROR):
            config = {"output": "md", "name": "test"}
            cortex_service._write_output("agent_id", "result", config)

    # Check error was logged but didn't raise
    assert "Failed to write agent agent_id output" in caplog.text


def test_write_output_with_day_parameter(cortex_service, mock_journal):
    """Test writing agent output to a specific day directory."""
    # Test writing output with explicit day parameter
    agent_id = "test_agent"
    result = "This is the agent result content"
    specified_day = "20240201"
    config = {"output": "md", "name": "reporter", "day": specified_day}

    cortex_service._write_output(agent_id, result, config)

    # Check file was created in specified day's agents directory
    expected_path = mock_journal / specified_day / "agents" / "reporter.md"
    assert expected_path.exists()
    assert expected_path.read_text() == result

    # Check directories were created
    assert (mock_journal / specified_day / "agents").is_dir()


def test_write_output_with_segment(cortex_service, mock_journal):
    """Test writing segment agent output to segment agents directory."""
    # Mock datetime to return a specific date
    test_date = "20240115"
    from datetime import datetime as dt

    mock_dt = dt(2024, 1, 15, 12, 0, 0)
    with patch("think.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt

        agent_id = "segment_agent"
        result = "Segment analysis content"
        config = {"output": "md", "name": "analyzer", "segment": "143000_600"}

        cortex_service._write_output(agent_id, result, config)

        # Check file was created in segment agents/ directory
        expected_path = (
            mock_journal / test_date / "143000_600" / "agents" / "analyzer.md"
        )
        assert expected_path.exists()
        assert expected_path.read_text() == result


def test_write_output_json_format(cortex_service, mock_journal):
    """Test writing agent output in JSON format."""
    test_date = "20240115"
    from datetime import datetime as dt

    mock_dt = dt(2024, 1, 15, 12, 0, 0)
    with patch("think.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt

        agent_id = "json_agent"
        result = '{"key": "value"}'
        config = {"output": "json", "name": "data_agent"}

        cortex_service._write_output(agent_id, result, config)

        # Check file was created with .json extension
        expected_path = mock_journal / test_date / "agents" / "data_agent.json"
        assert expected_path.exists()
        assert expected_path.read_text() == result


def test_monitor_stdout_with_output(cortex_service, mock_journal):
    """Test monitor_stdout writes output when output field is present."""
    from think.cortex import AgentProcess

    # Create agent with output in request
    agent_id = "output_test"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"

    # Store request with output field (format only, path derived from name)
    cortex_service.agent_requests = {
        agent_id: {
            "event": "request",
            "prompt": "test",
            "output": "md",
            "name": "test_agent",
        }
    }

    # Create mock process with stdout (MockPipe supports context manager protocol)
    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Test result"}\n',
    ]
    mock_process.stdout = MockPipe(mock_stdout)
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

    # Check result was written to agents/ with name-derived filename
    output_path = mock_journal / test_date / "agents" / "test_agent.md"
    assert output_path.exists()
    assert output_path.read_text() == "Test result"


def test_monitor_stdout_with_output_and_day(cortex_service, mock_journal):
    """Test monitor_stdout writes output to specific day when day field is present."""
    from think.cortex import AgentProcess

    # Create agent with output and day in request
    agent_id = "output_day_test"
    active_path = mock_journal / "agents" / f"{agent_id}_active.jsonl"
    specified_day = "20240220"

    # Store request with output and day fields
    cortex_service.agent_requests = {
        agent_id: {
            "event": "request",
            "prompt": "test",
            "output": "md",
            "name": "daily_reporter",
            "day": specified_day,
        }
    }

    # Create mock process with stdout (MockPipe supports context manager protocol)
    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Daily report content"}\n',
    ]
    mock_process.stdout = MockPipe(mock_stdout)
    mock_process.wait.return_value = 0

    agent = AgentProcess(agent_id, mock_process, active_path)

    with patch.object(cortex_service, "_complete_agent_file"):
        with patch.object(cortex_service, "_has_finish_event", return_value=True):
            cortex_service._monitor_stdout(agent)

    # Check result was written to specified day's agents directory
    output_path = mock_journal / specified_day / "agents" / "daily_reporter.md"
    assert output_path.exists()
    assert output_path.read_text() == "Daily report content"


def test_recover_orphaned_agents(cortex_service, mock_journal):
    """Test recovery of orphaned active agent files."""
    # Create orphaned active files
    agents_dir = mock_journal / "agents"
    agent1_active = agents_dir / "111_active.jsonl"
    agent2_active = agents_dir / "222_active.jsonl"

    agent1_active.write_text('{"event": "start", "ts": 1000}\n')
    agent2_active.write_text('{"event": "start", "ts": 2000}\n')

    active_files = [agent1_active, agent2_active]
    cortex_service._recover_orphaned_agents(active_files)

    # Check active files were renamed to completed
    assert not agent1_active.exists()
    assert not agent2_active.exists()
    assert (agents_dir / "111.jsonl").exists()
    assert (agents_dir / "222.jsonl").exists()

    # Check error events were appended
    content1 = (agents_dir / "111.jsonl").read_text()
    lines1 = content1.strip().split("\n")
    assert len(lines1) == 2
    error_event = json.loads(lines1[1])
    assert error_event["event"] == "error"
    assert "Recovered" in error_event["error"]
    assert error_event["agent_id"] == "111"

    content2 = (agents_dir / "222.jsonl").read_text()
    lines2 = content2.strip().split("\n")
    assert len(lines2) == 2
    assert json.loads(lines2[1])["event"] == "error"
