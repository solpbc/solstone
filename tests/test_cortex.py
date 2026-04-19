# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the file-based Cortex agent manager."""

import json
import os
import sys
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
    agents_path = journal_path / "talents"
    agents_path.mkdir()

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_path))
    return journal_path


@pytest.fixture
def cortex_service(mock_journal):
    """Create a CortexService instance for testing."""
    from think.cortex import CortexService

    return CortexService(str(mock_journal))


def test_agent_process_creation():
    """Test TalentProcess class initialization and methods."""
    from think.cortex import TalentProcess

    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Running
    mock_process.pid = 12345

    log_path = Path("/tmp/test.jsonl")
    agent = TalentProcess("123456789", mock_process, log_path)

    assert agent.use_id == "123456789"
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
    assert cortex_service.talents_dir == mock_journal / "talents"
    assert cortex_service.running_uses == {}
    assert cortex_service.talents_dir.exists()


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

    use_id = "123456789"
    file_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    request = {
        "event": "request",
        "ts": 123456789,
        "prompt": "Test prompt",
        "provider": "openai",
        "name": "unified",
        "model": GPT_5,
    }

    cortex_service._spawn_subprocess(
        use_id,
        file_path,
        request,
        [sys.executable, "-m", "think.talents"],
        "talent",
    )

    # Check subprocess was called
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    assert call_args[0][0] == [sys.executable, "-m", "think.talents"]
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
    assert ndjson["name"] == "unified"
    assert ndjson["model"] == GPT_5

    # Check stdin was closed
    mock_process.stdin.close.assert_called_once()

    # Check agent was tracked
    assert use_id in cortex_service.running_uses
    agent = cortex_service.running_uses[use_id]
    assert agent.use_id == use_id
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

    use_id = "987654321"
    file_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    # Generator config has "output" instead of "tools"
    config = {
        "event": "request",
        "ts": 987654321,
        "name": "work",
        "day": "20240101",
        "output": "md",
    }

    # Generators route through _spawn_subprocess
    cortex_service._spawn_subprocess(
        use_id,
        file_path,
        config,
        [sys.executable, "-m", "think.talents"],
        "talent",
    )

    # Check subprocess was called with agents command (generators route through agents)
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    assert call_args[0][0] == [sys.executable, "-m", "think.talents"]
    assert call_args[1]["stdin"] is not None
    assert call_args[1]["stdout"] is not None
    assert call_args[1]["stderr"] is not None

    # Check NDJSON was written to stdin
    mock_process.stdin.write.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    ndjson = json.loads(written_data.strip())
    assert ndjson["event"] == "request"
    assert ndjson["name"] == "work"
    assert ndjson["day"] == "20240101"
    assert ndjson["output"] == "md"

    # Check stdin was closed
    mock_process.stdin.close.assert_called_once()

    # Check generator was tracked
    assert use_id in cortex_service.running_uses
    agent = cortex_service.running_uses[use_id]
    assert agent.use_id == use_id
    assert agent.log_path == file_path

    # Check monitoring threads were started
    assert mock_thread.call_count == 2  # stdout and stderr

    # Check timer was created and started
    mock_timer.assert_called_once()
    mock_timer_instance.start.assert_called_once()


@patch("think.talent.get_talent")
@patch("think.cortex.subprocess.Popen")
@patch("think.cortex.threading.Thread")
@patch("think.cortex.threading.Timer")
def test_spawn_subprocess_uses_cwd_from_talent(
    mock_timer,
    mock_thread,
    mock_popen,
    mock_get_agent,
    cortex_service,
    mock_journal,
):
    mock_process = MagicMock()
    mock_process.pid = 24680
    mock_process.poll.return_value = None
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_popen.return_value = mock_process
    mock_get_agent.return_value = {"type": "cogitate", "cwd": "journal"}

    mock_timer_instance = MagicMock()
    mock_timer.return_value = mock_timer_instance

    use_id = "24680"
    file_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    request = {
        "event": "request",
        "ts": 24680,
        "prompt": "Test prompt",
        "provider": "openai",
        "name": "unified",
        "model": GPT_5,
    }

    cortex_service._spawn_subprocess(
        use_id,
        file_path,
        request,
        [sys.executable, "-m", "think.talents"],
        "talent",
    )

    assert mock_popen.call_args.kwargs["cwd"] == str(mock_journal)


@patch("think.talent.get_talent")
@patch("think.cortex.subprocess.Popen")
@patch("think.cortex.threading.Thread")
@patch("think.cortex.threading.Timer")
def test_spawn_subprocess_skips_cwd_for_generate(
    mock_timer,
    mock_thread,
    mock_popen,
    mock_get_agent,
    cortex_service,
    mock_journal,
):
    mock_process = MagicMock()
    mock_process.pid = 13579
    mock_process.poll.return_value = None
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    mock_popen.return_value = mock_process
    mock_get_agent.return_value = {"type": "generate"}

    mock_timer_instance = MagicMock()
    mock_timer.return_value = mock_timer_instance

    use_id = "13579"
    file_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    request = {
        "event": "request",
        "ts": 13579,
        "name": "decisions",
        "day": "20240101",
        "output": "md",
    }

    cortex_service._spawn_subprocess(
        use_id,
        file_path,
        request,
        [sys.executable, "-m", "think.talents"],
        "talent",
    )

    assert mock_popen.call_args.kwargs["cwd"] is None


def test_monitor_stdout_json_events(cortex_service, mock_journal):
    """Test monitoring stdout with JSON events."""
    from io import StringIO

    from think.cortex import TalentProcess

    use_id = "123456789"
    log_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Process exits
    mock_process.stdout = StringIO(
        '{"event": "start", "ts": 1234567890}\n'
        '{"event": "finish", "ts": 1234567891, "result": "Done"}\n'
    )

    agent = TalentProcess(use_id, mock_process, log_path)
    cortex_service.running_uses[use_id] = agent

    with patch.object(cortex_service, "_complete_use_file") as mock_complete:
        cortex_service._monitor_stdout(agent)

        # Check events were written to file
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "start"
        assert json.loads(lines[1])["event"] == "finish"

        # Check file was completed
        mock_complete.assert_called_once_with(use_id, log_path)

    # Check agent was removed
    assert use_id not in cortex_service.running_uses


def test_monitor_stdout_non_json_output(cortex_service, mock_journal):
    """Test monitoring stdout with non-JSON output."""
    from io import StringIO

    from think.cortex import TalentProcess

    use_id = "123456789"
    log_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.stdout = StringIO(
        'Plain text output\n{"event": "finish", "ts": 1234567890}\n'
    )

    agent = TalentProcess(use_id, mock_process, log_path)
    cortex_service.running_uses[use_id] = agent

    with patch.object(cortex_service, "_complete_use_file"):
        cortex_service._monitor_stdout(agent)

        # Check info event was created for non-JSON
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        info_event = json.loads(lines[0])
        assert info_event["event"] == "info"
        assert info_event["message"] == "Plain text output"
        assert "ts" in info_event


def test_monitor_stdout_no_finish_event(cortex_service, mock_journal):
    """Test monitoring stdout when process exits without finish event."""
    from io import StringIO

    from think.cortex import TalentProcess

    use_id = "123456789"
    log_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.wait.return_value = 1  # Non-zero exit
    mock_process.stdout = StringIO('{"event": "start", "ts": 1234567890}\n')

    agent = TalentProcess(use_id, mock_process, log_path)
    cortex_service.running_uses[use_id] = agent

    with patch.object(cortex_service, "_complete_use_file"):
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

    from think.cortex import TalentProcess

    use_id = "123456789"
    log_path = mock_journal / "talents" / f"{use_id}_active.jsonl"

    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Error exit
    mock_process.stderr = StringIO(
        "Error: Something went wrong\nStack trace line 1\nStack trace line 2\n"
    )

    agent = TalentProcess(use_id, mock_process, log_path)

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
    file_path = mock_journal / "talents" / "test.jsonl"

    # File with finish event
    file_path.write_text(
        '{"event": "start", "ts": 123}\n{"event": "finish", "ts": 124}\n'
    )
    assert cortex_service._has_finish_event(file_path) is True

    # File with error event
    file_path.write_text(
        '{"event": "start", "ts": 123}\n{"event": "error", "ts": 124}\n'
    )
    assert cortex_service._has_finish_event(file_path) is True

    # File without finish/error
    file_path.write_text('{"event": "start", "ts": 123}\n')
    assert cortex_service._has_finish_event(file_path) is False

    # Empty file
    file_path.write_text("")
    assert cortex_service._has_finish_event(file_path) is False


def test_complete_use_file(cortex_service, mock_journal):
    """Test completing an agent file (rename from active to completed)."""
    use_id = "123456789"
    unified_dir = mock_journal / "talents" / "unified"
    unified_dir.mkdir()
    active_path = unified_dir / f"{use_id}_active.jsonl"
    active_path.touch()
    cortex_service.use_requests[use_id] = {"name": "unified", "use_id": use_id}

    cortex_service._complete_use_file(use_id, active_path)

    # Check file was renamed
    assert not active_path.exists()
    completed_path = unified_dir / f"{use_id}.jsonl"
    assert completed_path.exists()
    symlink_path = mock_journal / "talents" / "unified.log"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"unified/{use_id}.jsonl"


def test_complete_use_file_replaces_symlink(cortex_service, mock_journal):
    """Test completing agent file replaces convenience symlink for same name."""
    unified_dir = mock_journal / "talents" / "unified"
    unified_dir.mkdir()

    first_agent_id = "111"
    first_active_path = unified_dir / f"{first_agent_id}_active.jsonl"
    first_active_path.touch()
    cortex_service.use_requests[first_agent_id] = {"name": "unified"}

    cortex_service._complete_use_file(first_agent_id, first_active_path)

    second_agent_id = "222"
    second_active_path = unified_dir / f"{second_agent_id}_active.jsonl"
    second_active_path.touch()
    cortex_service.use_requests[second_agent_id] = {"name": "unified"}

    cortex_service._complete_use_file(second_agent_id, second_active_path)

    symlink_path = mock_journal / "talents" / "unified.log"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"unified/{second_agent_id}.jsonl"


def test_complete_use_file_colon_name(cortex_service, mock_journal):
    """Test completing agent file sanitizes colon in convenience symlink name."""
    use_id = "123456789"
    entities_dir = mock_journal / "talents" / "entities--entity_assist"
    entities_dir.mkdir()
    active_path = entities_dir / f"{use_id}_active.jsonl"
    active_path.touch()
    cortex_service.use_requests[use_id] = {"name": "entities:entity_assist"}

    cortex_service._complete_use_file(use_id, active_path)

    symlink_path = mock_journal / "talents" / "entities--entity_assist.log"
    assert symlink_path.is_symlink()
    assert os.readlink(symlink_path) == f"entities--entity_assist/{use_id}.jsonl"


def test_complete_use_file_no_name(cortex_service, mock_journal):
    """Test completing agent file skips symlink when request name is missing."""
    use_id = "123456789"
    active_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    active_path.touch()

    cortex_service._complete_use_file(use_id, active_path)

    completed_path = mock_journal / "talents" / f"{use_id}.jsonl"
    assert completed_path.exists()
    assert not any(path.is_symlink() for path in (mock_journal / "talents").iterdir())


def test_write_error_and_complete(cortex_service, mock_journal):
    """Test writing error and completing file."""
    use_id = "123456789"
    file_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    file_path.touch()

    cortex_service._write_error_and_complete(file_path, "Test error message")

    # Check error was written
    completed_path = mock_journal / "talents" / f"{use_id}.jsonl"
    assert completed_path.exists()
    assert not file_path.exists()

    content = completed_path.read_text()
    error_event = json.loads(content)
    assert error_event["event"] == "error"
    assert error_event["error"] == "Test error message"
    assert "ts" in error_event


def test_get_status(cortex_service):
    """Test getting service status."""
    from think.cortex import TalentProcess

    # Empty status
    status = cortex_service.get_status()
    assert status["running_uses"] == 0
    assert status["use_ids"] == []

    # Add running agents
    mock_process = MagicMock()
    agent1 = TalentProcess("111", mock_process, Path("/tmp/1.jsonl"))
    agent2 = TalentProcess("222", mock_process, Path("/tmp/2.jsonl"))

    cortex_service.running_uses["111"] = agent1
    cortex_service.running_uses["222"] = agent2

    status = cortex_service.get_status()
    assert status["running_uses"] == 2
    assert set(status["use_ids"]) == {"111", "222"}


def test_write_output(cortex_service, mock_journal):
    """Test writing agent output using explicit output_path."""
    use_id = "test_agent"
    result = "This is the agent result content"
    expected_path = mock_journal / "20240115" / "talents" / "my_agent.md"
    config = {"output": "md", "name": "my_agent", "output_path": str(expected_path)}

    cortex_service._write_output(use_id, result, config)

    assert expected_path.exists()
    assert expected_path.read_text() == result
    assert expected_path.parent.is_dir()


def test_write_output_with_error(cortex_service, mock_journal, caplog):
    """Test write output handles errors gracefully."""
    import logging

    output_path = mock_journal / "20240115" / "talents" / "test.md"
    with patch("builtins.open", side_effect=PermissionError("Cannot write")):
        with caplog.at_level(logging.ERROR):
            config = {"output": "md", "name": "test", "output_path": str(output_path)}
            cortex_service._write_output("use_id", "result", config)

    # Check error was logged but didn't raise
    assert "Failed to write talent use_id output" in caplog.text


def test_write_output_missing_path_skips(cortex_service, mock_journal, caplog):
    """Test write output skips when output_path is missing."""
    config = {"output": "md", "name": "test"}
    cortex_service._write_output("use_id", "result", config)

    # No output written, no error — silent skip is expected
    assert "Failed to write" not in caplog.text


def test_write_output_with_day_parameter(cortex_service, mock_journal):
    """Test writing agent output to a specific day directory."""
    use_id = "test_agent"
    result = "This is the agent result content"
    specified_day = "20240201"
    expected_path = mock_journal / specified_day / "talents" / "reporter.md"
    config = {
        "output": "md",
        "name": "reporter",
        "day": specified_day,
        "output_path": str(expected_path),
    }

    cortex_service._write_output(use_id, result, config)

    assert expected_path.exists()
    assert expected_path.read_text() == result
    assert expected_path.parent.is_dir()


def test_write_output_with_segment(cortex_service, mock_journal):
    """Test writing segment agent output to segment agents directory."""
    use_id = "segment_agent"
    result = "Segment analysis content"
    expected_path = mock_journal / "20240115" / "143000_600" / "talents" / "analyzer.md"
    config = {
        "output": "md",
        "name": "analyzer",
        "segment": "143000_600",
        "output_path": str(expected_path),
    }

    cortex_service._write_output(use_id, result, config)

    assert expected_path.exists()
    assert expected_path.read_text() == result


def test_write_output_json_format(cortex_service, mock_journal):
    """Test writing agent output in JSON format."""
    use_id = "json_agent"
    result = '{"key": "value"}'
    expected_path = mock_journal / "20240115" / "talents" / "data_agent.json"
    config = {
        "output": "json",
        "name": "data_agent",
        "output_path": str(expected_path),
    }

    cortex_service._write_output(use_id, result, config)

    assert expected_path.exists()
    assert expected_path.read_text() == result


def test_monitor_stdout_with_output(cortex_service, mock_journal):
    """Test monitor_stdout writes output when output_path is present."""
    from think.cortex import TalentProcess

    use_id = "output_test"
    active_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    output_path = mock_journal / "20240115" / "talents" / "test_agent.md"

    # Store request with explicit output_path
    cortex_service.use_requests = {
        use_id: {
            "event": "request",
            "prompt": "test",
            "output": "md",
            "name": "test_agent",
            "output_path": str(output_path),
        }
    }

    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Test result"}\n',
    ]
    mock_process.stdout = MockPipe(mock_stdout)
    mock_process.wait.return_value = 0

    agent = TalentProcess(use_id, mock_process, active_path)

    with patch.object(cortex_service, "_complete_use_file"):
        with patch.object(cortex_service, "_has_finish_event", return_value=True):
            cortex_service._monitor_stdout(agent)

    assert output_path.exists()
    assert output_path.read_text() == "Test result"


def test_monitor_stdout_with_output_and_day(cortex_service, mock_journal):
    """Test monitor_stdout writes output to specific day via output_path."""
    from think.cortex import TalentProcess

    use_id = "output_day_test"
    active_path = mock_journal / "talents" / f"{use_id}_active.jsonl"
    specified_day = "20240220"
    output_path = mock_journal / specified_day / "talents" / "daily_reporter.md"

    # Store request with explicit output_path and day
    cortex_service.use_requests = {
        use_id: {
            "event": "request",
            "prompt": "test",
            "output": "md",
            "name": "daily_reporter",
            "day": specified_day,
            "output_path": str(output_path),
        }
    }

    mock_process = MagicMock()
    mock_stdout = [
        '{"event": "start", "ts": 1000}\n',
        '{"event": "finish", "ts": 2000, "result": "Daily report content"}\n',
    ]
    mock_process.stdout = MockPipe(mock_stdout)
    mock_process.wait.return_value = 0

    agent = TalentProcess(use_id, mock_process, active_path)

    with patch.object(cortex_service, "_complete_use_file"):
        with patch.object(cortex_service, "_has_finish_event", return_value=True):
            cortex_service._monitor_stdout(agent)

    assert output_path.exists()
    assert output_path.read_text() == "Daily report content"


def test_recover_orphaned_uses(cortex_service, mock_journal):
    """Test recovery of orphaned active agent files."""
    # Create orphaned active files
    talents_dir = mock_journal / "talents"
    unified_dir = talents_dir / "unified"
    unified_dir.mkdir()
    agent1_active = unified_dir / "111_active.jsonl"
    agent2_active = unified_dir / "222_active.jsonl"

    agent1_active.write_text('{"event": "start", "ts": 1000}\n')
    agent2_active.write_text('{"event": "start", "ts": 2000}\n')

    active_files = [agent1_active, agent2_active]
    cortex_service._recover_orphaned_uses(active_files)

    # Check active files were renamed to completed
    assert not agent1_active.exists()
    assert not agent2_active.exists()
    assert (unified_dir / "111.jsonl").exists()
    assert (unified_dir / "222.jsonl").exists()

    # Check error events were appended
    content1 = (unified_dir / "111.jsonl").read_text()
    lines1 = content1.strip().split("\n")
    assert len(lines1) == 2
    error_event = json.loads(lines1[1])
    assert error_event["event"] == "error"
    assert "Recovered" in error_event["error"]
    assert error_event["use_id"] == "111"

    content2 = (unified_dir / "222.jsonl").read_text()
    lines2 = content2.strip().split("\n")
    assert len(lines2) == 2
    assert json.loads(lines2[1])["event"] == "error"
