"""Test supervisor scheduling functionality."""

import json
import subprocess
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from think.supervisor import spawn_scheduled_agents


@patch("think.supervisor.subprocess.Popen")
@patch("think.supervisor.get_personas")
def test_spawn_scheduled_agents(mock_get_personas, mock_popen):
    """Test that scheduled agents are spawned correctly."""
    # Mock personas with one scheduled and one not
    mock_get_personas.return_value = {
        "todo": {
            "title": "TODO Task Manager",
            "config": {"schedule": "daily", "backend": "openai", "model": "gpt-4"},
        },
        "default": {
            "title": "Default Assistant",
            "config": {},  # No schedule
        },
        "another_daily": {
            "title": "Another Daily Task",
            "config": {"schedule": "daily"},  # No model specified
        },
    }

    # Mock the Popen process
    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_proc.stdin = Mock()
    mock_popen.return_value = mock_proc

    # Call the function
    spawn_scheduled_agents("/test/journal")

    # Should spawn 2 agents (todo and another_daily)
    assert mock_popen.call_count == 2

    # Check first spawn call (todo)
    first_call = mock_popen.call_args_list[0]
    assert first_call[0][0] == ["think-agents"]
    assert first_call[1]["stdin"] == subprocess.PIPE
    assert first_call[1]["env"]["JOURNAL_PATH"] == "/test/journal"

    # Check that correct NDJSON was written for todo
    written_data = mock_proc.stdin.write.call_args_list[0][0][0]
    request = json.loads(written_data.decode().strip())
    assert request["persona"] == "todo"
    assert request["backend"] == "openai"
    assert request["model"] == "gpt-4"
    assert "prompt" in request

    # Check second spawn call (another_daily)
    second_call = mock_popen.call_args_list[1]
    assert second_call[0][0] == ["think-agents"]

    # Check that correct NDJSON was written for another_daily
    written_data = mock_proc.stdin.write.call_args_list[1][0][0]
    request = json.loads(written_data.decode().strip())
    assert request["persona"] == "another_daily"
    assert request["backend"] == "openai"  # default
    assert "model" not in request  # not specified


@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_process_day")
def test_supervisor_runs_scheduled_after_process_day(
    mock_run_process_day, mock_spawn_scheduled
):
    """Test that scheduled agents run only after successful process_day."""
    from think.supervisor import supervise

    # Test successful process_day
    mock_run_process_day.return_value = True

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.time.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers process_day
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                try:
                    supervise("/test/journal", daily=True)
                except KeyboardInterrupt:
                    pass

                # Should have called process_day and spawn_scheduled_agents
                mock_run_process_day.assert_called_once()
                mock_spawn_scheduled.assert_called_once_with("/test/journal")


@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_process_day")
def test_supervisor_skips_scheduled_on_process_day_failure(
    mock_run_process_day, mock_spawn_scheduled
):
    """Test that scheduled agents don't run if process_day fails."""
    from think.supervisor import supervise

    # Test failed process_day
    mock_run_process_day.return_value = False

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.time.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers process_day
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                try:
                    supervise("/test/journal", daily=True)
                except KeyboardInterrupt:
                    pass

                # Should have called process_day but NOT spawn_scheduled_agents
                mock_run_process_day.assert_called_once()
                mock_spawn_scheduled.assert_not_called()