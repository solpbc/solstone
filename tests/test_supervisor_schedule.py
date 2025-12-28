"""Test supervisor scheduling functionality."""

import asyncio
import os
from unittest.mock import Mock, patch

import pytest

from think.supervisor import spawn_scheduled_agents


@patch("think.supervisor.day_input_summary")
@patch("think.supervisor.cortex_request")
@patch("think.supervisor.get_agents")
@pytest.mark.asyncio
async def test_spawn_scheduled_agents(
    mock_get_agents, mock_cortex_request, mock_input_summary
):
    """Test that scheduled agents are spawned correctly via Cortex."""
    from think.supervisor import check_scheduled_agents

    # Mock agents with one scheduled and one not
    mock_get_agents.return_value = {
        "todos:todo": {
            "title": "TODO Task Manager",
            "schedule": "daily",
            "backend": "openai",
            "model": "gpt-4",
        },
        "default": {
            "title": "Default Assistant",
            # No schedule
        },
        "another_daily": {
            "title": "Another Daily Task",
            "schedule": "daily",  # No model specified
        },
    }

    # Mock cortex_request to return agent IDs
    mock_cortex_request.side_effect = ["123456789", "987654321"]

    # Mock input summary
    mock_input_summary.return_value = "No recordings"

    # Call the functions (prepare then execute)
    with patch.dict(os.environ, {"JOURNAL_PATH": "/test/journal"}, clear=True):
        spawn_scheduled_agents()
        await check_scheduled_agents()

    # Should spawn 2 agents (todo and another_daily)
    assert mock_cortex_request.call_count == 2

    # Check first request call (todos:todo) - includes input summary
    first_call = mock_cortex_request.call_args_list[0]
    assert first_call[1]["persona"] == "todos:todo"
    assert "Running daily scheduled task" in first_call[1]["prompt"]
    assert "No recordings" in first_call[1]["prompt"]

    # Check second request call (another_daily) - includes input summary
    second_call = mock_cortex_request.call_args_list[1]
    assert second_call[1]["persona"] == "another_daily"
    assert "Running daily scheduled task" in second_call[1]["prompt"]
    assert "No recordings" in second_call[1]["prompt"]


@patch("think.supervisor.check_scheduled_agents")
@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_dream")
def test_supervisor_runs_scheduled_after_dream(
    mock_run_dream, mock_spawn_scheduled, mock_check_scheduled, tmp_path, mock_callosum
):
    """Test that scheduled agents run only after successful dream."""
    from think.supervisor import supervise

    # Test successful dream
    mock_run_dream.return_value = True

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.asyncio.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers dream
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                with patch.dict(
                    os.environ, {"JOURNAL_PATH": str(tmp_path)}, clear=True
                ):
                    try:
                        asyncio.run(supervise(daily=True))
                    except KeyboardInterrupt:
                        pass

                mock_run_dream.assert_called_once()
                mock_spawn_scheduled.assert_called_once_with()


@patch("think.supervisor.check_scheduled_agents")
@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_dream")
def test_supervisor_skips_scheduled_on_dream_failure(
    mock_run_dream, mock_spawn_scheduled, mock_check_scheduled, tmp_path, mock_callosum
):
    """Test that scheduled agents don't run if dream fails."""
    from think.supervisor import supervise

    # Test failed dream
    mock_run_dream.return_value = False

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.asyncio.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers dream
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                with patch.dict(
                    os.environ, {"JOURNAL_PATH": str(tmp_path)}, clear=True
                ):
                    try:
                        asyncio.run(supervise(daily=True))
                    except KeyboardInterrupt:
                        pass

                mock_run_dream.assert_called_once()
                mock_spawn_scheduled.assert_not_called()
