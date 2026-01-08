# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor scheduling functionality."""

import os
from unittest.mock import patch

import pytest

from think.supervisor import spawn_scheduled_agents


@patch("think.supervisor.day_input_summary")
@patch("think.supervisor.cortex_request")
@patch("think.supervisor.get_agents")
@pytest.mark.asyncio
async def test_spawn_scheduled_agents(
    mock_get_agents, mock_cortex_request, mock_input_summary, tmp_path
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
    with patch.dict(os.environ, {"JOURNAL_PATH": str(tmp_path)}, clear=True):
        spawn_scheduled_agents("20250101")
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


@patch("think.runner.run_task")
@patch("think.supervisor.spawn_scheduled_agents")
def test_run_daily_dream_spawns_agents_on_success(
    mock_spawn_scheduled, mock_run_task, mock_callosum
):
    """Test that _run_daily_dream spawns scheduled agents after successful dream."""
    from think.supervisor import _daily_state, _run_daily_dream

    # Reset state
    _daily_state["dream_running"] = True
    _daily_state["dream_completed"] = False

    # Mock run_task to return success
    mock_run_task.return_value = (True, 0)

    _run_daily_dream("20250101")

    # Verify state was updated
    assert _daily_state["dream_running"] is False
    assert _daily_state["dream_completed"] is True

    # Verify spawn_scheduled_agents was called with the day
    mock_spawn_scheduled.assert_called_once_with("20250101")


@patch("think.runner.run_task")
@patch("think.supervisor.spawn_scheduled_agents")
def test_run_daily_dream_skips_agents_on_failure(
    mock_spawn_scheduled, mock_run_task, mock_callosum
):
    """Test that _run_daily_dream does not spawn agents when dream fails."""
    from think.supervisor import _daily_state, _run_daily_dream

    # Reset state
    _daily_state["dream_running"] = True
    _daily_state["dream_completed"] = False

    # Mock run_task to return failure
    mock_run_task.return_value = (False, 1)

    _run_daily_dream("20250101")

    # Verify state was updated
    assert _daily_state["dream_running"] is False
    assert _daily_state["dream_completed"] is False  # Stays False on failure

    # Verify spawn_scheduled_agents was NOT called
    mock_spawn_scheduled.assert_not_called()


def test_handle_daily_tasks_spawns_dream_on_day_change(mock_callosum):
    """Test that handle_daily_tasks spawns dream thread when day changes."""
    from datetime import date

    from think.supervisor import _daily_state, handle_daily_tasks

    # Reset state to a previous day
    _daily_state["last_day"] = date(2025, 1, 1)
    _daily_state["dream_running"] = False
    _daily_state["dream_completed"] = False

    # Mock threading.Thread to capture the spawn
    spawned_threads = []

    class MockThread:
        def __init__(self, target, args=None, daemon=False):
            spawned_threads.append((target, args))
            self.target = target
            self.args = args

        def start(self):
            pass  # Don't actually start the thread

    with patch("think.supervisor.threading.Thread", MockThread):
        with patch("think.supervisor.datetime") as mock_datetime:
            mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
            handle_daily_tasks()

    # Verify a thread was spawned with the correct day argument
    assert len(spawned_threads) == 1
    target, args = spawned_threads[0]
    assert args == ("20250101",)  # The previous day (last_day) is processed
    assert _daily_state["dream_running"] is True
    assert _daily_state["last_day"] == date(2025, 1, 2)


def test_handle_daily_tasks_no_spawn_same_day(mock_callosum):
    """Test that handle_daily_tasks does not spawn dream on same day."""
    from datetime import date

    from think.supervisor import _daily_state, handle_daily_tasks

    today = date(2025, 1, 2)

    # Set state to today
    _daily_state["last_day"] = today
    _daily_state["dream_running"] = False
    _daily_state["dream_completed"] = True

    spawned_threads = []

    class MockThread:
        def __init__(self, target, args=None, daemon=False):
            spawned_threads.append((target, args))

        def start(self):
            pass

    with patch("think.supervisor.threading.Thread", MockThread):
        with patch("think.supervisor.datetime") as mock_datetime:
            mock_datetime.now.return_value.date.return_value = today
            handle_daily_tasks()

    # Verify no thread was spawned
    assert len(spawned_threads) == 0
