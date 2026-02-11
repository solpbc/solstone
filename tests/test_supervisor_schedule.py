# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor daily scheduling functionality."""

from datetime import date
from pathlib import Path
from unittest.mock import patch


def test_handle_daily_tasks_spawns_dream_on_day_change(mock_callosum):
    """Test that handle_daily_tasks spawns dream thread when day changes."""
    # Import from current module state (handles reloads from other tests)
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
    # Import from current module state (handles reloads from other tests)
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


def test_run_daily_processing_success(mock_callosum):
    """Test that _run_daily_processing updates state on success."""
    # Import from current module state (handles reloads from other tests)
    from think.supervisor import _daily_state, _run_daily_processing

    # Reset state
    _daily_state["dream_running"] = True
    _daily_state["dream_completed"] = False

    # Mock run_task to return success
    with patch("think.runner.run_task") as mock_run_task:
        mock_run_task.return_value = (True, 0, Path("/tmp/test.log"))
        _run_daily_processing("20250101")

    # Verify state was updated
    assert _daily_state["dream_running"] is False
    assert _daily_state["dream_completed"] is True

    # Verify sol dream was called with correct args
    mock_run_task.assert_called_once()
    call_args = mock_run_task.call_args[0][0]
    assert call_args[0] == "sol"
    assert call_args[1] == "dream"
    assert "-v" in call_args
    assert "--day" in call_args
    assert "20250101" in call_args
    assert "--force" in call_args


def test_run_daily_processing_failure(mock_callosum):
    """Test that _run_daily_processing handles failure correctly."""
    # Import from current module state (handles reloads from other tests)
    from think.supervisor import _daily_state, _run_daily_processing

    # Reset state
    _daily_state["dream_running"] = True
    _daily_state["dream_completed"] = False

    # Mock run_task to return failure
    with patch("think.runner.run_task") as mock_run_task:
        mock_run_task.return_value = (False, 1, Path("/tmp/test.log"))
        _run_daily_processing("20250101")

    # Verify state was updated
    assert _daily_state["dream_running"] is False
    assert _daily_state["dream_completed"] is False  # Stays False on failure


def test_handle_daily_tasks_skipped_in_remote_mode(mock_callosum):
    """Test that handle_daily_tasks skips entirely in remote mode."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    # Reset state to a previous day (would normally trigger dream)
    _daily_state["last_day"] = date(2025, 1, 1)
    _daily_state["dream_running"] = False
    _daily_state["dream_completed"] = False

    spawned_threads = []

    class MockThread:
        def __init__(self, target, args=None, daemon=False):
            spawned_threads.append((target, args))

        def start(self):
            pass

    # Enable remote mode (fixture resets after test)
    mod._is_remote_mode = True

    with patch("think.supervisor.threading.Thread", MockThread):
        with patch("think.supervisor.datetime") as mock_datetime:
            mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
            handle_daily_tasks()

    # Verify no thread was spawned (remote mode skips daily processing)
    assert len(spawned_threads) == 0
    # State should be unchanged (early return before any state updates)
    assert _daily_state["last_day"] == date(2025, 1, 1)
    assert _daily_state["dream_running"] is False
