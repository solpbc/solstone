# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor daily scheduling functionality."""

from datetime import date
from unittest.mock import patch


def test_handle_daily_tasks_submits_dream_on_day_change(mock_callosum):
    """Test that handle_daily_tasks submits dream via task queue when day changes."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    # Reset state to a previous day
    _daily_state["last_day"] = date(2025, 1, 1)

    # Track submitted commands
    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    # Verify a dream task was submitted
    assert len(submitted) == 1
    assert submitted[0][1] == "dream"

    # Verify day state was updated
    assert _daily_state["last_day"] == date(2025, 1, 2)


def test_handle_daily_tasks_no_spawn_same_day(mock_callosum):
    """Test that handle_daily_tasks does not submit dream on same day."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    today = date(2025, 1, 2)

    # Set state to today
    _daily_state["last_day"] = today

    # Track submitted commands
    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = today
        handle_daily_tasks()

    # Verify no task was submitted
    assert len(submitted) == 0


def test_handle_daily_tasks_submits_correct_command(mock_callosum):
    """Test that handle_daily_tasks submits sol dream with --refresh for the previous day."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    # Track submitted commands
    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 1
    cmd = submitted[0]
    assert cmd[0] == "sol"
    assert cmd[1] == "dream"
    assert "--day" in cmd
    assert "20250101" in cmd
    assert "--refresh" in cmd


def test_handle_daily_tasks_skipped_in_remote_mode(mock_callosum):
    """Test that handle_daily_tasks skips entirely in remote mode."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    # Reset state to a previous day (would normally trigger dream)
    _daily_state["last_day"] = date(2025, 1, 1)

    # Track submitted commands
    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    # Enable remote mode (fixture resets after test)
    mod._is_remote_mode = True

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    # Verify no task was submitted (remote mode skips daily processing)
    assert len(submitted) == 0
    # State should be unchanged (early return before any state updates)
    assert _daily_state["last_day"] == date(2025, 1, 1)
