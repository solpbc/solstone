# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor daily scheduling functionality."""

from datetime import date
from unittest.mock import patch


def test_handle_daily_tasks_submits_dream_on_day_change(mock_callosum):
    """Test that handle_daily_tasks submits dream via task queue when day changes."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.dirty_days", return_value=["20250101"]),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 1
    assert submitted[0][1] == "dream"

    assert _daily_state["last_day"] == date(2025, 1, 2)


def test_handle_daily_tasks_no_spawn_same_day(mock_callosum):
    """Test that handle_daily_tasks does not submit dream on same day."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    today = date(2025, 1, 2)
    _daily_state["last_day"] = today

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = today
        handle_daily_tasks()

    assert len(submitted) == 0


def test_handle_daily_tasks_submits_correct_command(mock_callosum):
    """Test that handle_daily_tasks submits sol dream without --refresh (dream auto-detects)."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.dirty_days", return_value=["20250101"]),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 1
    cmd = submitted[0]
    assert cmd == ["sol", "dream", "-v", "--day", "20250101"]


def test_handle_daily_tasks_skipped_in_remote_mode(mock_callosum):
    """Test that handle_daily_tasks skips entirely in remote mode."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    mod._is_remote_mode = True

    with patch("think.supervisor.datetime") as mock_datetime:
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 0
    assert _daily_state["last_day"] == date(2025, 1, 1)


def test_handle_daily_tasks_multiple_dirty_days_chronological(mock_callosum):
    """Dirty days are submitted oldest-first so yesterday is processed last."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 5)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch(
            "think.supervisor.dirty_days",
            return_value=["20250103", "20250104", "20250105"],
        ),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 6)
        handle_daily_tasks()

    assert len(submitted) == 3
    days = [cmd[cmd.index("--day") + 1] for cmd in submitted]
    assert days == ["20250103", "20250104", "20250105"]


def test_handle_daily_tasks_caps_at_max_dirty_catchup(mock_callosum):
    """Only the newest MAX_DIRTY_CATCHUP days are processed."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 10)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    all_dirty = [
        "20250104",
        "20250105",
        "20250106",
        "20250107",
        "20250108",
        "20250109",
        "20250110",
    ]

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.dirty_days", return_value=all_dirty),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 11)
        handle_daily_tasks()

    # Only newest 4
    assert len(submitted) == 4
    days = [cmd[cmd.index("--day") + 1] for cmd in submitted]
    assert days == ["20250107", "20250108", "20250109", "20250110"]


def test_handle_daily_tasks_no_dirty_days(mock_callosum):
    """No submissions when there are no dirty days."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.dirty_days", return_value=[]),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 0
    # State still advances even with no dirty days
    assert _daily_state["last_day"] == date(2025, 1, 2)


def test_handle_daily_tasks_excludes_today(mock_callosum):
    """Today is excluded from dirty_days query."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    captured_exclude = {}

    def fake_dirty_days(exclude=None):
        captured_exclude["value"] = exclude
        return ["20250101"]

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.dirty_days", side_effect=fake_dirty_days),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert captured_exclude["value"] == {"20250102"}
