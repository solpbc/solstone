# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor daily scheduling functionality."""

import os
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
        patch("think.supervisor.updated_days", return_value=["20250101"]),
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
        patch("think.supervisor.updated_days", return_value=["20250101"]),
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


def test_handle_daily_tasks_multiple_updated_days_chronological(mock_callosum):
    """Updated days are submitted oldest-first so yesterday is processed last."""
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
            "think.supervisor.updated_days",
            return_value=["20250103", "20250104", "20250105"],
        ),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 6)
        handle_daily_tasks()

    assert len(submitted) == 3
    days = [cmd[cmd.index("--day") + 1] for cmd in submitted]
    assert days == ["20250103", "20250104", "20250105"]


def test_handle_daily_tasks_caps_at_max_updated_catchup(mock_callosum):
    """Only the newest MAX_UPDATED_CATCHUP days are processed."""
    import think.supervisor as mod
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 10)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    all_updated = [
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
        patch("think.supervisor.updated_days", return_value=all_updated),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 11)
        handle_daily_tasks()

    # Only newest 4
    assert len(submitted) == 4
    days = [cmd[cmd.index("--day") + 1] for cmd in submitted]
    assert days == ["20250107", "20250108", "20250109", "20250110"]


def test_handle_daily_tasks_no_updated_days(mock_callosum):
    """No submissions when there are no updated days."""
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
        patch("think.supervisor.updated_days", return_value=[]),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert len(submitted) == 0
    # State still advances even with no updated days
    assert _daily_state["last_day"] == date(2025, 1, 2)


def test_handle_daily_tasks_excludes_today(mock_callosum):
    """Today is excluded from updated_days query."""
    from think.supervisor import _daily_state, handle_daily_tasks

    _daily_state["last_day"] = date(2025, 1, 1)

    captured_exclude = {}

    def fake_updated_days(exclude=None):
        captured_exclude["value"] = exclude
        return ["20250101"]

    with (
        patch("think.supervisor.datetime") as mock_datetime,
        patch("think.supervisor.updated_days", side_effect=fake_updated_days),
    ):
        mock_datetime.now.return_value.date.return_value = date(2025, 1, 2)
        handle_daily_tasks()

    assert captured_exclude["value"] == {"20250102"}


def test_handle_dream_daily_complete_submits_heartbeat(
    mock_callosum, tmp_path, monkeypatch
):
    """_handle_dream_daily_complete submits heartbeat when no PID file exists."""
    import think.supervisor as mod

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "health").mkdir(exist_ok=True)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    message = {
        "tract": "dream",
        "event": "daily_complete",
        "day": "20260318",
        "success": 3,
        "failed": 0,
        "duration_ms": 5000,
    }
    mod._handle_dream_daily_complete(message)

    assert len(submitted) == 1
    assert submitted[0] == ["sol", "heartbeat"]


def test_handle_dream_daily_complete_ignores_wrong_event(
    mock_callosum, tmp_path, monkeypatch
):
    """_handle_dream_daily_complete ignores messages with wrong tract or event."""
    import think.supervisor as mod

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "health").mkdir(exist_ok=True)

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    mod._handle_dream_daily_complete({"tract": "supervisor", "event": "daily_complete"})
    mod._handle_dream_daily_complete({"tract": "dream", "event": "started"})
    mod._handle_dream_daily_complete({})

    assert len(submitted) == 0


def test_handle_dream_daily_complete_skips_when_pid_alive(
    mock_callosum, tmp_path, monkeypatch
):
    """_handle_dream_daily_complete does not submit when PID file shows running process."""
    import think.supervisor as mod

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    health = tmp_path / "health"
    health.mkdir(exist_ok=True)

    (health / "heartbeat.pid").write_text(str(os.getpid()))

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    message = {"tract": "dream", "event": "daily_complete", "day": "20260318"}
    mod._handle_dream_daily_complete(message)

    assert len(submitted) == 0


def test_handle_dream_daily_complete_proceeds_on_dead_pid(
    mock_callosum, tmp_path, monkeypatch
):
    """_handle_dream_daily_complete submits heartbeat when PID file has dead process."""
    import think.supervisor as mod

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    health = tmp_path / "health"
    health.mkdir(exist_ok=True)
    (health / "heartbeat.pid").write_text("99999999")

    submitted = []
    original_submit = mod._task_queue.submit

    def capture_submit(cmd, *args, **kwargs):
        submitted.append(cmd)
        return original_submit(cmd, *args, **kwargs)

    mod._task_queue.submit = capture_submit

    message = {"tract": "dream", "event": "daily_complete", "day": "20260318"}
    mod._handle_dream_daily_complete(message)

    assert len(submitted) == 1
    assert submitted[0] == ["sol", "heartbeat"]
