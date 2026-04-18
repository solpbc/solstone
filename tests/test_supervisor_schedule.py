# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test supervisor daily scheduling functionality."""

import os
from datetime import date
from unittest.mock import MagicMock, Mock, call

import pytest

import think.supervisor as mod


@pytest.fixture
def submit_mock(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(mod._task_queue, "submit", mock)
    return mock


@pytest.fixture
def set_today(monkeypatch):
    def _set_today(today):
        fake_datetime = Mock()
        fake_datetime.now.return_value.date.return_value = today
        monkeypatch.setattr(mod, "datetime", fake_datetime)

    return _set_today


def daily_complete_message(**overrides):
    message = {
        "tract": "think",
        "event": "daily_complete",
        "day": "20260318",
        "success": 3,
        "failed": 0,
        "duration_ms": 5000,
    }
    message.update(overrides)
    return message


@pytest.mark.parametrize(
    ("last_day", "today", "updated_days_return", "expected_days"),
    [
        pytest.param(
            date(2025, 1, 1),
            date(2025, 1, 2),
            ["20250101"],
            ["20250101"],
            id="one-updated-day",
        ),
        pytest.param(
            date(2025, 1, 5),
            date(2025, 1, 6),
            ["20250103", "20250104", "20250105"],
            ["20250103", "20250104", "20250105"],
            id="multiple-updated-days",
        ),
        pytest.param(
            date(2025, 1, 10),
            date(2025, 1, 11),
            [
                "20250104",
                "20250105",
                "20250106",
                "20250107",
                "20250108",
                "20250109",
                "20250110",
            ],
            ["20250107", "20250108", "20250109", "20250110"],
            id="max-updated-catchup",
        ),
    ],
)
def test_handle_daily_tasks_submits_think_runs_on_day_change(
    mock_callosum,
    monkeypatch,
    submit_mock,
    set_today,
    last_day,
    today,
    updated_days_return,
    expected_days,
):
    mod._daily_state["last_day"] = last_day
    set_today(today)
    monkeypatch.setattr(mod, "updated_days", lambda **kwargs: updated_days_return)

    mod.handle_daily_tasks()

    assert submit_mock.call_args_list == [
        call(["sol", "think", "-v", "--day", day], day=day) for day in expected_days
    ]
    assert mod._daily_state["last_day"] == today


def test_no_spawn_same_day(mock_callosum, submit_mock, set_today):
    today = date(2025, 1, 2)
    mod._daily_state["last_day"] = today
    set_today(today)

    mod.handle_daily_tasks()

    submit_mock.assert_not_called()


def test_skipped_in_remote_mode(mock_callosum, submit_mock, set_today):
    mod._daily_state["last_day"] = date(2025, 1, 1)
    mod._is_remote_mode = True
    set_today(date(2025, 1, 2))

    mod.handle_daily_tasks()

    submit_mock.assert_not_called()
    assert mod._daily_state["last_day"] == date(2025, 1, 1)


def test_advances_state_with_no_updated_days(
    mock_callosum, monkeypatch, submit_mock, set_today
):
    mod._daily_state["last_day"] = date(2025, 1, 1)
    set_today(date(2025, 1, 2))
    monkeypatch.setattr(mod, "updated_days", lambda **kwargs: [])

    mod.handle_daily_tasks()

    submit_mock.assert_not_called()
    assert mod._daily_state["last_day"] == date(2025, 1, 2)


def test_excludes_today(mock_callosum, monkeypatch, set_today):
    mod._daily_state["last_day"] = date(2025, 1, 1)
    set_today(date(2025, 1, 2))
    updated_days = MagicMock(return_value=["20250101"])
    monkeypatch.setattr(mod, "updated_days", updated_days)

    mod.handle_daily_tasks()

    assert updated_days.call_args.kwargs["exclude"] == {"20250102"}


def test_handle_think_daily_complete_submits_heartbeat(
    mock_callosum, tmp_path, monkeypatch, submit_mock
):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "health").mkdir(exist_ok=True)

    mod._handle_think_daily_complete(daily_complete_message())

    submit_mock.assert_called_once_with(["sol", "heartbeat"])


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            {"tract": "supervisor", "event": "daily_complete"}, id="wrong-tract"
        ),
        pytest.param({"tract": "think", "event": "started"}, id="wrong-event"),
        pytest.param({}, id="empty-message"),
    ],
)
def test_ignores_non_think_daily_complete(
    mock_callosum, tmp_path, monkeypatch, submit_mock, message
):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    (tmp_path / "health").mkdir(exist_ok=True)

    mod._handle_think_daily_complete(message)

    submit_mock.assert_not_called()


def test_skips_when_pid_alive(mock_callosum, tmp_path, monkeypatch, submit_mock):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    health = tmp_path / "health"
    health.mkdir(exist_ok=True)
    (health / "heartbeat.pid").write_text(str(os.getpid()))

    mod._handle_think_daily_complete(daily_complete_message())

    submit_mock.assert_not_called()


def test_proceeds_on_dead_pid(mock_callosum, tmp_path, monkeypatch, submit_mock):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    health = tmp_path / "health"
    health.mkdir(exist_ok=True)
    (health / "heartbeat.pid").write_text("99999999")

    mod._handle_think_daily_complete(daily_complete_message())

    submit_mock.assert_called_once_with(["sol", "heartbeat"])
