# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from think.utils import get_owner_timezone, sunday_of_week


def test_get_owner_timezone_uses_configured_zone(monkeypatch):
    monkeypatch.setattr(
        "think.utils.get_config",
        lambda: {"identity": {"timezone": "America/New_York"}},
    )

    tz = get_owner_timezone()

    assert tz == ZoneInfo("America/New_York")


def test_get_owner_timezone_falls_back_to_host_zone(monkeypatch, caplog):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 20, 12, 0, tzinfo=ZoneInfo("UTC"))

    monkeypatch.setattr(
        "think.utils.get_config",
        lambda: {"identity": {"timezone": "Mars/Olympus"}},
    )
    monkeypatch.setattr("think.utils.datetime", FixedDateTime)

    with caplog.at_level(logging.WARNING):
        tz = get_owner_timezone()

    assert tz == ZoneInfo("UTC")
    assert "falling back to host timezone" in caplog.text


def test_sunday_of_week_returns_most_recent_sunday():
    tz = ZoneInfo("America/Denver")

    assert sunday_of_week(datetime(2026, 3, 8, 9, 0), tz) == "20260308"
    assert sunday_of_week(datetime(2026, 3, 10, 9, 0), tz) == "20260308"


def _patch_weekly_runtime(
    monkeypatch,
    journal_path: Path,
    prompts: dict[str, dict],
    *,
    enabled_facets: dict[str, dict] | None = None,
    active_facets: set[str] | None = None,
) -> list[tuple[str, str, dict]]:
    captured: list[tuple[str, str, dict]] = []

    monkeypatch.setattr("think.thinking.get_owner_timezone", lambda: ZoneInfo("UTC"))
    monkeypatch.setattr("think.thinking.get_journal", lambda: str(journal_path))
    monkeypatch.setattr("think.thinking.get_talent_configs", lambda schedule: prompts)
    monkeypatch.setattr("think.thinking.day_input_summary", lambda day: "summary")
    monkeypatch.setattr(
        "think.thinking.get_enabled_facets", lambda: enabled_facets or {}
    )
    monkeypatch.setattr(
        "think.thinking.get_active_facets", lambda day: active_facets or set()
    )
    monkeypatch.setattr("think.thinking._update_status", lambda **kwargs: None)
    monkeypatch.setattr("think.thinking.emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("think.thinking._jsonl_log", lambda *args, **kwargs: None)
    monkeypatch.setattr("think.thinking._log_skip", lambda *args, **kwargs: None)

    def fake_request(prompt: str, name: str, config: dict) -> str:
        captured.append((prompt, name, config))
        return f"use-{len(captured)}"

    monkeypatch.setattr("think.thinking._cortex_request_with_retry", fake_request)
    monkeypatch.setattr(
        "think.thinking._drain_priority_batch",
        lambda spawned, *_args: (len(spawned), 0, []),
    )
    return captured


def test_run_weekly_prompts_sets_weekly_reflection_output_override(
    tmp_path, monkeypatch
):
    from think.thinking import run_weekly_prompts

    captured = _patch_weekly_runtime(
        monkeypatch,
        tmp_path / "journal",
        {"weekly_reflection": {"type": "cogitate", "priority": 90}},
    )

    success, failed, failed_names = run_weekly_prompts(
        day="20260310",
        refresh=False,
        verbose=False,
    )

    assert (success, failed, failed_names) == (1, 0, [])
    assert len(captured) == 1
    _prompt, name, config = captured[0]
    assert name == "weekly_reflection"
    assert config["day"] == "20260308"
    assert config["output"] == "md"
    assert config["output_path"] == str(
        tmp_path / "journal" / "reflections" / "weekly" / "20260308.md"
    )
    assert config["env"]["SOL_DAY"] == "20260308"
    assert config["schedule"] == "weekly"


def test_run_weekly_prompts_sets_override_for_multifacet_weekly_reflection(
    tmp_path, monkeypatch
):
    from think.thinking import run_weekly_prompts

    captured = _patch_weekly_runtime(
        monkeypatch,
        tmp_path / "journal",
        {
            "weekly_reflection": {
                "type": "cogitate",
                "priority": 90,
                "multi_facet": True,
            }
        },
        enabled_facets={"work": {"title": "Work"}},
        active_facets={"work"},
    )

    success, failed, failed_names = run_weekly_prompts(
        day="20260310",
        refresh=False,
        verbose=False,
    )

    assert (success, failed, failed_names) == (1, 0, [])
    assert len(captured) == 1
    _prompt, name, config = captured[0]
    assert name == "weekly_reflection"
    assert config["facet"] == "work"
    assert config["day"] == "20260308"
    assert config["output"] == "md"
    assert config["output_path"] == str(
        tmp_path / "journal" / "reflections" / "weekly" / "20260308.md"
    )
    assert config["env"]["SOL_DAY"] == "20260308"
    assert config["env"]["SOL_FACET"] == "work"
