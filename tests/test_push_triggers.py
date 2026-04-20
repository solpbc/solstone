# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from think.push import triggers


def _log_path(tmp_path: Path) -> Path:
    return tmp_path / "push" / "nudge_log.jsonl"


def test_handle_briefing_finish_polls_until_briefing_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    responses = iter(
        [
            ({}, None, []),
            ({}, None, []),
            (
                {"needs_attention": "- item"},
                {"generated": "2026-04-19T06:45:00"},
                ["one"],
            ),
        ]
    )
    sent_calls: list[dict[str, object]] = []
    monkeypatch.setattr(triggers, "_load_briefing_md", lambda today: next(responses))
    monkeypatch.setattr(triggers.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(
                {"devices": devices, "payload": payload, "collapse_id": collapse_id}
            )
            or (1, 0)
        ),
    )

    triggers.handle_briefing_finish(
        {"tract": "cortex", "event": "finish", "name": "morning_briefing"}
    )

    assert len(sent_calls) == 1
    assert sent_calls[0]["collapse_id"].startswith("briefing.")
    log_lines = _log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1


def test_handle_briefing_finish_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    sent_calls: list[str] = []
    monkeypatch.setattr(
        triggers,
        "_load_briefing_md",
        lambda today: (
            {"needs_attention": "- item"},
            {"generated": "2026-04-19T06:45:00"},
            ["one"],
        ),
    )
    monkeypatch.setattr(triggers.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(collapse_id) or (1, 0)
        ),
    )

    message = {"tract": "cortex", "event": "finish", "name": "morning_briefing"}
    triggers.handle_briefing_finish(message)
    triggers.handle_briefing_finish(message)

    assert sent_calls == [sent_calls[0]]


def test_check_pre_meeting_prep_skips_muted_facets(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    monkeypatch.setattr(triggers, "get_enabled_facets", lambda: {})
    sent_calls: list[str] = []
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(collapse_id) or (1, 0)
        ),
    )

    triggers.check_pre_meeting_prep(datetime(2026, 4, 20, 8, 45, 0))

    assert sent_calls == []


def test_check_pre_meeting_prep_skips_non_anticipated(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    monkeypatch.setattr(triggers, "get_enabled_facets", lambda: {"work": {}})
    monkeypatch.setattr(
        triggers,
        "load_activity_records",
        lambda facet, day: [{"id": "meeting", "source": "cogitate", "start": "09:00"}],
    )
    sent_calls: list[str] = []
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(collapse_id) or (1, 0)
        ),
    )

    triggers.check_pre_meeting_prep(datetime(2026, 4, 20, 8, 45, 0))

    assert sent_calls == []


def test_check_pre_meeting_prep_fires_for_hhmm_and_hhmmss(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    monkeypatch.setattr(triggers, "get_enabled_facets", lambda: {"work": {}})
    monkeypatch.setattr(
        triggers,
        "load_activity_records",
        lambda facet, day: [
            {
                "id": "anticipated_meeting_090000_0420",
                "source": "anticipated",
                "start": "09:00",
                "title": "Launch sync",
            },
            {
                "id": "anticipated_call_090030_0420",
                "source": "anticipated",
                "start": "09:00:30",
                "title": "Prep call",
            },
        ],
    )
    sent_calls: list[str] = []
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(collapse_id) or (1, 0)
        ),
    )

    triggers.check_pre_meeting_prep(datetime(2026, 4, 20, 8, 45, 0))

    assert sent_calls == [
        "meeting.anticipated_meeting_090000_0420",
        "meeting.anticipated_call_090030_0420",
    ]


def test_check_pre_meeting_prep_zero_devices_skips_log(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [])
    monkeypatch.setattr(triggers, "get_enabled_facets", lambda: {"work": {}})
    monkeypatch.setattr(
        triggers,
        "load_activity_records",
        lambda facet, day: [
            {
                "id": "anticipated_meeting_090000_0420",
                "source": "anticipated",
                "start": "09:00",
                "title": "Launch sync",
            }
        ],
    )

    triggers.check_pre_meeting_prep(datetime(2026, 4, 20, 8, 45, 0))

    assert _log_path(tmp_path).exists() is False


def test_send_agent_alert_same_context_id_fires_once(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(triggers, "_eligible_devices", lambda: [{"token": "a" * 64}])
    sent_calls: list[str] = []
    monkeypatch.setattr(
        triggers,
        "send_many",
        lambda devices, payload, *, collapse_id: (
            sent_calls.append(collapse_id) or (1, 0)
        ),
    )

    first = triggers.send_agent_alert(
        title="Agent Alert", body="Needs review", context_id="ctx-1"
    )
    second = triggers.send_agent_alert(
        title="Agent Alert", body="Needs review", context_id="ctx-1"
    )

    assert first == (1, 0)
    assert second == (0, 0)
    assert sent_calls == ["alert.ctx-1"]
    lines = [
        json.loads(line)
        for line in _log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 1
