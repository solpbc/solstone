# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.pipeline_health."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from think.pipeline_health import pipeline_status_message, summarize_pipeline_day


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


@pytest.fixture
def pipeline_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return journal


def test_empty_day_is_healthy(pipeline_journal):
    summary = summarize_pipeline_day("20260101")

    assert summary["status"] == "healthy"
    assert summary["anomalies"] == []
    assert summary["agents"] == {
        "dispatched": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "failed_list": [],
        "failed_list_truncated": False,
    }
    assert summary["activities"] == {
        "detected": 0,
        "persisted": 0,
        "agents_fired": False,
    }
    assert all(run == {"count": 0, "duration_ms_total": 0} for run in summary["runs"].values())


def test_missing_health_dir(pipeline_journal):
    (pipeline_journal / "20260101").mkdir()

    summary = summarize_pipeline_day("20260101")

    assert summary["status"] == "healthy"
    assert summary["anomalies"] == []
    assert summary["runs"]["daily"]["count"] == 0


def test_healthy_day_with_all_modes(pipeline_journal):
    day = "20990101"
    base = pipeline_journal / day / "health"
    _write_jsonl(
        base / "1_segment_dream.jsonl",
        [
            {"event": "run.start", "mode": "segment"},
            {"event": "agent.dispatch", "mode": "segment"},
            {"event": "agent.complete", "mode": "segment"},
            {"event": "run.complete", "mode": "segment", "duration_ms": 10},
        ],
    )
    _write_jsonl(
        base / "2_daily_dream.jsonl",
        [
            {"event": "run.start", "mode": "daily"},
            {"event": "agent.dispatch", "mode": "daily"},
            {"event": "agent.complete", "mode": "daily"},
            {"event": "run.complete", "mode": "daily", "duration_ms": 20},
        ],
    )
    _write_jsonl(
        base / "3_activity_dream.jsonl",
        [
            {"event": "run.start", "mode": "activity"},
            {"event": "agent.dispatch", "mode": "activity"},
            {"event": "agent.complete", "mode": "activity"},
            {"event": "run.complete", "mode": "activity", "duration_ms": 30},
        ],
    )

    summary = summarize_pipeline_day(day)

    assert summary["status"] == "healthy"
    assert summary["agents"]["dispatched"] == 3
    assert summary["agents"]["completed"] == 3
    assert summary["runs"]["segment"] == {"count": 1, "duration_ms_total": 10}
    assert summary["runs"]["daily"] == {"count": 1, "duration_ms_total": 20}
    assert summary["runs"]["activity"] == {"count": 1, "duration_ms_total": 30}
    assert summary["activities"]["agents_fired"] is True


def test_agent_failure_promotes_warning(pipeline_journal):
    day = "20990102"
    _write_jsonl(
        pipeline_journal / day / "health" / "1_segment_dream.jsonl",
        [
            {
                "event": "agent.fail",
                "mode": "segment",
                "name": "screen",
                "agent_id": "a-1",
                "state": "timeout",
            }
        ],
    )

    summary = summarize_pipeline_day(day)

    assert summary["status"] == "warning"
    assert summary["agents"]["failed"] == 1
    assert summary["agents"]["failed_list"] == [
        {"mode": "segment", "name": "screen", "agent_id": "a-1", "state": "timeout"}
    ]
    assert summary["anomalies"] == [
        {
            "kind": "agent_failure",
            "mode": "segment",
            "name": "screen",
            "agent_id": "a-1",
            "state": "timeout",
        }
    ]


def test_failed_list_truncates_at_20(pipeline_journal):
    day = "20990103"
    events = [
        {
            "event": "agent.fail",
            "mode": "daily",
            "name": f"agent-{idx}",
            "agent_id": f"id-{idx}",
            "state": "error",
        }
        for idx in range(25)
    ]
    _write_jsonl(pipeline_journal / day / "health" / "1_daily_dream.jsonl", events)

    summary = summarize_pipeline_day(day)

    assert summary["agents"]["failed"] == 25
    assert len(summary["agents"]["failed_list"]) == 20
    assert summary["agents"]["failed_list_truncated"] is True
    assert sum(1 for a in summary["anomalies"] if a["kind"] == "agent_failure") == 20


def test_activity_detected_without_run_is_stale(pipeline_journal):
    day = "20990104"
    _write_jsonl(
        pipeline_journal / day / "health" / "1_segment_dream.jsonl",
        [{"event": "activity.detected", "mode": "segment"}],
    )

    summary = summarize_pipeline_day(day)

    assert summary["status"] == "stale"
    assert {"kind": "activity_agents_missing"} in summary["anomalies"]


def test_past_day_without_daily_run_is_stale(pipeline_journal, monkeypatch):
    day = "20200101"
    _write_jsonl(
        pipeline_journal / day / "health" / "1_segment_dream.jsonl",
        [{"event": "run.start", "mode": "segment"}],
    )
    monkeypatch.setattr(
        "think.pipeline_health._now", lambda: datetime(2020, 1, 2, 12, 0, 0)
    )

    summary = summarize_pipeline_day(day)

    assert summary["status"] == "stale"
    assert {"kind": "daily_agents_missing"} in summary["anomalies"]


def test_today_before_23h_no_daily_run_is_healthy(pipeline_journal, monkeypatch):
    current = datetime(2026, 4, 16, 12, 0, 0)
    monkeypatch.setattr("think.pipeline_health._now", lambda: current)
    (pipeline_journal / current.strftime("%Y%m%d") / "health").mkdir(parents=True)

    summary = summarize_pipeline_day(current.strftime("%Y%m%d"))

    assert summary["status"] == "healthy"
    assert {"kind": "daily_agents_missing"} not in summary["anomalies"]


def test_today_after_23h_no_daily_run_is_stale(pipeline_journal, monkeypatch):
    current = datetime(2026, 4, 16, 23, 30, 0)
    monkeypatch.setattr("think.pipeline_health._now", lambda: current)
    (pipeline_journal / current.strftime("%Y%m%d") / "health").mkdir(parents=True)

    summary = summarize_pipeline_day(current.strftime("%Y%m%d"))

    assert summary["status"] == "stale"
    assert {"kind": "daily_agents_missing"} in summary["anomalies"]


def test_segment_runs_missing_is_soft(pipeline_journal, monkeypatch):
    day = "20990105"
    (pipeline_journal / day / "health").mkdir(parents=True)
    monkeypatch.setattr(
        "think.pipeline_health.iter_segments",
        lambda value: [("default", "120000_300", Path("/tmp/fake"))],
    )

    summary = summarize_pipeline_day(day)

    assert summary["status"] == "healthy"
    assert {"kind": "segment_runs_missing"} in summary["anomalies"]


def test_invalid_day_returns_healthy_empty(pipeline_journal):
    summary = summarize_pipeline_day("not-a-date")

    assert summary["status"] == "healthy"
    assert summary["anomalies"] == []
    assert summary["agents"] == {
        "dispatched": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "failed_list": [],
        "failed_list_truncated": False,
    }


def test_malformed_json_lines_skipped(pipeline_journal):
    day = "20990106"
    path = pipeline_journal / day / "health" / "1_segment_dream.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"event": "run.start", "mode": "segment"})
        + "\nnot json at all\n"
        + json.dumps({"event": "agent.dispatch", "mode": "segment"})
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_pipeline_day(day)

    assert summary["runs"]["segment"]["count"] == 1
    assert summary["agents"]["dispatched"] == 1


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({"status": "healthy", "anomalies": [], "agents": {"failed": 0}, "day": "20260101"}, None),
        (
            {
                "status": "stale",
                "anomalies": [
                    {"kind": "activity_agents_missing"},
                    {"kind": "daily_agents_missing"},
                    {"kind": "agent_failure"},
                ],
                "agents": {"failed": 3},
                "day": "20260101",
            },
            {
                "status": "stale",
                "message": "Activity agents didn't run — persisted activities untouched.",
            },
        ),
        (
            {
                "status": "stale",
                "anomalies": [
                    {"kind": "daily_agents_missing"},
                    {"kind": "agent_failure"},
                ],
                "agents": {"failed": 2},
                "day": "20260102",
            },
            {
                "status": "stale",
                "message": "Daily dream didn't run for 20260102.",
            },
        ),
        (
            {
                "status": "warning",
                "anomalies": [{"kind": "agent_failure"}],
                "agents": {"failed": 1},
                "day": "20260101",
            },
            {"status": "warning", "message": "1 dream agent failed."},
        ),
        (
            {
                "status": "warning",
                "anomalies": [{"kind": "agent_failure"}] * 3,
                "agents": {"failed": 3},
                "day": "20260101",
            },
            {"status": "warning", "message": "3 dream agents failed."},
        ),
        (
            {
                "status": "healthy",
                "anomalies": [{"kind": "segment_runs_missing"}],
                "agents": {"failed": 0},
                "day": "20260101",
            },
            None,
        ),
    ],
)
def test_status_message_priorities(summary, expected):
    assert pipeline_status_message(summary) == expected
