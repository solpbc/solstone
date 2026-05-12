# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for timeline schedule registration maint task."""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from solstone.apps.timeline.tests.conftest import write_json

mod = importlib.import_module("solstone.apps.timeline.maint.001_register_schedules")


def _schedules_path(journal):
    return journal / "config" / "schedules.json"


def test_adds_both_entries_when_missing(timeline_journal):
    """AC#11."""
    write_json(_schedules_path(timeline_journal), {})

    summary = mod.run_registration(timeline_journal)

    data = json.loads(_schedules_path(timeline_journal).read_text())
    assert summary.added == 2
    assert data["timeline-rollup-day"] == mod.EXPECTED_ENTRIES["timeline-rollup-day"]
    assert (
        data["timeline-rollup-master"] == mod.EXPECTED_ENTRIES["timeline-rollup-master"]
    )


def test_idempotent_no_rewrite_when_present(timeline_journal, monkeypatch):
    """AC#12."""
    write_json(_schedules_path(timeline_journal), mod.EXPECTED_ENTRIES)
    monkeypatch.setattr(
        mod, "_atomic_write_json", lambda *args, **kwargs: pytest.fail("rewrite")
    )

    summary = mod.run_registration(timeline_journal)

    assert summary.added == 0
    assert summary.preserved == 2


def test_warns_and_preserves_divergent_cmd(timeline_journal, monkeypatch):
    """AC#13."""
    data = {
        "timeline-rollup-day": {
            "cmd": ["custom"],
            "every": "daily",
            "max_runtime": "30m",
        },
        "timeline-rollup-master": mod.EXPECTED_ENTRIES["timeline-rollup-master"],
    }
    write_json(_schedules_path(timeline_journal), data)
    monkeypatch.setattr(
        mod, "_atomic_write_json", lambda *args, **kwargs: pytest.fail("rewrite")
    )

    summary = mod.run_registration(timeline_journal)

    assert summary.warnings == 1
    assert json.loads(_schedules_path(timeline_journal).read_text()) == data


def test_preserves_disabled_entry(timeline_journal, monkeypatch):
    """AC#14."""
    data = {
        "timeline-rollup-day": {
            **mod.EXPECTED_ENTRIES["timeline-rollup-day"],
            "enabled": False,
        },
        "timeline-rollup-master": mod.EXPECTED_ENTRIES["timeline-rollup-master"],
    }
    write_json(_schedules_path(timeline_journal), data)
    monkeypatch.setattr(
        mod, "_atomic_write_json", lambda *args, **kwargs: pytest.fail("rewrite")
    )

    summary = mod.run_registration(timeline_journal)

    assert summary.preserved == 2
    assert (
        json.loads(_schedules_path(timeline_journal).read_text())[
            "timeline-rollup-day"
        ]["enabled"]
        is False
    )


def test_malformed_json_exits_nonzero(timeline_journal, monkeypatch):
    _schedules_path(timeline_journal).write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["register-schedules"])

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 1


def test_creates_schedules_json_when_missing(timeline_journal):
    summary = mod.run_registration(timeline_journal)

    assert summary.added == 2
    assert _schedules_path(timeline_journal).is_file()
