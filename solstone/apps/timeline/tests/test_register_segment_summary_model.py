# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for timeline segment-summary provider context registration."""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from solstone.apps.timeline.tests.conftest import write_json

mod = importlib.import_module(
    "solstone.apps.timeline.maint.002_register_segment_summary_model"
)


def _journal_config_path(journal):
    return journal / "config" / "journal.json"


def test_adds_providers_contexts_entry_when_missing(timeline_journal):
    write_json(_journal_config_path(timeline_journal), {"providers": {"contexts": {}}})

    summary = mod.run_registration(timeline_journal)

    data = json.loads(_journal_config_path(timeline_journal).read_text())
    assert summary.added == 1
    assert data["providers"]["contexts"][mod.CONTEXT_NAME] == mod.EXPECTED_CONTEXT


def test_idempotent_when_present_and_matches(timeline_journal, monkeypatch):
    write_json(
        _journal_config_path(timeline_journal),
        {"providers": {"contexts": {mod.CONTEXT_NAME: mod.EXPECTED_CONTEXT}}},
    )
    monkeypatch.setattr(
        mod, "_atomic_write_json", lambda *args, **kwargs: pytest.fail("rewrite")
    )

    summary = mod.run_registration(timeline_journal)

    assert summary.preserved == 1


def test_warns_and_preserves_divergent_model(timeline_journal, monkeypatch):
    data = {
        "providers": {
            "contexts": {
                mod.CONTEXT_NAME: {"provider": "google", "model": "different-model"}
            }
        }
    }
    write_json(_journal_config_path(timeline_journal), data)
    monkeypatch.setattr(
        mod, "_atomic_write_json", lambda *args, **kwargs: pytest.fail("rewrite")
    )

    summary = mod.run_registration(timeline_journal)

    assert summary.warnings == 1
    assert json.loads(_journal_config_path(timeline_journal).read_text()) == data


def test_creates_providers_contexts_when_missing_section(timeline_journal):
    write_json(_journal_config_path(timeline_journal), {"identity": {"name": "Test"}})

    summary = mod.run_registration(timeline_journal)

    data = json.loads(_journal_config_path(timeline_journal).read_text())
    assert summary.added == 1
    assert data["providers"]["contexts"][mod.CONTEXT_NAME] == mod.EXPECTED_CONTEXT


def test_malformed_json_exits_nonzero(timeline_journal, monkeypatch):
    _journal_config_path(timeline_journal).write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["register-segment-summary-model"])

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 1
