# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for convey configuration validation and defaults."""

from __future__ import annotations

import json
from pathlib import Path


def _set_convey_journal(monkeypatch, tmp_path: Path) -> Path:
    from solstone.convey import state

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(state, "journal_root", str(tmp_path))
    return tmp_path


def _write_convey_config(journal: Path, payload: dict) -> None:
    config_path = journal / "config" / "convey.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_reporting_enabled_defaults_true_when_absent(monkeypatch, tmp_path):
    from solstone.convey.config import reporting_enabled

    _set_convey_journal(monkeypatch, tmp_path)

    assert reporting_enabled() is True


def test_reporting_enabled_round_trips_false(monkeypatch, tmp_path):
    from solstone.convey.config import (
        load_convey_config,
        reporting_enabled,
        validate_config,
    )

    journal = _set_convey_journal(monkeypatch, tmp_path)
    payload = {"reporting": {"enabled": False}}
    valid, error = validate_config(payload)
    assert valid is True
    assert error is None

    _write_convey_config(journal, payload)

    assert load_convey_config() == payload
    assert reporting_enabled() is False


def test_validate_config_rejects_non_dict_reporting():
    from solstone.convey.config import validate_config

    valid, error = validate_config({"reporting": "yes"})

    assert valid is False
    assert error == "reporting must be an object"


def test_validate_config_rejects_non_bool_enabled():
    from solstone.convey.config import validate_config

    valid, error = validate_config({"reporting": {"enabled": "no"}})

    assert valid is False
    assert error == "reporting.enabled must be a boolean"


def test_validate_config_rejects_unknown_reporting_key():
    from solstone.convey.config import validate_config

    valid, error = validate_config({"reporting": {"enabled": True, "bogus": 1}})

    assert valid is False
    assert error == "reporting contains unknown key(s): bogus"
