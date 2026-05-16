# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for convey configuration validation and defaults."""

from __future__ import annotations

import json
import logging
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


def _read_convey_config(journal: Path) -> dict:
    return json.loads((journal / "config" / "convey.json").read_text("utf-8"))


def test_reporting_enabled_defaults_true_when_absent(monkeypatch, tmp_path):
    from solstone.convey.config import reporting_enabled

    _set_convey_journal(monkeypatch, tmp_path)

    assert reporting_enabled() is True


def test_seed_default_app_navigation_seeds_absent_apps():
    from solstone.convey.config import (
        DEFAULT_RAIL_APPS,
        seed_default_app_navigation,
    )

    config = {}

    changed = seed_default_app_navigation(config)

    assert changed is True
    assert config["apps"]["starred"] == DEFAULT_RAIL_APPS
    assert config["apps"]["order"] == DEFAULT_RAIL_APPS


def test_seed_default_app_navigation_preserves_present_empty_lists():
    from solstone.convey.config import seed_default_app_navigation

    config = {"apps": {"starred": [], "order": []}}

    changed = seed_default_app_navigation(config)

    assert changed is False
    assert config["apps"]["starred"] == []
    assert config["apps"]["order"] == []


def test_seed_default_app_navigation_preserves_curated_starred():
    from solstone.convey.config import (
        DEFAULT_RAIL_APPS,
        seed_default_app_navigation,
    )

    config = {"apps": {"starred": ["chat"]}}

    changed = seed_default_app_navigation(config)

    assert changed is True
    assert config["apps"]["starred"] == ["chat"]
    assert config["apps"]["order"] == DEFAULT_RAIL_APPS


def test_seed_default_app_navigation_preserves_curated_order():
    from solstone.convey.config import (
        DEFAULT_RAIL_APPS,
        seed_default_app_navigation,
    )

    config = {"apps": {"order": ["chat"]}}

    changed = seed_default_app_navigation(config)

    assert changed is True
    assert config["apps"]["starred"] == DEFAULT_RAIL_APPS
    assert config["apps"]["order"] == ["chat"]


def test_seed_default_app_navigation_is_idempotent():
    from solstone.convey.config import (
        DEFAULT_RAIL_APPS,
        seed_default_app_navigation,
    )

    config = {}

    first_changed = seed_default_app_navigation(config)
    first_config = json.loads(json.dumps(config))
    second_changed = seed_default_app_navigation(config)

    assert first_changed is True
    assert second_changed is False
    assert config == first_config
    assert config["apps"]["starred"] == DEFAULT_RAIL_APPS
    assert config["apps"]["order"] == DEFAULT_RAIL_APPS


def test_init_finalize_seeds_default_app_navigation(journal_copy):
    from solstone.convey import create_app
    from solstone.convey.config import DEFAULT_RAIL_APPS

    (journal_copy / "config" / "convey.json").unlink()
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True

    resp = app.test_client().post(
        "/init/finalize",
        json={},
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["redirect"] == "/"

    config = _read_convey_config(journal_copy)
    assert config["apps"]["starred"] == DEFAULT_RAIL_APPS
    assert config["apps"]["order"] == DEFAULT_RAIL_APPS


def test_init_finalize_logs_convey_seed_persist_failure(
    journal_copy, monkeypatch, caplog
):
    from solstone.convey import create_app
    from solstone.convey import root as root_module

    (journal_copy / "config" / "convey.json").unlink()
    monkeypatch.setattr(root_module, "save_convey_config", lambda _config: False)
    caplog.set_level(logging.ERROR, logger="solstone.convey.root")
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True

    resp = app.test_client().post(
        "/init/finalize",
        json={},
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["redirect"] == "/"
    assert "default app navigation seed convey-config PERSIST failed" in caplog.text


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
