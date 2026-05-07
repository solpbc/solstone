# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solstone.think.push import config


def _write_config(tmp_path: Path, payload: dict) -> None:
    config_path = tmp_path / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_push_config_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(tmp_path, {"agent": {"name": "sol"}})

    assert config.get_apns_key_path() is None
    assert config.get_apns_key_id() is None
    assert config.get_apns_team_id() is None
    assert config.get_bundle_id() is None
    assert config.get_environment() == "development"
    assert config.is_configured() is False


def test_push_config_reads_journal_values(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    key_path = tmp_path / "keys" / "apns.p8"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("PRIVATE KEY", encoding="utf-8")
    _write_config(
        tmp_path,
        {
            "push": {
                "apns_key_path": f"  {key_path}  ",
                "apns_key_id": "  KEY123  ",
                "apns_team_id": "  TEAM123  ",
                "bundle_id": "  org.solpbc.solstone-swift  ",
                "environment": "production",
            }
        },
    )

    assert config.get_apns_key_path() == key_path
    assert config.get_apns_key_id() == "KEY123"
    assert config.get_apns_team_id() == "TEAM123"
    assert config.get_bundle_id() == "org.solpbc.solstone-swift"
    assert config.get_environment() == "production"
    assert config.is_configured() is True


def test_push_config_blank_values_normalize_to_none(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(
        tmp_path,
        {
            "push": {
                "apns_key_path": " ",
                "apns_key_id": "\t",
                "apns_team_id": "",
                "bundle_id": " ",
                "environment": " ",
            }
        },
    )

    assert config.get_apns_key_path() is None
    assert config.get_apns_key_id() is None
    assert config.get_apns_team_id() is None
    assert config.get_bundle_id() is None
    assert config.get_environment() == "development"
    assert config.is_configured() is False


def test_push_config_invalid_environment_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(tmp_path, {"push": {"environment": "staging"}})

    with pytest.raises(
        ValueError, match="push.environment must be 'development' or 'production'"
    ):
        config.get_environment()

    assert config.is_configured() is False


def test_push_config_missing_key_file_is_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _write_config(
        tmp_path,
        {
            "push": {
                "apns_key_path": str(tmp_path / "missing.p8"),
                "apns_key_id": "KEY123",
                "apns_team_id": "TEAM123",
                "bundle_id": "org.solpbc.solstone-swift",
                "environment": "development",
            }
        },
    )

    assert config.is_configured() is False


def test_push_config_relative_key_path_is_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    relative_key_path = Path("keys/apns.p8")
    _write_config(
        tmp_path,
        {
            "push": {
                "apns_key_path": str(relative_key_path),
                "apns_key_id": "KEY123",
                "apns_team_id": "TEAM123",
                "bundle_id": "org.solpbc.solstone-swift",
                "environment": "development",
            }
        },
    )

    assert config.get_apns_key_path() == relative_key_path
    assert config.is_configured() is False


def test_push_config_ignores_env_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("APNS_KEY_ID", "ENVKEY")
    _write_config(tmp_path, {"push": {}})

    assert config.get_apns_key_id() is None
    assert config.get_apns_team_id() is None
    assert config.get_bundle_id() is None
    assert config.is_configured() is False
