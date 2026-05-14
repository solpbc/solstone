# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import stat

import pytest

from solstone.think import utils


@pytest.fixture(autouse=True)
def reset_default_config(monkeypatch):
    monkeypatch.setattr(utils, "_default_config", None)


def _config_path(journal):
    return journal / "config" / "journal.json"


def _mock_os(monkeypatch, identity=("Test User", "tester"), timezone="America/Denver"):
    monkeypatch.setattr(utils, "_resolve_os_identity", lambda: identity)
    monkeypatch.setattr(utils, "_resolve_os_timezone", lambda: timezone)


def test_ensure_journal_config_creates_file_with_os_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _mock_os(monkeypatch)

    config = utils.ensure_journal_config()

    config_path = _config_path(tmp_path)
    assert config_path.exists()
    assert config["identity"]["name"] == "Test User"
    assert config["identity"]["preferred"] == "tester"
    assert config["identity"]["timezone"] == "America/Denver"
    assert config["convey"]["secret"]


def test_ensure_journal_config_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _mock_os(monkeypatch)

    utils.ensure_journal_config()
    config_path = _config_path(tmp_path)
    first = config_path.stat()
    utils.ensure_journal_config()
    second = config_path.stat()

    assert second.st_ino == first.st_ino
    assert second.st_size == first.st_size


def test_ensure_journal_config_file_mode_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _mock_os(monkeypatch)

    utils.ensure_journal_config()

    assert stat.S_IMODE(_config_path(tmp_path).stat().st_mode) == 0o600


def test_ensure_journal_config_identity_resolver_failure_is_isolated(
    tmp_path, monkeypatch
):
    def fail():
        raise RuntimeError("identity failed")

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(utils, "_resolve_os_identity", fail)
    monkeypatch.setattr(utils, "_resolve_os_timezone", lambda: "America/Denver")

    config = utils.ensure_journal_config()

    assert config["identity"]["name"] == ""
    assert config["identity"]["preferred"] == ""
    assert config["identity"]["timezone"] == "America/Denver"
    assert _config_path(tmp_path).exists()


def test_ensure_journal_config_timezone_resolver_failure_is_isolated(
    tmp_path, monkeypatch
):
    def fail():
        raise RuntimeError("timezone failed")

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(utils, "_resolve_os_identity", lambda: ("Test User", "tester"))
    monkeypatch.setattr(utils, "_resolve_os_timezone", fail)

    config = utils.ensure_journal_config()

    assert config["identity"]["name"] == "Test User"
    assert config["identity"]["preferred"] == "tester"
    assert config["identity"]["timezone"] == ""
    assert _config_path(tmp_path).exists()


def test_ensure_journal_config_backfills_secret_without_touching_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _mock_os(monkeypatch, identity=("OS User", "osuser"), timezone="America/New_York")
    config_path = _config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    staged = {
        "identity": {
            "name": "Existing User",
            "preferred": "Existing",
            "timezone": "UTC",
        },
        "convey": {"trust_localhost": True},
    }
    config_path.write_text(json.dumps(staged), encoding="utf-8")

    config = utils.ensure_journal_config()

    assert config["identity"] == staged["identity"]
    assert config["convey"]["trust_localhost"] is True
    assert config["convey"]["secret"]


def test_ensure_journal_config_returned_dict_does_not_mutate_defaults(
    tmp_path, monkeypatch
):
    first_journal = tmp_path / "first"
    second_journal = tmp_path / "second"
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(first_journal))
    _mock_os(monkeypatch)
    config = utils.ensure_journal_config()
    config["identity"]["name"] = "Mutated"

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(second_journal))
    fresh = utils.get_config()

    assert fresh["identity"]["name"] == ""
