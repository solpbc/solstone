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


# Direct tests of the resolver primitives — these are mocked away in every
# end-to-end test above, so the parsing logic itself needs explicit coverage.


class _FakePwEntry:
    def __init__(self, pw_gecos: str = "", pw_name: str = ""):
        self.pw_gecos = pw_gecos
        self.pw_name = pw_name


def test_resolve_os_identity_linux_gecos(monkeypatch):
    monkeypatch.setattr(
        utils.pwd, "getpwuid", lambda _uid: _FakePwEntry("Jane Doe,,,,", "jane")
    )
    assert utils._resolve_os_identity() == ("Jane Doe", "jane")


def test_resolve_os_identity_macos_single_name(monkeypatch):
    monkeypatch.setattr(
        utils.pwd, "getpwuid", lambda _uid: _FakePwEntry("Jane Doe", "jane")
    )
    assert utils._resolve_os_identity() == ("Jane Doe", "jane")


def test_resolve_os_identity_empty_gecos(monkeypatch):
    monkeypatch.setattr(
        utils.pwd, "getpwuid", lambda _uid: _FakePwEntry("", "jane")
    )
    assert utils._resolve_os_identity() == ("", "jane")


def test_resolve_os_identity_comma_only_gecos(monkeypatch):
    monkeypatch.setattr(
        utils.pwd, "getpwuid", lambda _uid: _FakePwEntry(",,,,", "jane")
    )
    assert utils._resolve_os_identity() == ("", "jane")


def test_resolve_os_identity_gecos_whitespace(monkeypatch):
    monkeypatch.setattr(
        utils.pwd, "getpwuid", lambda _uid: _FakePwEntry("  Jane Doe  ,extra", "jane")
    )
    assert utils._resolve_os_identity() == ("Jane Doe", "jane")


def test_resolve_os_identity_keyerror(monkeypatch):
    def _raise(_uid):
        raise KeyError("no such uid")

    monkeypatch.setattr(utils.pwd, "getpwuid", _raise)
    assert utils._resolve_os_identity() == ("", "")


def test_zone_from_localtime_path_linux():
    assert (
        utils._zone_from_localtime_path("/usr/share/zoneinfo/America/Denver")
        == "America/Denver"
    )


def test_zone_from_localtime_path_macos():
    assert (
        utils._zone_from_localtime_path(
            "/var/db/timezone/zoneinfo/America/Los_Angeles"
        )
        == "America/Los_Angeles"
    )


def test_zone_from_localtime_path_nested_zone():
    assert (
        utils._zone_from_localtime_path("/usr/share/zoneinfo/Etc/GMT+7") == "Etc/GMT+7"
    )


def test_zone_from_localtime_path_no_zoneinfo_segment():
    assert utils._zone_from_localtime_path("/etc/localtime") == ""
    assert utils._zone_from_localtime_path("/var/db/timezone/icu/icudt68l.dat") == ""
