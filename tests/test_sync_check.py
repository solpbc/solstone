# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from solstone.think import sync_check


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    return journal


def _reset_machine_id(monkeypatch):
    monkeypatch.setattr(sync_check, "_MACHINE_ID", None)


def _set_identity(monkeypatch, *, machine_id="self-machine-1234", hostname="self-host"):
    monkeypatch.setattr(sync_check, "get_machine_id", lambda: machine_id)
    monkeypatch.setattr(sync_check, "get_self_hostname_sanitized", lambda: hostname)
    monkeypatch.setattr(sync_check, "_solstone_version", lambda: "test-version")


def _write_foreign(
    journal: Path,
    name: str = "other-host",
    *,
    mtime: float | None = None,
    payload: dict | None = None,
    raw: bytes | None = None,
) -> Path:
    sync_dir = journal / "health" / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    path = sync_dir / f"{name}.check"
    if raw is not None:
        path.write_bytes(raw)
    else:
        data = {
            "schema": sync_check.SCHEMA_VERSION,
            "machine_id": f"{name}-machine",
            "hostname": name,
            "pid": 123,
            "wall_time": "2026-05-11T00:00:00Z",
            "solstone_version": "1.2.3",
            "journal_path": "/foreign/journal",
        }
        if payload is not None:
            data = payload
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_get_machine_id_linux_reads_machine_id(monkeypatch):
    _reset_machine_id(monkeypatch)
    monkeypatch.setattr(sync_check.sys, "platform", "linux")

    def fake_read_text(self, encoding="utf-8"):
        if str(self) == "/etc/machine-id":
            return "linux-machine\n"
        raise OSError

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert sync_check.get_machine_id() == "linux-machine"


def test_get_machine_id_darwin_reads_ioplatform_uuid(monkeypatch):
    _reset_machine_id(monkeypatch)
    monkeypatch.setattr(sync_check.sys, "platform", "darwin")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout='"IOPlatformUUID" = "ABC-123"\n')

    monkeypatch.setattr(sync_check.subprocess, "run", fake_run)

    assert sync_check.get_machine_id() == "ABC-123"


def test_get_machine_id_returns_empty_when_unavailable(monkeypatch):
    _reset_machine_id(monkeypatch)
    monkeypatch.setattr(sync_check.sys, "platform", "linux")

    def fake_read_text(self, encoding="utf-8"):
        raise OSError

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert sync_check.get_machine_id() == ""

    _reset_machine_id(monkeypatch)
    monkeypatch.setattr(sync_check.sys, "platform", "darwin")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(sync_check.subprocess, "run", fake_run)

    assert sync_check.get_machine_id() == ""


def test_get_self_hostname_sanitized_normal_weird_empty(monkeypatch):
    monkeypatch.setattr(sync_check.socket, "gethostname", lambda: "my-laptop")
    assert sync_check.get_self_hostname_sanitized() == "my-laptop"

    monkeypatch.setattr(sync_check.socket, "gethostname", lambda: "My Laptop.local!")
    assert sync_check.get_self_hostname_sanitized() == "my-laptop-local"

    monkeypatch.setattr(sync_check.socket, "gethostname", lambda: "")
    assert sync_check.get_self_hostname_sanitized() == "unknown-host"


def test_write_self_heartbeat_writes_expected_json_and_overwrites(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)

    first = sync_check.write_self_heartbeat(journal=journal)
    second = sync_check.write_self_heartbeat(journal=journal)

    assert first == second
    data = json.loads(second.read_text(encoding="utf-8"))
    assert data["schema"] == sync_check.SCHEMA_VERSION
    assert data["machine_id"] == "self-machine-1234"
    assert data["hostname"] == "self-host"
    assert data["pid"] == os.getpid()
    assert "boot" + "_id" not in data
    assert "monotonic" + "_ms" not in data
    assert data["solstone_version"] == "test-version"
    assert data["interval_seconds"] == sync_check.DEFAULT_INTERVAL_SECONDS
    assert data["journal_path"] == str(journal.resolve())
    assert second.read_text(encoding="utf-8").endswith("\n")


def test_write_self_heartbeat_tolerates_empty_machine_id(isolated_journal, monkeypatch):
    monkeypatch.setattr(sync_check, "get_machine_id", lambda: "")

    path = sync_check.write_self_heartbeat(journal=isolated_journal)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["machine_id"] == ""


def test_write_self_heartbeat_replace_failure_leaves_existing_file_intact(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    path = sync_check.write_self_heartbeat(journal=journal)
    original = path.read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(sync_check.os, "replace", fail_replace)

    with pytest.raises(OSError):
        sync_check.write_self_heartbeat(journal=journal)

    assert path.read_text(encoding="utf-8") == original
    assert list(path.parent.glob(".tmp_*.check")) == []


def test_clear_self_heartbeat_removes_file_and_missing_is_noop(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    path = sync_check.write_self_heartbeat(journal=journal)

    sync_check.clear_self_heartbeat(journal=journal)
    assert not path.exists()

    sync_check.clear_self_heartbeat(journal=journal)


def test_check_journal_sync_no_dir_empty_dir_only_self_are_clean(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)

    result = sync_check.check_journal_sync(journal=journal, now=1000.0)
    assert not result.is_conflict
    assert result.foreign_writers == ()

    (journal / "health" / "sync").mkdir(parents=True)
    result = sync_check.check_journal_sync(journal=journal, now=1000.0)
    assert not result.is_conflict
    assert result.foreign_writers == ()

    sync_check.write_self_heartbeat(journal=journal)
    result = sync_check.check_journal_sync(journal=journal, now=time.time())
    assert not result.is_conflict
    assert result.foreign_writers == ()


def test_check_journal_sync_tolerates_empty_machine_id(isolated_journal, monkeypatch):
    monkeypatch.setattr(sync_check, "get_machine_id", lambda: "")
    (isolated_journal / "health" / "sync").mkdir(parents=True)

    result = sync_check.check_journal_sync(journal=isolated_journal, now=1000.0)

    assert result.is_conflict is False
    assert isinstance(sync_check.format_doctor_report(result), str)


def test_check_journal_sync_fresh_foreign_is_live(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = 1000.0
    _write_foreign(journal, mtime=now - 5)

    result = sync_check.check_journal_sync(journal=journal, now=now)

    assert result.is_conflict
    assert result.primary_conflict is not None
    assert result.primary_conflict.hostname == "other-host"


def test_check_journal_sync_stale_foreign_is_history_only(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = 1000.0
    _write_foreign(journal, mtime=now - 120)

    result = sync_check.check_journal_sync(journal=journal, now=now)

    assert not result.is_conflict
    assert len(result.history_foreign_writers) == 1


def test_check_journal_sync_new_stale_file_since_snapshot_is_live(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    sync_dir = journal / "health" / "sync"
    sync_dir.mkdir(parents=True)
    now = 1000.0
    previous = sync_check.check_journal_sync(journal=journal, now=now)

    _write_foreign(journal, mtime=now - 120)
    result = sync_check.check_journal_sync(
        previous=previous.snapshot, journal=journal, now=now
    )

    assert result.is_conflict
    assert result.live_foreign_writers[0].appeared_since_snapshot


def test_check_journal_sync_changed_mtime_and_sha_since_snapshot_is_live(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = 1000.0
    path = _write_foreign(journal, mtime=now - 120)
    previous = sync_check.check_journal_sync(journal=journal, now=now)

    path.write_text(
        json.dumps(
            {
                "machine_id": "other-host-machine",
                "hostname": "other-host",
                "pid": 999,
                "journal_path": "/foreign/journal",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(path, (now - 119, now - 119))
    result = sync_check.check_journal_sync(
        previous=previous.snapshot, journal=journal, now=now
    )

    assert result.is_conflict
    assert result.live_foreign_writers[0].changed_since_snapshot


def test_check_journal_sync_missing_fields_tolerated_unknown_hostname(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = 1000.0
    _write_foreign(journal, payload={}, mtime=now - 5)

    result = sync_check.check_journal_sync(journal=journal, now=now)

    assert result.foreign_writers[0].display_hostname == "(unknown)"


def test_format_conflict_message_includes_resolution_guidance(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = time.time()
    _write_foreign(journal, mtime=now - 5)
    result = sync_check.check_journal_sync(journal=journal, now=now)

    message = sync_check.format_conflict_message(result)

    assert "one service per journal" in message
    assert "Multiple observers" in message
    assert "stop solstone on" in message
    assert "other-host" in message


def test_format_conflict_message_multiple_foreign_lists_bullets(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = time.time()
    _write_foreign(journal, "other-host", mtime=now - 5)
    _write_foreign(journal, "third-host", mtime=now - 10)
    result = sync_check.check_journal_sync(journal=journal, now=now)

    message = sync_check.format_conflict_message(result)

    assert "other-host" in message
    assert "third-host" in message
    assert "\n  • " in message


def test_format_conflict_message_future_mtime_shows_in_the_future(
    tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = time.time()
    _write_foreign(journal, mtime=now + 600)
    result = sync_check.check_journal_sync(journal=journal, now=now)

    message = sync_check.format_conflict_message(result)

    assert "in the future" in message


def test_format_doctor_report_clean_history_live(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    _set_identity(monkeypatch)
    now = time.time()

    clean = sync_check.check_journal_sync(journal=journal, now=now)
    assert sync_check.format_doctor_report(clean).startswith("this device only")

    _write_foreign(journal, mtime=now - 120)
    history = sync_check.check_journal_sync(journal=journal, now=now)
    history_report = sync_check.format_doctor_report(history)
    assert "last foreign writer" in history_report
    assert "other-host" in history_report

    _write_foreign(journal, "live-host", mtime=now - 5)
    live = sync_check.check_journal_sync(journal=journal, now=now)
    live_report = sync_check.format_doctor_report(live)
    assert "Refusing to start" in live_report
    assert "live-host" in live_report
