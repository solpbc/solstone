# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest


def test_load_sync_state_missing(tmp_path):
    """Returns None when no state file exists."""
    from think.importers.sync import load_sync_state

    assert load_sync_state(tmp_path, "plaud") is None


def test_save_and_load_sync_state(tmp_path):
    """Round-trip: save then load."""
    from think.importers.sync import load_sync_state, save_sync_state

    state = {
        "backend": "plaud",
        "last_sync": "2026-02-14T12:00:00",
        "synced_files": {
            "abc123": {
                "filename": "test.opus",
                "synced_at": "2026-02-14T12:00:00",
            }
        },
    }
    save_sync_state(tmp_path, "plaud", state)
    loaded = load_sync_state(tmp_path, "plaud")
    assert loaded == state


def test_save_sync_state_creates_imports_dir(tmp_path):
    """imports/ dir is created if missing."""
    from think.importers.sync import save_sync_state

    save_sync_state(tmp_path, "plaud", {"backend": "plaud"})
    assert (tmp_path / "imports" / "plaud.json").exists()


def test_save_sync_state_no_temp_files(tmp_path):
    """No .tmp files left after save."""
    from think.importers.sync import save_sync_state

    save_sync_state(tmp_path, "plaud", {"backend": "plaud"})
    imports = tmp_path / "imports"
    temps = list(imports.glob("*.tmp"))
    assert temps == []


def test_load_sync_state_corrupt_json(tmp_path):
    """Returns None on corrupt JSON."""
    from think.importers.sync import load_sync_state

    state_path = tmp_path / "imports" / "plaud.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not json{{{", encoding="utf-8")
    assert load_sync_state(tmp_path, "plaud") is None


def test_get_syncable_backends():
    """Discovers plaud backend."""
    from think.importers.sync import get_syncable_backends

    backends = get_syncable_backends()
    names = [b.name for b in backends]
    assert "plaud" in names


def test_plaud_protocol_conformance():
    """PlaudBackend satisfies SyncableBackend protocol."""
    from think.importers.plaud import PlaudBackend
    from think.importers.sync import SyncableBackend

    assert isinstance(PlaudBackend(), SyncableBackend)


def test_plaud_sync_not_implemented(tmp_path):
    """PlaudBackend.sync() raises NotImplementedError."""
    from think.importers.plaud import PlaudBackend

    with pytest.raises(NotImplementedError):
        PlaudBackend().sync(tmp_path)


def test_backends_cli_flag(capsys, monkeypatch):
    """sol import --backends lists plaud."""
    import sys

    from think.importers.cli import main

    monkeypatch.setattr(sys, "argv", ["sol import", "--backends"])
    monkeypatch.setenv("JOURNAL_PATH", "/tmp/test-journal")
    main()
    captured = capsys.readouterr()
    assert "plaud" in captured.out
