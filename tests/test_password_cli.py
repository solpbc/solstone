# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol password CLI (think/password_cli.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from werkzeug.security import check_password_hash

from think.password_cli import main


@pytest.fixture
def journal_dir(tmp_path, monkeypatch):
    """Copy test fixture to temp dir for mutation tests."""
    src = Path(__file__).resolve().parent / "fixtures" / "journal"
    dst = tmp_path / "journal"
    shutil.copytree(src, dst, symlinks=True)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst))
    return dst


def _read_config(journal_dir):
    return json.loads((journal_dir / "config" / "journal.json").read_text())


def test_set_writes_hash(journal_dir, monkeypatch, capsys):
    """sol password set writes a password_hash that verifies correctly."""
    monkeypatch.setattr("sys.argv", ["sol password", "set"])
    monkeypatch.setattr("think.password_cli.getpass.getpass", lambda prompt="": "secret123")

    main()

    config = _read_config(journal_dir)
    assert "password_hash" in config["convey"]
    assert check_password_hash(config["convey"]["password_hash"], "secret123")
    assert "Password set successfully." in capsys.readouterr().out


def test_mismatch_rejected(journal_dir, monkeypatch):
    """Mismatched passwords exit with code 1 and don't change config."""
    config_before = _read_config(journal_dir)
    monkeypatch.setattr("sys.argv", ["sol password", "set"])
    call_count = 0

    def fake_getpass(prompt=""):
        nonlocal call_count
        call_count += 1
        return "first" if call_count == 1 else "second"

    monkeypatch.setattr("think.password_cli.getpass.getpass", fake_getpass)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert _read_config(journal_dir) == config_before


def test_old_plaintext_cleaned(journal_dir, monkeypatch):
    """Running set removes convey.password and writes password_hash."""
    config_path = journal_dir / "config" / "journal.json"
    config = json.loads(config_path.read_text())
    config.setdefault("convey", {})["password"] = "old-plaintext"
    config_path.write_text(json.dumps(config, indent=2))

    monkeypatch.setattr("sys.argv", ["sol password", "set"])
    monkeypatch.setattr("think.password_cli.getpass.getpass", lambda prompt="": "new-pass")

    main()

    config = _read_config(journal_dir)
    assert "password" not in config["convey"]
    assert "password_hash" in config["convey"]
    assert check_password_hash(config["convey"]["password_hash"], "new-pass")


def test_creates_config_from_scratch(tmp_path, monkeypatch):
    """Works when no journal.json exists - creates the config dir and file."""
    journal = tmp_path / "empty-journal"
    journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    monkeypatch.setattr("sys.argv", ["sol password", "set"])
    monkeypatch.setattr("think.password_cli.getpass.getpass", lambda prompt="": "fresh")

    main()

    config_path = journal / "config" / "journal.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert check_password_hash(config["convey"]["password_hash"], "fresh")
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_reset_alias(journal_dir, monkeypatch, capsys):
    """sol password reset behaves identically to sol password set."""
    monkeypatch.setattr("sys.argv", ["sol password", "reset"])
    monkeypatch.setattr("think.password_cli.getpass.getpass", lambda prompt="": "reset-pw")

    main()

    config = _read_config(journal_dir)
    assert check_password_hash(config["convey"]["password_hash"], "reset-pw")
    assert "Password set successfully." in capsys.readouterr().out
