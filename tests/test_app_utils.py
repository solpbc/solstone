# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import pytest

from apps.utils import get_app_storage_path
from convey import state


def test_get_app_storage_path_uses_state_journal_root(tmp_path, monkeypatch):
    monkeypatch.setattr("convey.state.journal_root", str(tmp_path))
    assert state.journal_root == str(tmp_path)

    result = get_app_storage_path("sampleapp", ensure_exists=False)

    assert result == tmp_path / "apps" / "sampleapp"
    assert result.is_absolute()


def test_get_app_storage_path_falls_back_to_get_journal_when_state_empty(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("convey.state.journal_root", "")
    other_dir = tmp_path / "other"
    other_dir.mkdir(parents=True)
    monkeypatch.chdir(other_dir)
    fake_journal = tmp_path / "journal"
    fake_journal.mkdir(parents=True)
    monkeypatch.setattr("apps.utils.get_journal", lambda: str(fake_journal))

    result = get_app_storage_path("sampleapp", ensure_exists=False)

    assert result.is_absolute()
    assert result == fake_journal / "apps" / "sampleapp"
    assert Path.cwd() not in result.parents
    assert result != Path.cwd() / "apps" / "sampleapp"


def test_get_app_storage_path_raises_on_non_absolute_root(tmp_path, monkeypatch):
    monkeypatch.setattr("convey.state.journal_root", "apps")

    with pytest.raises(RuntimeError) as excinfo:
        get_app_storage_path("sampleapp", ensure_exists=False)

    assert (
        str(excinfo.value)
        == "get_app_storage_path: resolved journal root is not absolute: apps"
    )


def test_get_app_storage_path_rejects_invalid_app_name(tmp_path, monkeypatch):
    monkeypatch.setattr("convey.state.journal_root", str(tmp_path))

    with pytest.raises(ValueError):
        get_app_storage_path("Bad-Name")
