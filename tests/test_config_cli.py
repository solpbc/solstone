# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from think import config_cli, install_guard


@pytest.fixture
def home_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def ensure_expected_target(repo: Path) -> Path:
    target = install_guard.expected_target(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def make_alias(home_root: Path, target: Path | str) -> Path:
    alias = home_root / ".local" / "bin" / "sol"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)
    return alias


def make_managed_wrapper(home_root: Path, *, journal: str, sol_bin: str) -> Path:
    alias = home_root / ".local" / "bin" / "sol"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.write_text(
        install_guard.render_wrapper(journal, sol_bin),
        encoding="utf-8",
    )
    alias.chmod(0o755)
    return alias


def test_config_command_registered():
    from think import sol_cli as sol

    assert sol.COMMANDS["config"] == "think.config_cli"
    assert "config" in sol.GROUPS["Specialized tools"]


def test_show_reports_wrapper_embedded(home_root, monkeypatch, tmp_path, capsys):
    journal = str((tmp_path / "journal").resolve())
    target = ensure_expected_target(tmp_path / "repo")
    make_managed_wrapper(home_root, journal=journal, sol_bin=str(target))
    monkeypatch.setenv("SOLSTONE_JOURNAL", journal)

    rc = config_cli.cmd_show()
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"path: {journal}",
        "source: wrapper-embedded",
        "wrapper-status: managed",
    ]


def test_show_reports_caller_override(home_root, monkeypatch, tmp_path, capsys):
    embedded = str((tmp_path / "embedded").resolve())
    override = str((tmp_path / "override").resolve())
    target = ensure_expected_target(tmp_path / "repo")
    make_managed_wrapper(home_root, journal=embedded, sol_bin=str(target))
    monkeypatch.setenv("SOLSTONE_JOURNAL", override)

    rc = config_cli.cmd_show()
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"path: {override}",
        "source: caller-override",
        "wrapper-status: managed",
    ]


def test_show_reports_source_tree_fallback(home_root, monkeypatch, capsys):
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)

    rc = config_cli.cmd_show()
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"path: {Path(config_cli.get_project_root()) / 'journal'}",
        "source: source-tree fallback",
        "wrapper-status: absent",
    ]


def test_journal_noops_when_path_already_embedded(
    home_root, monkeypatch, tmp_path, capsys
):
    target_path = str((tmp_path / "journal").resolve())
    target = ensure_expected_target(tmp_path / "repo")
    alias = make_managed_wrapper(home_root, journal=target_path, sol_bin=str(target))
    original = alias.read_text(encoding="utf-8")

    rc = config_cli.cmd_journal(target_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out == f"sol config: journal already set to {target_path}\n"
    assert alias.read_text(encoding="utf-8") == original


def test_journal_rewrites_wrapper(home_root, monkeypatch, tmp_path, capsys):
    source_path = str((tmp_path / "source").resolve())
    target_path = str((tmp_path / "target").resolve())
    target = ensure_expected_target(tmp_path / "repo")
    alias = make_managed_wrapper(home_root, journal=source_path, sol_bin=str(target))
    run_mock = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(config_cli.subprocess, "run", run_mock)
    monkeypatch.chdir(home_root)

    rc = config_cli.cmd_journal(target_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out == f"sol config: journal set to {target_path}\n"
    assert install_guard.parse_wrapper(alias.read_text(encoding="utf-8")) == {
        "journal": target_path,
        "sol_bin": str(target),
    }
    run_mock.assert_called_once_with(
        [str(target), "service", "restart", "--if-installed"],
        check=False,
    )


def test_journal_refuses_without_managed_wrapper(home_root, tmp_path, capsys):
    target_path = str((tmp_path / "journal").resolve())

    rc = config_cli.cmd_journal(target_path)
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "make install-service" in captured.err


def test_journal_refuses_legacy_symlink(home_root, tmp_path, capsys):
    target_path = str((tmp_path / "journal").resolve())
    make_alias(home_root, "/tmp/elsewhere/.venv/bin/sol")

    rc = config_cli.cmd_journal(target_path)
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "make install-service" in captured.err


def test_journal_refuses_invalid_chars(home_root, capsys):
    rc = config_cli.cmd_journal("/tmp/bad$path")
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "shell-active character '$'" in captured.err


def test_journal_refuses_source_tree_path_outside_source_checkout(
    home_root, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(config_cli, "get_project_root", lambda: str(tmp_path))

    rc = config_cli.cmd_journal(str((tmp_path / "journal").resolve()))
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "source-tree fallback path" in captured.err


def test_journal_exits_2_on_restart_failure(home_root, monkeypatch, tmp_path, capsys):
    source_path = str((tmp_path / "source").resolve())
    target_path = str((tmp_path / "target").resolve())
    target = ensure_expected_target(tmp_path / "repo")
    alias = make_managed_wrapper(home_root, journal=source_path, sol_bin=str(target))
    monkeypatch.setattr(
        config_cli.subprocess,
        "run",
        MagicMock(return_value=MagicMock(returncode=1)),
    )
    monkeypatch.chdir(home_root)

    rc = config_cli.cmd_journal(target_path)
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "wrapper rewritten to" in captured.err
    assert install_guard.parse_wrapper(alias.read_text(encoding="utf-8")) == {
        "journal": target_path,
        "sol_bin": str(target),
    }
