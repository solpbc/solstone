# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from think import install_guard


@pytest.fixture
def home_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def make_repo(tmp_path: Path, *, worktree: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    if worktree:
        (repo / ".git").write_text("gitdir: /tmp/worktree\n")
    else:
        (repo / ".git").mkdir()
    return repo


def ensure_expected_target(repo: Path) -> Path:
    target = install_guard.expected_target(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")
    return target


def make_alias(home_root: Path, target: Path | str) -> Path:
    alias = home_root / ".local" / "bin" / "sol"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)
    return alias


def other_target(tmp_path: Path) -> Path:
    target = tmp_path / "other" / ".venv" / "bin" / "sol"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")
    return target


def run_main(monkeypatch, capsys, repo: Path, *argv: str) -> tuple[int, str, str]:
    monkeypatch.chdir(repo)
    rc = install_guard.main(list(argv))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def alias_error(curdir: Path, installed: str) -> str:
    return (
        "ERROR: Another solstone install owns ~/.local/bin/sol.\n"
        f"  this repo:  {curdir}\n"
        f"{installed}\n"
        "Run 'make uninstall-service' from the installed repo first,\n"
        "or remove ~/.local/bin/sol manually if that repo is gone. No --force available.\n"
    )


def worktree_error(curdir: Path) -> str:
    return f"ERROR: refusing to run from a git worktree ({curdir}). Run from the primary clone.\n"


class TestCheckAlias:
    def test_absent(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.ABSENT
        assert other is None

    def test_owned(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        make_alias(home_root, target)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.OWNED
        assert other == target

    def test_cross_repo(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = other_target(tmp_path)
        make_alias(home_root, target)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.CROSS_REPO
        assert other == target

    def test_dangling(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = tmp_path / "missing" / ".venv" / "bin" / "sol"
        make_alias(home_root, target)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.DANGLING
        assert other == target

    def test_not_symlink(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a symlink")
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.NOT_SYMLINK
        assert other is None

    def test_worktree(self, home_root, tmp_path):
        repo = make_repo(tmp_path, worktree=True)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.WORKTREE
        assert other is None

    def test_worktree_takes_precedence(self, home_root, tmp_path):
        repo = make_repo(tmp_path, worktree=True)
        target = ensure_expected_target(repo)
        make_alias(home_root, target)
        state, other = install_guard.check_alias(repo)
        assert state is install_guard.AliasState.WORKTREE
        assert other is None


class TestErrorFormat:
    def test_worktree(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path, worktree=True).resolve()
        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == "worktree\n"
        assert captured.err == worktree_error(repo)

    def test_cross_repo(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        target = other_target(tmp_path).resolve()
        make_alias(home_root, target)
        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == "cross_repo\n"
        assert captured.err == alias_error(repo, f"  installed:  {target}")

    def test_dangling(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        target = (tmp_path / "missing" / ".venv" / "bin" / "sol").resolve()
        make_alias(home_root, target)
        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == "dangling\n"
        assert captured.err == alias_error(
            repo, f"  installed:  dangling: {target} does not exist"
        )

    def test_not_symlink(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a symlink")
        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == "not_symlink\n"
        assert captured.err == alias_error(repo, "  installed:  not a symlink")


class TestInstall:
    @pytest.fixture(autouse=True)
    def path_already_present(self, monkeypatch):
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            lambda _path: True,
        )

    def test_creates_symlink_on_absent(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        assert rc == 0
        assert out == "installed\npath: ~/.local/bin already on PATH\n"
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()

    def test_rewrites_owned_symlink(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        original = ensure_expected_target(repo)
        alias = make_alias(home_root, original)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 0
        assert out == "installed\npath: ~/.local/bin already on PATH\n"
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == original.resolve()

    def test_path_already_on_path_absent(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        append_mock = Mock(return_value=True)
        monkeypatch.setattr("think.install_guard.userpath.append", append_mock)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        assert rc == 0
        assert out.endswith("path: ~/.local/bin already on PATH\n")
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()
        append_mock.assert_not_called()

    def test_path_appended_restart_needed_absent(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        append_mock = Mock(return_value=True)
        restart_mock = Mock(return_value=True)
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            lambda _path: False,
        )
        monkeypatch.setattr("think.install_guard.userpath.append", append_mock)
        monkeypatch.setattr(
            "think.install_guard.userpath.need_shell_restart",
            restart_mock,
        )
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        assert rc == 0
        assert (
            out == "installed\n"
            "path: added ~/.local/bin to shell PATH — restart your shell or run 'exec $SHELL -l' to pick it up\n"
        )
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()
        append_mock.assert_called_once_with(
            str(alias.parent),
            app_name="solstone",
            all_shells=True,
        )
        restart_mock.assert_called_once_with(str(alias.parent))

    def test_path_appended_no_restart_owned(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        alias = make_alias(home_root, ensure_expected_target(repo))
        append_mock = Mock(return_value=True)
        restart_mock = Mock(return_value=False)
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            lambda _path: False,
        )
        monkeypatch.setattr("think.install_guard.userpath.append", append_mock)
        monkeypatch.setattr(
            "think.install_guard.userpath.need_shell_restart",
            restart_mock,
        )
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 0
        assert out == "installed\npath: added ~/.local/bin to shell PATH\n"
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()
        append_mock.assert_called_once_with(
            str(alias.parent),
            app_name="solstone",
            all_shells=True,
        )
        restart_mock.assert_called_once_with(str(alias.parent))

    def test_path_append_returns_false(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        append_mock = Mock(return_value=False)
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            lambda _path: False,
        )
        monkeypatch.setattr("think.install_guard.userpath.append", append_mock)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        assert rc == 0
        assert (
            out
            == 'installed\npath: could not auto-add ~/.local/bin to PATH — add this line to your shell rc manually: export PATH="$HOME/.local/bin:$PATH"\n'
        )
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()
        append_mock.assert_called_once_with(
            str(alias.parent),
            app_name="solstone",
            all_shells=True,
        )

    def test_path_unexpected_exception(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        append_mock = Mock(return_value=True)
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            Mock(side_effect=RuntimeError("boom")),
        )
        monkeypatch.setattr("think.install_guard.userpath.append", append_mock)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        assert rc == 0
        assert (
            out
            == 'installed\npath: could not auto-add ~/.local/bin to PATH (RuntimeError: boom) — add this line to your shell rc manually: export PATH="$HOME/.local/bin:$PATH"\n'
        )
        assert err == ""
        assert alias.is_symlink()
        assert alias.resolve() == install_guard.expected_target(repo).resolve()
        append_mock.assert_not_called()

    def test_refuses_cross_repo(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        target = other_target(tmp_path).resolve()
        alias = make_alias(home_root, target)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 1
        assert out == ""
        assert err == alias_error(repo, f"  installed:  {target}")
        assert alias.is_symlink()
        assert alias.resolve() == target

    def test_refuses_dangling(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        target = (tmp_path / "missing" / ".venv" / "bin" / "sol").resolve()
        alias = make_alias(home_root, target)
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 1
        assert out == ""
        assert err == alias_error(
            repo, f"  installed:  dangling: {target} does not exist"
        )
        assert alias.is_symlink()
        assert Path(os.readlink(alias)).name == "sol"

    def test_refuses_not_symlink(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a symlink")
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 1
        assert out == ""
        assert err == alias_error(repo, "  installed:  not a symlink")
        assert alias.read_text() == "not a symlink"

    def test_refuses_worktree(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path, worktree=True).resolve()
        rc, out, err = run_main(monkeypatch, capsys, repo, "install")
        assert rc == 1
        assert out == ""
        assert err == worktree_error(repo)
        assert not install_guard.alias_path().exists()


class TestUninstall:
    def test_removes_owned_alias(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        alias = make_alias(home_root, target)
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 0
        assert out == "removed\n"
        assert err == ""
        assert not alias.exists()
        assert not alias.is_symlink()

    def test_noop_on_absent(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 0
        assert out == "absent\n"
        assert err == ""
        assert not install_guard.alias_path().exists()

    def test_refuses_cross_repo(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        target = other_target(tmp_path).resolve()
        alias = make_alias(home_root, target)
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 1
        assert out == ""
        assert err == alias_error(repo, f"  installed:  {target}")
        assert alias.is_symlink()
        assert alias.resolve() == target

    def test_refuses_dangling(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        target = (tmp_path / "missing" / ".venv" / "bin" / "sol").resolve()
        alias = make_alias(home_root, target)
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 1
        assert out == ""
        assert err == alias_error(
            repo, f"  installed:  dangling: {target} does not exist"
        )
        assert alias.is_symlink()

    def test_refuses_not_symlink(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a symlink")
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 1
        assert out == ""
        assert err == alias_error(repo, "  installed:  not a symlink")
        assert alias.read_text() == "not a symlink"

    def test_refuses_worktree(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path, worktree=True).resolve()
        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")
        assert rc == 1
        assert out == ""
        assert err == worktree_error(repo)
        assert not install_guard.alias_path().exists()
