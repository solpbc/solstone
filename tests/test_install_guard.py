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
        (repo / ".git").write_text("gitdir: /tmp/worktree\n", encoding="utf-8")
    else:
        (repo / ".git").mkdir()
    return repo


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


def make_managed_wrapper(
    home_root: Path,
    *,
    journal: str,
    sol_bin: str,
    mode: int = 0o755,
) -> Path:
    alias = home_root / ".local" / "bin" / "sol"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.write_text(
        install_guard.render_wrapper(journal, sol_bin),
        encoding="utf-8",
    )
    alias.chmod(mode)
    return alias


def other_target(tmp_path: Path) -> Path:
    target = tmp_path / "other" / ".venv" / "bin" / "sol"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def run_main(monkeypatch, capsys, repo: Path, *argv: str) -> tuple[int, str, str]:
    monkeypatch.chdir(repo)
    rc = install_guard.main(list(argv))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def alias_error(curdir: Path, installed: str, *, allow_force: bool = False) -> str:
    message = (
        "ERROR: Another solstone install owns ~/.local/bin/sol.\n"
        f"  this repo:  {curdir}\n"
        f"{installed}\n"
        "Run 'make uninstall-service' from the installed repo first,\n"
        "or remove ~/.local/bin/sol manually if that repo is gone.\n"
    )
    if allow_force:
        message += "Rerun 'python -m think.install_guard install --force' only if you intend to replace it from this repo.\n"
    return message


def worktree_error(curdir: Path) -> str:
    return f"ERROR: refusing to run from a git worktree ({curdir}). Run from the primary clone.\n"


class TestWrapperHelpers:
    def test_render_wrapper_round_trip_simple(self):
        journal = "/tmp/solstone"
        sol_bin = "/tmp/repo/.venv/bin/sol"

        content = install_guard.render_wrapper(journal, sol_bin)

        assert install_guard.parse_wrapper(content) == {
            "journal": journal,
            "sol_bin": sol_bin,
        }

    def test_render_wrapper_round_trip_tricky_paths(self):
        journal = "/tmp/solstone notes/über"
        sol_bin = "/tmp/it's a test/über/.venv/bin/sol"

        content = install_guard.render_wrapper(journal, sol_bin)

        assert install_guard.parse_wrapper(content) == {
            "journal": journal,
            "sol_bin": sol_bin,
        }

    def test_render_wrapper_matches_spec_template(self):
        journal = "/Users/jer/Documents/Solstone"
        sol_bin = "/Users/jer/projects/solstone/.venv/bin/sol"

        content = install_guard.render_wrapper(journal, sol_bin)

        assert (
            content == "#!/bin/sh\n"
            "# sol — managed by 'sol config'. Edits will be overwritten.\n"
            "# managed-version: 1\n"
            ': "${SOLSTONE_JOURNAL:=/Users/jer/Documents/Solstone}"\n'
            "export SOLSTONE_JOURNAL\n"
            "SOL_BIN='/Users/jer/projects/solstone/.venv/bin/sol'\n"
            'if [ ! -x "$SOL_BIN" ]; then\n'
            "    printf 'sol: venv binary missing or not executable: %s\\n' \"$SOL_BIN\" >&2\n"
            "    exit 127\n"
            "fi\n"
            'exec "$SOL_BIN" "$@"\n'
        )

    @pytest.mark.parametrize("char", ["$", "`", '"', "\\"])
    def test_validate_journal_path_for_wrapper_rejects_invalid_chars(self, char: str):
        with pytest.raises(ValueError, match="shell-active character"):
            install_guard.validate_journal_path_for_wrapper(f"/tmp/bad{char}path")


class TestCheckAlias:
    def test_absent(self, home_root, tmp_path):
        repo = make_repo(tmp_path)

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.ABSENT
        assert other is None

    def test_owned_legacy_symlink(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        make_alias(home_root, target)

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.OWNED
        assert other == target.resolve()

    def test_owned_managed_wrapper(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        make_managed_wrapper(
            home_root,
            journal="/tmp/solstone",
            sol_bin=str(target),
        )

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.OWNED
        assert other == target.resolve()

    def test_cross_repo(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = other_target(tmp_path)
        make_alias(home_root, target)

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.CROSS_REPO
        assert other == target.resolve()

    def test_dangling(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        target = tmp_path / "missing" / ".venv" / "bin" / "sol"
        make_alias(home_root, target)

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.DANGLING
        assert other == target.resolve()

    def test_foreign_regular_file(self, home_root, tmp_path):
        repo = make_repo(tmp_path)
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a wrapper", encoding="utf-8")

        state, other = install_guard.check_alias(repo)

        assert state is install_guard.AliasState.FOREIGN
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


class TestCheckCommand:
    def test_worktree(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path, worktree=True).resolve()

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 1
        assert captured.out == "worktree\n"
        assert captured.err == worktree_error(repo)

    def test_absent(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 0
        assert captured.out == "fresh\n"
        assert captured.err == ""

    def test_check_reports_current_for_managed_wrapper_with_matching_paths(
        self, home_root, tmp_path, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        make_managed_wrapper(
            home_root,
            journal=install_guard._current_journal_for_alias(),
            sol_bin=str(target),
        )

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 0
        assert captured.out == "current\n"
        assert captured.err == ""

    def test_check_reports_upgrade_for_legacy_symlink(
        self, home_root, tmp_path, capsys
    ):
        repo = make_repo(tmp_path)
        make_alias(home_root, ensure_expected_target(repo))

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 0
        assert captured.out == "upgrade\n"
        assert captured.err == ""

    def test_check_reports_upgrade_for_wrapper_with_stale_paths(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        old_journal = str((tmp_path / "old-journal").resolve())
        new_journal = str((tmp_path / "new-journal").resolve())
        make_managed_wrapper(home_root, journal=old_journal, sol_bin=str(target))
        monkeypatch.setenv("SOLSTONE_JOURNAL", new_journal)

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 0
        assert captured.out == "upgrade\n"
        assert captured.err == ""

    def test_cross_repo(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        target = other_target(tmp_path).resolve()
        make_alias(home_root, target)

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 1
        assert captured.out == "cross_repo\n"
        assert captured.err == alias_error(
            repo,
            f"  installed:  {target}",
            allow_force=True,
        )

    def test_dangling(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        target = (tmp_path / "missing" / ".venv" / "bin" / "sol").resolve()
        make_alias(home_root, target)

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 1
        assert captured.out == "dangling\n"
        assert captured.err == alias_error(
            repo,
            f"  installed:  dangling: {target} does not exist",
            allow_force=True,
        )

    def test_foreign(self, home_root, tmp_path, capsys):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a wrapper", encoding="utf-8")

        rc = install_guard.cmd_check(repo)
        captured = capsys.readouterr()

        assert rc == 1
        assert captured.out == "not_symlink\n"
        assert captured.err == alias_error(
            repo,
            "  installed:  not a symlink",
            allow_force=True,
        )


class TestInstall:
    @pytest.fixture(autouse=True)
    def path_already_present(self, monkeypatch):
        monkeypatch.setattr(
            "think.install_guard.userpath.in_current_path",
            lambda _path: True,
        )

    def test_install_upgrades_legacy_symlink_to_managed_wrapper(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        alias = make_alias(home_root, target)

        rc, out, err = run_main(monkeypatch, capsys, repo, "install")

        assert rc == 0
        assert out == "installed\npath: ~/.local/bin already on PATH\n"
        assert err == ""
        assert not alias.is_symlink()
        assert os.access(alias, os.X_OK)
        assert alias.read_text(encoding="utf-8") == install_guard.render_wrapper(
            install_guard._current_journal_for_alias(),
            str(target),
        )
        assert install_guard.parse_wrapper(alias.read_text(encoding="utf-8")) == {
            "journal": install_guard._current_journal_for_alias(),
            "sol_bin": str(target),
        }

    def test_install_refuses_foreign_regular_file_without_force(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("foreign", encoding="utf-8")

        rc, out, err = run_main(monkeypatch, capsys, repo, "install")

        assert rc == 1
        assert out == ""
        assert err == alias_error(
            repo,
            "  installed:  not a symlink",
            allow_force=True,
        )
        assert alias.read_text(encoding="utf-8") == "foreign"

    def test_install_force_overwrites_foreign_regular_file(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("foreign", encoding="utf-8")

        rc, out, err = run_main(monkeypatch, capsys, repo, "install", "--force")

        assert rc == 0
        assert out == "installed\npath: ~/.local/bin already on PATH\n"
        assert err == ""
        assert install_guard.parse_wrapper(alias.read_text(encoding="utf-8")) == {
            "journal": install_guard._current_journal_for_alias(),
            "sol_bin": str(target),
        }
        assert os.access(alias, os.X_OK)

    def test_install_is_idempotent(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)

        rc1, out1, err1 = run_main(monkeypatch, capsys, repo, "install")
        alias = install_guard.alias_path()
        first_content = alias.read_text(encoding="utf-8")

        rc2, out2, err2 = run_main(monkeypatch, capsys, repo, "install")

        assert rc1 == 0
        assert out1 == "installed\npath: ~/.local/bin already on PATH\n"
        assert err1 == ""
        assert rc2 == 0
        assert out2 == "installed\npath: ~/.local/bin already on PATH\n"
        assert err2 == ""
        assert alias.read_text(encoding="utf-8") == first_content
        assert install_guard.parse_wrapper(first_content) == {
            "journal": install_guard._current_journal_for_alias(),
            "sol_bin": str(target),
        }

    def test_install_refuses_invalid_journal_path(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        monkeypatch.setenv("SOLSTONE_JOURNAL", "/tmp/bad$path")

        rc, out, err = run_main(monkeypatch, capsys, repo, "install")

        assert rc == 1
        assert out == ""
        assert "refused: journal path contains shell-active character '$'" in err
        assert not install_guard.alias_path().exists()

    def test_path_appended_when_not_on_path(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
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
        alias = install_guard.alias_path()

        assert rc == 0
        assert out == "installed\npath: added ~/.local/bin to shell PATH\n"
        assert err == ""
        assert install_guard.parse_wrapper(alias.read_text(encoding="utf-8")) == {
            "journal": install_guard._current_journal_for_alias(),
            "sol_bin": str(target),
        }
        append_mock.assert_called_once_with(
            str(alias.parent),
            app_name="solstone",
            all_shells=True,
        )
        restart_mock.assert_called_once_with(str(alias.parent))


class TestUninstall:
    def test_uninstall_removes_managed_wrapper(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        alias = make_managed_wrapper(
            home_root,
            journal=install_guard._current_journal_for_alias(),
            sol_bin=str(target),
        )

        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")

        assert rc == 0
        assert out == "uninstalled\n"
        assert err == ""
        assert not alias.exists()
        assert not alias.is_symlink()

    def test_uninstall_removes_legacy_symlink(
        self, home_root, tmp_path, monkeypatch, capsys
    ):
        repo = make_repo(tmp_path)
        target = ensure_expected_target(repo)
        alias = make_alias(home_root, target)

        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")

        assert rc == 0
        assert out == "uninstalled\n"
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

    def test_refuses_foreign(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path).resolve()
        alias = install_guard.alias_path()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("foreign", encoding="utf-8")

        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")

        assert rc == 1
        assert out == ""
        assert err == alias_error(repo, "  installed:  not a symlink")
        assert alias.read_text(encoding="utf-8") == "foreign"

    def test_refuses_worktree(self, home_root, tmp_path, monkeypatch, capsys):
        repo = make_repo(tmp_path, worktree=True).resolve()

        rc, out, err = run_main(monkeypatch, capsys, repo, "uninstall")

        assert rc == 1
        assert out == ""
        assert err == worktree_error(repo)
        assert not install_guard.alias_path().exists()
