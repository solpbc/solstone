# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from solstone.think import skills_cli
from solstone.think.skills_cli import (
    install_project,
    install_user,
    list_project_status,
    list_user_status,
    resolve_user_skill,
    uninstall_user,
)


def _write_skill(path: Path, content: bytes | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_bytes(content or b"---\nname: test\n---\n")


def _mini_user_repo(tmp_path: Path, content: bytes | None = None) -> Path:
    skill_dir = tmp_path / "solstone"
    _write_skill(skill_dir, content)
    return skill_dir


def _mini_project_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write_skill(repo / "solstone" / "talent" / "journal")
    _write_skill(repo / "solstone" / "talent" / "routines")
    _write_skill(repo / "solstone" / "apps" / "foo" / "talent" / "bar")
    return repo


def _home(tmp_path: Path, *parents: str) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    for parent in parents:
        (home / parent).mkdir()
    return home


def test_install_user_creates_targets_for_present_agents(tmp_path):
    repo = _mini_user_repo(tmp_path, b"solstone bytes")
    home = _home(tmp_path, ".claude", ".codex")

    report = install_user(repo, home, ["all"])

    assert report.error_count == 0
    source = repo / "SKILL.md"
    assert (
        home / ".claude" / "skills" / "solstone" / "SKILL.md"
    ).read_bytes() == source.read_bytes()
    assert (
        home / ".codex" / "skills" / "solstone" / "SKILL.md"
    ).read_bytes() == source.read_bytes()
    assert {path.name for path in (home / ".claude" / "skills").iterdir()} == {
        "solstone"
    }
    assert {path.name for path in (home / ".codex" / "skills").iterdir()} == {
        "solstone"
    }


def test_install_user_creates_missing_codex_parent_dir(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path)

    report = install_user(repo, home, ["codex"])

    assert report.error_count == 0
    assert (home / ".codex" / "skills" / "solstone" / "SKILL.md").exists()
    assert report.rows == [
        skills_cli.ActionRow(
            "codex",
            "solstone",
            "installed",
            home / ".codex" / "skills" / "solstone",
        )
    ]


def test_install_user_creates_missing_gemini_parent_dir(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path)

    report = install_user(repo, home, ["gemini"])

    assert report.error_count == 0
    assert (home / ".gemini" / "skills" / "solstone" / "SKILL.md").exists()
    assert report.rows == [
        skills_cli.ActionRow(
            "gemini",
            "solstone",
            "installed",
            home / ".gemini" / "skills" / "solstone",
        )
    ]


def test_install_user_creates_all_three_when_none_exist(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path)

    report = install_user(repo, home, ["all"])

    assert report.error_count == 0
    for agent in [".claude", ".codex", ".gemini"]:
        assert (home / agent / "skills" / "solstone" / "SKILL.md").exists()
    assert [row.action for row in report.rows] == [
        "installed",
        "installed",
        "installed",
    ]


def test_install_user_replaces_modified_source(tmp_path):
    repo = _mini_user_repo(tmp_path, b"first")
    home = _home(tmp_path, ".claude")
    install_user(repo, home, ["claude"])
    (repo / "SKILL.md").write_bytes(b"second")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert (
        home / ".claude" / "skills" / "solstone" / "SKILL.md"
    ).read_bytes() == b"second"


def test_install_user_replaces_existing_regular_file_target(tmp_path):
    repo = _mini_user_repo(tmp_path, b"fresh")
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_bytes(b"stale")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert (target / "SKILL.md").read_bytes() == b"fresh"


def test_install_user_replaces_stray_symlink_target(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "whatever")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert target.is_dir()
    assert (target / "SKILL.md").exists()
    assert report.rows[0].action == "replaced"


def test_install_user_replaces_stray_regular_file_target(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.parent.mkdir(parents=True)
    target.write_text("not a dir", encoding="utf-8")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert target.is_dir()
    assert (target / "SKILL.md").exists()
    assert report.rows[0].action == "replaced"


def test_install_user_permission_error_prints_clean_message(tmp_path, capsys):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.mkdir(parents=True)
    target.chmod(0o500)
    try:
        report = install_user(repo, home, ["claude"])
        skills_cli._print_report(report, "install")
    finally:
        target.chmod(0o700)

    captured = capsys.readouterr()
    assert report.error_count == 1
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_uninstall_user_removes_only_bundle_dirs(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    solstone = home / ".claude" / "skills" / "solstone"
    hop = home / ".claude" / "skills" / "hop"
    _write_skill(solstone)
    _write_skill(hop)

    report = uninstall_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert not solstone.exists()
    assert hop.exists()


def test_uninstall_user_absent_target_is_no_op(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")

    report = uninstall_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert report.rows[0].action == "skipped"
    assert report.rows[0].reason == "nothing to remove"


def test_install_user_agent_filter(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude", ".codex")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 0
    assert (home / ".claude" / "skills" / "solstone").exists()
    assert not (home / ".codex" / "skills" / "solstone").exists()


def test_install_project_creates_symlinks(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"

    report = install_project(repo, target, ["all"])

    assert report.error_count == 0
    for agent_dir in [".claude", ".agents"]:
        link_parent = target / agent_dir / "skills"
        for name in ["journal", "routines", "bar"]:
            link = link_parent / name
            assert link.is_symlink()
            assert os.readlink(link) == os.path.relpath(
                repo
                / "solstone"
                / ("talent" if name != "bar" else "apps/foo/talent")
                / name,
                link_parent,
            )


def test_install_project_idempotent(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    install_project(repo, target, ["all"])
    before = {
        path: (os.readlink(path), path.lstat().st_mtime_ns)
        for path in sorted((target / ".claude" / "skills").iterdir())
    }

    report = install_project(repo, target, ["all"])

    after = {
        path: (os.readlink(path), path.lstat().st_mtime_ns)
        for path in sorted((target / ".claude" / "skills").iterdir())
    }
    assert report.error_count == 0
    assert all(row.action == "noop" for row in report.rows)
    assert before == after


def test_install_project_cleans_stale_symlinks(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    install_project(repo, target, ["all"])
    shutil.rmtree(repo / "solstone" / "talent" / "routines")

    report = install_project(repo, target, ["all"])

    assert report.error_count == 0
    assert not (target / ".claude" / "skills" / "routines").exists()
    assert any(row.action == "removed" and row.reason == "stale" for row in report.rows)


def test_install_project_dedupe_error(tmp_path):
    repo = tmp_path / "repo"
    _write_skill(repo / "solstone" / "talent" / "foo")
    _write_skill(repo / "solstone" / "apps" / "x" / "talent" / "foo")

    with pytest.raises(ValueError) as exc_info:
        install_project(repo, tmp_path / "work", ["all"])

    message = str(exc_info.value)
    assert str(repo / "solstone" / "talent" / "foo") in message
    assert str(repo / "solstone" / "apps" / "x" / "talent" / "foo") in message


def test_install_project_agent_claude_only(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"

    report = install_project(repo, target, ["claude"])

    assert report.error_count == 0
    assert (target / ".claude" / "skills" / "journal").is_symlink()
    assert not (target / ".agents").exists()


def test_install_project_rejects_codex_or_gemini(tmp_path):
    repo = _mini_project_repo(tmp_path)

    with pytest.raises(ValueError, match="--agent codex is not supported"):
        install_project(repo, tmp_path / "work", ["codex"])
    with pytest.raises(ValueError, match="--agent gemini is not supported"):
        install_project(repo, tmp_path / "work", ["gemini"])


def test_install_project_relative_target_outside_repo(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "outside" / "work"

    install_project(repo, target, ["all"])

    link_parent = target / ".claude" / "skills"
    link = link_parent / "journal"
    assert os.readlink(link) == os.path.relpath(
        repo / "solstone" / "talent" / "journal", link_parent
    )


def test_install_project_emits_warning_for_user_content_at_target(tmp_path, capsys):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    link = target / ".claude" / "skills" / "journal"
    link.parent.mkdir(parents=True)
    link.write_bytes(b"user-content")

    report = install_project(repo, target, ["all"])

    assert link.read_bytes() == b"user-content"
    assert report.error_count == 0
    assert report.warning_count == 1
    warning = next(row for row in report.rows if row.action == "warning")
    assert warning.agent == "claude"
    assert warning.skill == "journal"
    assert warning.path == link
    assert warning.reason == "user content at target preserved"
    assert (
        skills_cli._run_report("install", lambda *_args: report, repo, target, ["all"])
        == 0
    )
    assert "Warnings:" in capsys.readouterr().out


def test_install_project_emits_warning_for_user_directory_at_target(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    link = target / ".claude" / "skills" / "journal"
    link.mkdir(parents=True)
    (link / "SKILL.md").write_bytes(b"user-content")

    report = install_project(repo, target, ["all"])

    assert (link / "SKILL.md").read_bytes() == b"user-content"
    assert report.error_count == 0
    assert report.warning_count == 1
    warning = next(row for row in report.rows if row.action == "warning")
    assert warning.agent == "claude"
    assert warning.skill == "journal"
    assert warning.path == link
    assert warning.reason == "user content at target preserved"


def test_list_status_reports_installed_and_not_installed(tmp_path):
    user_repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude", ".codex")
    install_user(user_repo, home, ["claude"])

    rows = list_user_status(user_repo, home, ["all"])

    assert ("claude", "solstone", "installed") in {
        (row.agent, row.skill, row.state) for row in rows
    }
    assert ("codex", "solstone", "not installed") in {
        (row.agent, row.skill, row.state) for row in rows
    }


def test_list_project_status_reports_correct_symlink_only(tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    install_project(repo, target, ["claude"])

    rows = list_project_status(repo, target, ["all"])

    assert ("claude", "journal", "installed") in {
        (row.agent, row.skill, row.state) for row in rows
    }
    assert ("agents", "journal", "not installed") in {
        (row.agent, row.skill, row.state) for row in rows
    }


def test_main_install_user_default(monkeypatch, tmp_path, capsys):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(skills_cli, "get_project_root", lambda: str(repo))
    monkeypatch.setattr(skills_cli, "resolve_user_skill", lambda: repo)
    monkeypatch.setattr(sys, "argv", ["sol skills", "install"])

    exit_code = skills_cli.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "installed claude solstone" in captured.out
    assert (home / ".claude" / "skills" / "solstone" / "SKILL.md").exists()


def test_main_install_project_no_dir_uses_cwd(monkeypatch, tmp_path):
    repo = _mini_project_repo(tmp_path)
    target = tmp_path / "work"
    target.mkdir()
    monkeypatch.chdir(target)
    monkeypatch.setattr(skills_cli, "get_project_root", lambda: str(repo))
    monkeypatch.setattr(sys, "argv", ["sol skills", "install", "--project"])

    exit_code = skills_cli.main()

    assert exit_code == 0
    assert (target / ".claude" / "skills" / "journal").is_symlink()


def test_repo_root_resolution_works_from_arbitrary_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    result = resolve_user_skill()

    assert result.name == "solstone"
    assert (result / "SKILL.md").is_file()


def test_user_skill_missing_file_fails_loudly(monkeypatch, tmp_path, capsys):
    fake_talent = tmp_path / "talent"
    fake_talent.mkdir()
    home = _home(tmp_path, ".claude")
    monkeypatch.setattr(skills_cli.resources, "files", lambda _package: fake_talent)

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_user_skill()

    assert "solstone/talent/solstone/SKILL.md" in str(exc_info.value)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(skills_cli, "get_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["sol skills", "install"])

    exit_code = skills_cli.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error:" in captured.err
    assert "solstone/talent/solstone/SKILL.md" in captured.err
    assert "Traceback" not in captured.err
