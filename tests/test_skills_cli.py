# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
import shutil
import sys
from importlib import resources
from pathlib import Path

import pytest

from solstone.think import skills_cli
from solstone.think.skills_cli import (
    GLOBAL_SKIP_MESSAGE,
    discover_user_bundles,
    install_project,
    install_user,
    list_project_status,
    list_user_status,
    uninstall_user,
)


def _write_skill(path: Path, content: bytes | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_bytes(content or b"---\nname: test\n---\n")


def _mini_user_repo(tmp_path: Path, content: bytes | None = None) -> Path:
    bundle_dir = tmp_path / "bundles"
    _write_skill(bundle_dir / "solstone", content)
    return bundle_dir


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
    source = repo / "solstone" / "SKILL.md"
    assert (
        home / ".claude" / "skills" / "solstone" / "SKILL.md"
    ).read_bytes() == source.read_bytes()
    assert (
        home / ".codex" / "skills" / "solstone" / "SKILL.md"
    ).read_bytes() == source.read_bytes()


def test_install_user_skips_codex_when_parent_absent(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")

    report = install_user(repo, home, ["all"])

    assert report.error_count == 0
    assert any(
        row.agent == "claude" and row.action == "installed" for row in report.rows
    )
    assert any(row.agent == "codex" and row.action == "skipped" for row in report.rows)


def test_install_user_skips_gemini_silently_in_default_all(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")

    default_report = install_user(repo, home, ["all"])
    explicit_report = install_user(repo, home, ["gemini"])

    assert all(row.agent != "gemini" for row in default_report.rows)
    assert explicit_report.rows == [
        skills_cli.ActionRow(
            "gemini",
            "",
            "skipped",
            home / ".gemini",
            reason=f"config dir absent at {home / '.gemini'}",
        )
    ]


def test_install_user_all_skipped_prints_global_message(tmp_path, capsys):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path)

    report = install_user(repo, home, ["all"])
    skills_cli._print_report(report, "install")

    captured = capsys.readouterr()
    assert report.error_count == 0
    assert GLOBAL_SKIP_MESSAGE in captured.out


def test_install_user_replaces_modified_source(tmp_path):
    repo = _mini_user_repo(tmp_path, b"first")
    home = _home(tmp_path, ".claude")
    install_user(repo, home, ["claude"])
    (repo / "solstone" / "SKILL.md").write_bytes(b"second")

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


def test_install_user_refuses_symlink_target(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path)

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 1
    assert target.is_symlink()
    assert report.rows[0].reason == "refusing to overwrite symlink"


def test_install_user_refuses_regular_file_target(tmp_path):
    repo = _mini_user_repo(tmp_path)
    home = _home(tmp_path, ".claude")
    target = home / ".claude" / "skills" / "solstone"
    target.parent.mkdir(parents=True)
    target.write_text("not a dir", encoding="utf-8")

    report = install_user(repo, home, ["claude"])

    assert report.error_count == 1
    assert target.is_file()
    assert report.rows[0].reason == "refusing to overwrite non-directory"


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
    bundle_dir = Path(str(resources.files("solstone") / "_user_bundles"))

    bundles = discover_user_bundles(bundle_dir)

    assert [bundle.name for bundle in bundles] == ["solstone"]
