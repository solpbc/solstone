# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from think.skills_cli import install_project


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _assert_inside_repo(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    assert resolved.is_relative_to(repo_root)


def _tracked_symlinks(*roots: str) -> list[Path]:
    repo_root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files", *roots],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        repo_root / line
        for line in result.stdout.splitlines()
        if line and (repo_root / line).is_symlink()
    ]


def test_journal_skill_references_exist_and_linked():
    repo_root = _repo_root()
    skill_path = repo_root / "talent" / "journal" / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")
    references = [
        "references/cli.md",
        "references/config.md",
        "references/facets.md",
        "references/captures.md",
        "references/logs.md",
        "references/storage.md",
    ]

    for rel_path in references:
        ref_path = skill_path.parent / rel_path
        assert ref_path.exists()
        assert ref_path.read_text(encoding="utf-8").strip()
        assert rel_path in skill_text


def test_journal_template_symlinks_resolve_inside_repo():
    repo_root = _repo_root()
    for path in _tracked_symlinks("journal", "tests/fixtures/journal"):
        _assert_inside_repo(path, repo_root)


@pytest.mark.timeout(30)
def test_make_skills_idempotent(tmp_path):
    """The make skills wrapper delegates to the idempotent project installer."""
    repo_root = _repo_root()
    temp_root = tmp_path / "repo"
    temp_root.mkdir()

    shutil.copy2(repo_root / "Makefile", temp_root / "Makefile")
    shutil.copytree(repo_root / "talent", temp_root / "talent", symlinks=True)
    shutil.copytree(repo_root / "apps", temp_root / "apps", symlinks=True)

    def link_state(root: Path) -> dict[str, tuple[str, int]]:
        return {
            path.relative_to(root).as_posix(): (
                path.readlink().as_posix(),
                path.lstat().st_mtime_ns,
            )
            for path in sorted(root.rglob("*"))
            if path.is_symlink()
        }

    first_report = install_project(temp_root, temp_root, ["all"])
    assert first_report.error_count == 0

    first = link_state(temp_root)

    second_report = install_project(temp_root, temp_root, ["all"])
    assert second_report.error_count == 0

    second = link_state(temp_root)
    assert first == second
    assert (
        temp_root / ".claude" / "skills" / "journal"
    ).readlink().as_posix() == "../../talent/journal"

    # Skill-discovery contract: claude code looks at <cwd>/.claude/skills/, so
    # after project skill installation the cwd path must resolve to a real
    # SKILL.md whose content starts with frontmatter. Verifying it here against
    # the tmp tree means the test is hermetic — it doesn't depend on the dev box
    # having previously run `make install` or `make skills`.
    discovered = temp_root / ".claude" / "skills" / "journal" / "SKILL.md"
    assert discovered.is_file()
    assert discovered.read_text(encoding="utf-8").startswith("---")
