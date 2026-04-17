# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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


def test_make_skills_idempotent(tmp_path):
    repo_root = _repo_root()
    temp_root = tmp_path / "repo"
    temp_root.mkdir()

    shutil.copy2(repo_root / "Makefile", temp_root / "Makefile")
    shutil.copytree(repo_root / "talent", temp_root / "talent", symlinks=True)
    shutil.copytree(repo_root / "apps", temp_root / "apps", symlinks=True)
    shutil.copytree(repo_root / "journal", temp_root / "journal", symlinks=True)

    subprocess.run(
        ["make", "skills"],
        cwd=temp_root,
        check=True,
        capture_output=True,
        text=True,
    )

    def link_state(root: Path) -> dict[str, tuple[str, int]]:
        return {
            path.relative_to(root).as_posix(): (
                path.readlink().as_posix(),
                path.lstat().st_mtime_ns,
            )
            for path in sorted(root.rglob("*"))
            if path.is_symlink()
        }

    first = link_state(temp_root / "journal")

    subprocess.run(
        ["make", "skills"],
        cwd=temp_root,
        check=True,
        capture_output=True,
        text=True,
    )

    second = link_state(temp_root / "journal")
    assert first == second


def test_skill_discovery_from_journal_cwd():
    repo_root = _repo_root()
    skill_path = repo_root / "journal" / ".claude" / "skills" / "journal" / "SKILL.md"

    assert skill_path.is_file()
    assert skill_path.read_text(encoding="utf-8").startswith("---")
