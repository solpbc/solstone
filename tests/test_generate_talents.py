# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path


def test_root_agents_md_is_hand_maintained():
    project_root = Path(__file__).resolve().parent.parent
    agents_path = project_root / "AGENTS.md"
    content = agents_path.read_text(encoding="utf-8")

    assert content.startswith("# solstone Developer Guide")
    assert "generated from sol/identity.md" not in content
    assert "docs/project-structure.md" in content
    assert "journal/AGENTS.md" in content


def test_root_agent_symlinks_point_to_agents():
    project_root = Path(__file__).resolve().parent.parent
    claude_path = project_root / "CLAUDE.md"
    gemini_path = project_root / "GEMINI.md"

    assert claude_path.is_symlink()
    assert gemini_path.is_symlink()
    assert claude_path.readlink() == Path("AGENTS.md")
    assert gemini_path.readlink() == Path("AGENTS.md")
