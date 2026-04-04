# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_generate_agents_md_uses_fixture_journal(monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    agents_path = project_root / "AGENTS.md"
    original_content = agents_path.read_text(encoding="utf-8")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", "tests/fixtures/journal")

    try:
        subprocess.run(
            [sys.executable, "scripts/generate_agents_md.py"],
            cwd=project_root,
            check=True,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        generated = agents_path.read_text(encoding="utf-8")
        assert generated.startswith(
            "<!-- generated from sol/identity.md — do not edit directly -->"
        )
        assert "Sol" in generated
        assert "Test User" in generated
        assert "$Agent_name" not in generated
        assert "$name" not in generated
    finally:
        agents_path.write_text(original_content, encoding="utf-8")


def test_generate_agents_md_no_config(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parent.parent
    agents_path = project_root / "AGENTS.md"
    original_content = agents_path.read_text(encoding="utf-8")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    try:
        subprocess.run(
            [sys.executable, "scripts/generate_agents_md.py"],
            cwd=project_root,
            check=True,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        generated = agents_path.read_text(encoding="utf-8")
        assert generated.startswith(
            "<!-- generated from sol/identity.md — do not edit directly -->"
        )
        assert "your journal owner" in generated
        assert "Sol" in generated
        assert "$Agent_name" not in generated
        assert "$name" not in generated
        assert "$pronouns_subject" not in generated
    finally:
        agents_path.write_text(original_content, encoding="utf-8")
