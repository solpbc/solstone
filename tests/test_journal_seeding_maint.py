# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def journal_path(tmp_path, monkeypatch):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "journal.json").write_text(json.dumps({}))
    return tmp_path


def _run_main(journal_path):
    env = os.environ.copy()
    env["_SOLSTONE_JOURNAL_OVERRIDE"] = str(journal_path)
    return subprocess.run(
        [sys.executable, "-m", "apps.sol.maint.003_seed_agents_md"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_seed_agents_md_creates_all_files(journal_path):
    docs_text = Path("docs/JOURNAL.md").read_text(encoding="utf-8")

    result = _run_main(journal_path)

    assert result.returncode == 0
    talents_path = journal_path / "AGENTS.md"
    claude_path = journal_path / "CLAUDE.md"
    gemini_path = journal_path / "GEMINI.md"
    assert talents_path.read_text(encoding="utf-8") == docs_text
    assert claude_path.is_symlink()
    assert claude_path.readlink() == Path("AGENTS.md")
    assert gemini_path.is_symlink()
    assert gemini_path.readlink() == Path("AGENTS.md")


def test_seed_agents_md_is_noop_when_already_seeded(journal_path):
    docs_text = Path("docs/JOURNAL.md").read_text(encoding="utf-8")
    talents_path = journal_path / "AGENTS.md"
    talents_path.write_text(docs_text, encoding="utf-8")
    (journal_path / "CLAUDE.md").symlink_to("AGENTS.md")
    (journal_path / "GEMINI.md").symlink_to("AGENTS.md")
    before = {
        "talents": talents_path.stat().st_mtime_ns,
        "claude": (journal_path / "CLAUDE.md").lstat().st_mtime_ns,
        "gemini": (journal_path / "GEMINI.md").lstat().st_mtime_ns,
    }

    result = _run_main(journal_path)

    after = {
        "talents": talents_path.stat().st_mtime_ns,
        "claude": (journal_path / "CLAUDE.md").lstat().st_mtime_ns,
        "gemini": (journal_path / "GEMINI.md").lstat().st_mtime_ns,
    }
    assert result.returncode == 0
    assert before == after


def test_seed_agents_md_refreshes_on_drift(journal_path):
    talents_path = journal_path / "AGENTS.md"
    talents_path.write_text("stale content", encoding="utf-8")
    (journal_path / "CLAUDE.md").symlink_to("AGENTS.md")
    (journal_path / "GEMINI.md").symlink_to("AGENTS.md")
    docs_text = Path("docs/JOURNAL.md").read_text(encoding="utf-8")

    result = _run_main(journal_path)

    assert result.returncode == 0
    assert talents_path.read_text(encoding="utf-8") == docs_text
