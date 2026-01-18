# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_journal(tmp_path: Path) -> Path:
    src = FIXTURES / "journal"
    dest = tmp_path / "journal"
    shutil.copytree(src, dest)
    return dest


def test_main_runs(tmp_path, monkeypatch):
    mod = importlib.import_module("think.dream")
    journal = copy_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    called = []

    def mock_run_command(cmd, day):
        called.append(cmd)
        return True  # Return success

    monkeypatch.setattr(mod, "run_command", mock_run_command)
    # Also mock run_daily_agents to avoid agent execution
    monkeypatch.setattr(mod, "run_daily_agents", lambda day: (0, 0))
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        ["sol dream", "--day", "20240101", "--force", "--verbose", "--skip-agents"],
    )
    mod.main()
    assert any(c[0] == "sol" and c[1] == "sense" for c in called)
    assert any(c[0] == "sol" and c[1] == "insight" for c in called)
    # Verify indexer is called with --rescan (light mode)
    indexer_cmds = [c for c in called if c[0] == "sol" and c[1] == "indexer"]
    assert len(indexer_cmds) == 1
    assert "--rescan" in indexer_cmds[0]
