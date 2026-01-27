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
    generators_called = False

    def mock_run_command(cmd, day):
        called.append(cmd)
        return True  # Return success

    def mock_run_queued_command(cmd, day, timeout=600):
        called.append(cmd)
        return True  # Return success

    def mock_run_generators_via_cortex(day, force, segment=None):
        nonlocal generators_called
        generators_called = True
        return (1, 0)  # 1 success, 0 failures

    monkeypatch.setattr(mod, "run_command", mock_run_command)
    monkeypatch.setattr(mod, "run_queued_command", mock_run_queued_command)
    monkeypatch.setattr(mod, "run_generators_via_cortex", mock_run_generators_via_cortex)
    # Also mock run_daily_agents to avoid agent execution
    monkeypatch.setattr(mod, "run_daily_agents", lambda day: (0, 0))
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        ["sol dream", "--day", "20240101", "--force", "--verbose", "--skip-agents"],
    )
    mod.main()
    assert any(c[0] == "sol" and c[1] == "sense" for c in called)
    # Generators now run via cortex, not as direct subprocess
    assert generators_called, "run_generators_via_cortex should have been called"
    # Verify indexer is called with --rescan (light mode) via queued command
    indexer_cmds = [c for c in called if c[0] == "sol" and c[1] == "indexer"]
    assert len(indexer_cmds) == 1
    assert "--rescan" in indexer_cmds[0]
