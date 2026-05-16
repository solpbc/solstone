# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from solstone.convey.config import DEFAULT_RAIL_APPS
from solstone.think.maint import MaintTask, get_state_file, run_pending_tasks

TASK_NAME = "003_seed_default_app_navigation"
TASK_MODULE = f"solstone.apps.settings.maint.{TASK_NAME}"


def _write_convey_config(journal: Path, payload: dict) -> None:
    config_path = journal / "config" / "convey.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_convey_config(journal: Path) -> dict:
    return json.loads((journal / "config" / "convey.json").read_text("utf-8"))


def _task_env(journal: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["SOLSTONE_JOURNAL"] = str(journal)
    env["PYTHONPATH"] = (
        str(root)
        if not env.get("PYTHONPATH")
        else str(root) + os.pathsep + env["PYTHONPATH"]
    )
    return env


def _run_task(journal: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", TASK_MODULE],
        cwd=cwd,
        env=_task_env(journal),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_seed_task_writes_resolved_journal_not_cwd(tmp_path):
    journal = tmp_path / "journal"
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    result = _run_task(journal, cwd)

    assert result.returncode == 0, result.stderr
    config = _read_convey_config(journal)
    assert config["apps"]["starred"] == DEFAULT_RAIL_APPS
    assert config["apps"]["order"] == DEFAULT_RAIL_APPS
    assert not (cwd / "config" / "convey.json").exists()


def test_seed_task_noops_when_keys_present(tmp_path):
    journal = tmp_path / "journal"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    payload = {"apps": {"starred": [], "order": []}}
    _write_convey_config(journal, payload)

    result = _run_task(journal, cwd)

    assert result.returncode == 0, result.stderr
    assert _read_convey_config(journal) == payload
    assert "already present" in result.stdout


def test_seed_task_write_failure_exits_nonzero(tmp_path):
    journal = tmp_path / "journal"
    cwd = tmp_path / "cwd"
    config_dir = journal / "config"
    cwd.mkdir()
    config_dir.mkdir(parents=True)
    config_dir.chmod(0o500)

    try:
        result = _run_task(journal, cwd)
    finally:
        config_dir.chmod(0o700)

    assert result.returncode != 0
    assert "PERSIST failed" in result.stderr
    assert not (config_dir / "convey.json").exists()


def test_successful_maint_task_is_skipped(monkeypatch, tmp_path):
    journal = tmp_path / "journal"
    payload = {"apps": {"starred": [], "order": []}}
    _write_convey_config(journal, payload)

    task = MaintTask(
        app="settings",
        name=TASK_NAME,
        script_path=Path("solstone/apps/settings/maint") / f"{TASK_NAME}.py",
    )
    monkeypatch.setattr("solstone.think.maint.discover_tasks", lambda: [task])

    state_file = get_state_file(journal, "settings", TASK_NAME)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        '{"event": "exec", "ts": 1000}\n'
        '{"event": "exit", "ts": 2000, "exit_code": 0}\n',
        encoding="utf-8",
    )

    ran, succeeded = run_pending_tasks(journal)

    assert (ran, succeeded) == (0, 0)
    assert _read_convey_config(journal) == payload
