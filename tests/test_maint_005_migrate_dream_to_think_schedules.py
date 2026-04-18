import importlib
import json
from pathlib import Path

mod = importlib.import_module("apps.sol.maint.005_migrate_dream_to_think_schedules")
DREAM = "dream"


def _write_schedules(journal: Path, data: object) -> Path:
    config_dir = journal / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    schedules_path = config_dir / "schedules.json"
    schedules_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return schedules_path


def test_happy_path_rewrites_weekly_agents(tmp_path):
    schedules_path = _write_schedules(
        tmp_path,
        {
            "daily_time": "03:17",
            "weekly-agents": {
                "cmd": ["sol", DREAM, "--weekly", "-v"],
                "every": "weekly",
                "enabled": True,
            },
        },
    )

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.discovered == 1
    assert summary.rewritten == 1
    assert summary.preserved == 1
    assert summary.errors == 0
    data = json.loads(schedules_path.read_text(encoding="utf-8"))
    assert data["daily_time"] == "03:17"
    assert data["weekly-agents"]["cmd"] == ["sol", "think", "--weekly", "-v"]
    assert data["weekly-agents"]["every"] == "weekly"
    assert data["weekly-agents"]["enabled"] is True


def test_custom_dream_entry_is_rewritten(tmp_path):
    schedules_path = _write_schedules(
        tmp_path,
        {
            "my-custom": {
                "cmd": ["sol", DREAM, "--segments"],
                "every": "daily",
                "enabled": True,
            }
        },
    )

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.discovered == 1
    assert summary.rewritten == 1
    assert summary.preserved == 0
    assert summary.errors == 0
    data = json.loads(schedules_path.read_text(encoding="utf-8"))
    assert data["my-custom"] == {
        "cmd": ["sol", "think", "--segments"],
        "every": "daily",
        "enabled": True,
    }


def test_non_dream_entries_preserved_byte_for_byte(tmp_path):
    initial = {
        "daily_time": "03:17",
        "sync:plaud": {
            "cmd": ["sol", "import", "--sync", "plaud", "--save"],
            "every": "hourly",
            "enabled": True,
        },
        "heartbeat": {
            "cmd": ["sol", "heartbeat"],
            "every": "daily",
            "enabled": True,
        },
        "weekly-agents": {
            "cmd": ["sol", DREAM, "--weekly", "-v"],
            "every": "weekly",
            "enabled": True,
        },
    }
    schedules_path = _write_schedules(tmp_path, initial)

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.discovered == 1
    assert summary.rewritten == 1
    assert summary.preserved == 3
    assert summary.errors == 0
    data = json.loads(schedules_path.read_text(encoding="utf-8"))
    assert data["daily_time"] == initial["daily_time"]
    assert data["sync:plaud"] == initial["sync:plaud"]
    assert data["heartbeat"] == initial["heartbeat"]
    assert data["weekly-agents"]["cmd"] == ["sol", "think", "--weekly", "-v"]


def test_idempotent_rerun(tmp_path):
    initial = {
        "weekly-agents": {
            "cmd": ["sol", "think", "--weekly", "-v"],
            "every": "weekly",
            "enabled": True,
        }
    }
    schedules_path = _write_schedules(tmp_path, initial)
    before_bytes = schedules_path.read_bytes()
    before_mtime_ns = schedules_path.stat().st_mtime_ns

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.discovered == 0
    assert summary.rewritten == 0
    assert summary.preserved == 1
    assert summary.errors == 0
    assert summary.skipped_reason is None
    assert schedules_path.read_bytes() == before_bytes
    assert json.loads(schedules_path.read_text(encoding="utf-8")) == initial
    assert schedules_path.stat().st_mtime_ns == before_mtime_ns


def test_missing_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.skipped_reason == "no file"
    assert summary.errors == 0
    assert summary.discovered == 0
    assert not (config_dir / "schedules.json").exists()


def test_empty_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    schedules_path = config_dir / "schedules.json"
    schedules_path.write_text("", encoding="utf-8")

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.skipped_reason == "empty file"
    assert summary.errors == 0
    assert summary.discovered == 0
    assert schedules_path.read_text(encoding="utf-8") == ""


def test_malformed_json(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    schedules_path = config_dir / "schedules.json"
    schedules_path.write_text("{not json", encoding="utf-8")

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.skipped_reason == "unparseable"
    assert summary.errors == 0
    assert summary.discovered == 0
    assert schedules_path.read_text(encoding="utf-8") == "{not json"


def test_dry_run_does_not_write(tmp_path):
    schedules_path = _write_schedules(
        tmp_path,
        {
            "weekly-agents": {
                "cmd": ["sol", DREAM, "--weekly", "-v"],
                "every": "weekly",
                "enabled": True,
            }
        },
    )
    before_bytes = schedules_path.read_bytes()
    config_dir = tmp_path / "config"

    summary = mod.run_migration(tmp_path, dry_run=True)

    assert summary.discovered == 1
    assert summary.rewritten == 1
    assert summary.preserved == 0
    assert summary.errors == 0
    assert schedules_path.read_bytes() == before_bytes
    assert list(config_dir.glob(".schedules_*.tmp")) == []


def test_atomic_write_failure_cleans_up_tmpfile(tmp_path, monkeypatch):
    schedules_path = _write_schedules(
        tmp_path,
        {
            "weekly-agents": {
                "cmd": ["sol", DREAM, "--weekly", "-v"],
                "every": "weekly",
                "enabled": True,
            }
        },
    )
    before_bytes = schedules_path.read_bytes()
    config_dir = tmp_path / "config"

    def _boom(self, target):
        raise OSError("boom")

    monkeypatch.setattr(mod.Path, "replace", _boom)

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.errors >= 1
    assert schedules_path.read_bytes() == before_bytes
    assert list(config_dir.glob(".schedules_*.tmp")) == []
