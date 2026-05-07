# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import json
from pathlib import Path

mod = importlib.import_module(
    "solstone.apps.sol.maint.006_rename_unified_triage_providers"
)


def _write_journal_config(journal: Path, data: object) -> Path:
    config_dir = journal / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "journal.json"
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return config_path


def test_rename_unified_and_remove_triage_idempotent(tmp_path):
    config_path = _write_journal_config(
        tmp_path,
        {
            "providers": {
                "contexts": {
                    "talent.system.unified": {"provider": "openai"},
                    "talent.system.triage": {"provider": "anthropic"},
                    "talent.system.digest": {"provider": "google"},
                }
            }
        },
    )

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.renamed == 1
    assert summary.removed == 1
    assert summary.preserved == 0
    assert summary.errors == 0
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "talent.system.unified" not in data["providers"]["contexts"]
    assert "talent.system.triage" not in data["providers"]["contexts"]
    assert data["providers"]["contexts"]["talent.system.chat"] == {"provider": "openai"}
    assert data["providers"]["contexts"]["talent.system.digest"] == {
        "provider": "google"
    }

    before_bytes = config_path.read_bytes()
    before_mtime_ns = config_path.stat().st_mtime_ns

    rerun = mod.run_migration(tmp_path, dry_run=False)

    assert rerun.renamed == 0
    assert rerun.removed == 0
    assert rerun.preserved == 0
    assert rerun.errors == 0
    assert rerun.skipped_reason is None
    assert config_path.read_bytes() == before_bytes
    assert config_path.stat().st_mtime_ns == before_mtime_ns


def test_preserves_existing_chat_context_when_unified_exists(tmp_path):
    config_path = _write_journal_config(
        tmp_path,
        {
            "providers": {
                "contexts": {
                    "talent.system.unified": {"provider": "openai"},
                    "talent.system.chat": {"provider": "google"},
                }
            }
        },
    )

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.renamed == 0
    assert summary.removed == 0
    assert summary.preserved == 1
    assert summary.errors == 0
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "talent.system.unified" not in data["providers"]["contexts"]
    assert data["providers"]["contexts"]["talent.system.chat"] == {"provider": "google"}


def test_noop_when_no_legacy_provider_contexts_present(tmp_path):
    config_path = _write_journal_config(
        tmp_path,
        {
            "providers": {
                "contexts": {
                    "talent.system.chat": {"provider": "openai"},
                    "talent.system.digest": {"provider": "google"},
                }
            }
        },
    )
    before_bytes = config_path.read_bytes()
    before_mtime_ns = config_path.stat().st_mtime_ns

    summary = mod.run_migration(tmp_path, dry_run=False)

    assert summary.renamed == 0
    assert summary.removed == 0
    assert summary.preserved == 0
    assert summary.errors == 0
    assert summary.skipped_reason is None
    assert config_path.read_bytes() == before_bytes
    assert config_path.stat().st_mtime_ns == before_mtime_ns
