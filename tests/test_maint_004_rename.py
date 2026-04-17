# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import importlib
from pathlib import Path

mod = importlib.import_module("apps.sol.maint.004_rename_agents_to_talents")


def _patch_journal(monkeypatch, journal: Path, day: str = "20260417") -> None:
    monkeypatch.setattr(mod, "get_journal", lambda: str(journal))
    monkeypatch.setattr(mod, "day_dirs", lambda: {day: str(journal / day)})
    monkeypatch.setattr(
        mod,
        "iter_segments",
        lambda _day: [
            ("default", "090000_300", journal / day / "default" / "090000_300")
        ],
    )


def test_run_migration_moves_all_paths(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    day = journal / "20260417"
    segment = day / "default" / "090000_300"

    (journal / "agents").mkdir(parents=True)
    (journal / "agents" / "root.jsonl").write_text("{}\n", encoding="utf-8")
    (journal / "health").mkdir(parents=True)
    (journal / "health" / "agents.json").write_text("{}", encoding="utf-8")
    (day / "agents").mkdir(parents=True)
    (day / "agents" / "flow.md").write_text("# flow\n", encoding="utf-8")
    (segment / "agents").mkdir(parents=True)
    (segment / "agents" / "screen.md").write_text("# screen\n", encoding="utf-8")

    _patch_journal(monkeypatch, journal)

    summary, collisions = mod.run_migration(journal, dry_run=False)

    assert collisions == []
    assert summary.discovered == 4
    assert summary.moved == 4
    assert summary.skipped == 0
    assert summary.errors == 0
    assert summary.collisions == 0
    assert (journal / "talents" / "root.jsonl").exists()
    assert (journal / "health" / "talents.json").exists()
    assert (day / "talents" / "flow.md").exists()
    assert (segment / "talents" / "screen.md").exists()


def test_run_migration_aborts_on_collision(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    day = journal / "20260417"

    (journal / "agents").mkdir(parents=True)
    (journal / "agents" / "root.jsonl").write_text("{}\n", encoding="utf-8")
    (journal / "talents").mkdir(parents=True)
    (day / "agents").mkdir(parents=True)
    (day / "agents" / "flow.md").write_text("# flow\n", encoding="utf-8")

    _patch_journal(monkeypatch, journal)

    summary, collisions = mod.run_migration(journal, dry_run=False)

    assert summary.collisions == 1
    assert summary.moved == 0
    assert len(collisions) == 1
    assert (journal / "agents" / "root.jsonl").exists()
    assert not (day / "talents").exists()


def test_run_migration_reports_already_migrated(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    day = journal / "20260417"
    segment = day / "default" / "090000_300"

    (journal / "talents").mkdir(parents=True)
    (journal / "talents" / "root.jsonl").write_text("{}\n", encoding="utf-8")
    (journal / "health").mkdir(parents=True)
    (journal / "health" / "talents.json").write_text("{}", encoding="utf-8")
    (day / "talents").mkdir(parents=True)
    (day / "talents" / "flow.md").write_text("# flow\n", encoding="utf-8")
    (segment / "talents").mkdir(parents=True)
    (segment / "talents" / "screen.md").write_text("# screen\n", encoding="utf-8")

    _patch_journal(monkeypatch, journal)

    summary, collisions = mod.run_migration(journal, dry_run=False)

    assert collisions == []
    assert summary.discovered == 0
    assert summary.moved == 0
    assert summary.skipped == 4
    assert summary.errors == 0
    assert summary.collisions == 0
