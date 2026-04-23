# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import importlib
import json
from pathlib import Path

mod = importlib.import_module("apps.entities.maint.001_migrate_to_journal_entities")


def test_load_all_legacy_entities_no_facets_dir(tmp_path: Path) -> None:
    """Fresh journal with no facets/ dir must return (empty list, 0) — not bare []."""
    journal = tmp_path / "journal"
    journal.mkdir()

    entities, skipped = mod.load_all_legacy_entities(journal)

    assert entities == []
    assert skipped == 0


def test_migrate_entities_fresh_journal_no_crash(tmp_path: Path) -> None:
    """Fresh-install path: running the migration against a journal with no facets/
    must complete cleanly. Regression guard for the ValueError (expected 2, got 0)
    that crashed Ramon's install on 2026-04-22."""
    journal = tmp_path / "journal"
    journal.mkdir()

    summary = mod.migrate_entities(journal, dry_run=False)

    assert summary == {"loaded": 0, "canonicals": 0, "merges": 0, "relationships": 0}


def test_load_all_legacy_entities_with_facets(tmp_path: Path) -> None:
    """Normal path: journal with facets/ directory still returns (entities, skipped)."""
    journal = tmp_path / "journal"
    facet_dir = journal / "facets" / "work"
    facet_dir.mkdir(parents=True)

    entities_file = facet_dir / "entities.jsonl"
    entities_file.write_text(
        json.dumps(
            {
                "name": "Acme Corp",
                "type": "organization",
                "description": "A company.",
                "aka": ["Acme"],
                "is_principal": False,
                "detached": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "name": "Ghost Entity",
                "type": "person",
                "description": "Soft-deleted.",
                "aka": [],
                "is_principal": False,
                "detached": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entities, skipped = mod.load_all_legacy_entities(journal)

    assert len(entities) == 1
    assert entities[0].name == "Acme Corp"
    assert entities[0].facet == "work"
    assert skipped == 1
