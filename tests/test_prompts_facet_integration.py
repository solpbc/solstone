# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration coverage for prompt-layer facet substitution."""

import json
from pathlib import Path

from slugify import slugify

from think.prompts import _resolve_facets


def setup_entities_new_structure(
    journal_path: Path,
    facet: str,
    entities: list[dict],
) -> None:
    """Create journal-level entity files and facet relationships for tests."""
    for entity in entities:
        name = entity.get("name", "")
        desc = entity.get("description", "")
        entity_id = slugify(name, separator="_")
        if not entity_id:
            continue

        journal_entity_dir = journal_path / "entities" / entity_id
        journal_entity_dir.mkdir(parents=True, exist_ok=True)
        (journal_entity_dir / "entity.json").write_text(
            json.dumps({"id": entity_id, "name": name, "type": entity.get("type", "")}),
            encoding="utf-8",
        )

        facet_entity_dir = journal_path / "facets" / facet / "entities" / entity_id
        facet_entity_dir.mkdir(parents=True, exist_ok=True)
        (facet_entity_dir / "entity.json").write_text(
            json.dumps({"entity_id": entity_id, "description": desc}),
            encoding="utf-8",
        )
        (facet_entity_dir / "observations.jsonl").write_text(
            json.dumps(
                {
                    "content": f"Observed {name}",
                    "observed_at": "2026-04-20T12:00:00Z",
                    "source_day": "20260420",
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_resolve_facets_none_uses_capped_facet_summaries(tmp_path, monkeypatch):
    """The prompt-layer $facets resolver uses capped facet_summaries output."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "journal.json").write_text(
        json.dumps({"identity": {"name": "Test User", "preferred": "Tester"}}),
        encoding="utf-8",
    )
    facet_dir = tmp_path / "facets" / "capped"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Capped Facet", "description": "Prompt integration"}),
        encoding="utf-8",
    )
    setup_entities_new_structure(
        tmp_path,
        "capped",
        [
            {
                "type": "Person",
                "name": f"Entity {index:02d}",
                "description": f"Description {index:02d}",
            }
            for index in range(1, 26)
        ],
    )
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    resolved = _resolve_facets(None)

    assert "## Available Facets" in resolved
    assert "**Capped Facet** (`capped`)" in resolved
    assert "    - _and 5 more entities_" in resolved
