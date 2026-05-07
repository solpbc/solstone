# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

from solstone.apps.entities.routes import get_journal_entities_data
from solstone.think.entities.observations import add_observation, count_observations


def _entity(name: str) -> dict[str, Any]:
    return {
        "type": "Person",
        "name": name,
        "description": "Test entity",
        "attached_at": 1000,
        "updated_at": 1000,
    }


def test_observation_count_memo(entity_env, monkeypatch):
    facet = "personal"
    entity_name = "Alice Johnson"
    entity_env(
        attached=[_entity(entity_name)],
        observations=["Prefers async updates"],
        observation_entity=entity_name,
        facet=facet,
    )

    real_open = builtins.open
    observation_opens = 0

    def counting_open(file, *args, **kwargs):
        nonlocal observation_opens
        path = Path(file)
        if path.name == "observations.jsonl":
            observation_opens += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    assert count_observations(facet, entity_name) == 1
    assert count_observations(facet, entity_name) == 1
    assert observation_opens == 1

    add_observation(facet, entity_name, "Prefers morning meetings", "20260427")
    opens_after_write = observation_opens

    assert count_observations(facet, entity_name) == 2
    assert observation_opens == opens_after_write + 1


def test_relationship_cache_warm(entity_env, monkeypatch):
    facet = "personal"
    entity_name = "Alice Johnson"
    journal = entity_env(attached=[_entity(entity_name)], facet=facet)
    facet_dir = journal / "facets" / facet
    facet_dir.mkdir(parents=True, exist_ok=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Personal", "description": "Personal facet"}),
        encoding="utf-8",
    )

    real_open = builtins.open
    relationship_opens = 0

    def counting_open(file, *args, **kwargs):
        nonlocal relationship_opens
        path = Path(file)
        if (
            path.name == "entity.json"
            and "facets" in path.parts
            and path.parent.parent.name == "entities"
        ):
            relationship_opens += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    first = get_journal_entities_data()
    assert len(first["entities"]) == 1
    assert relationship_opens > 0

    relationship_opens = 0
    second = get_journal_entities_data()
    assert len(second["entities"]) == 1
    assert relationship_opens == 0
