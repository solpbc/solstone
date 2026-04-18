# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for entities app tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.speakers.tests.conftest import speakers_env as _speakers_env
from think.entities.journal import clear_journal_entity_cache
from think.entities.loading import clear_entity_loading_cache
from think.entities.observations import (
    add_observation,
    clear_observation_cache,
    save_observations,
)
from think.entities.relationships import clear_relationship_caches
from think.entities.saving import save_entities


@pytest.fixture
def speakers_env(tmp_path, monkeypatch):
    yield from _speakers_env.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def entity_env(tmp_path, monkeypatch):
    """Create a temporary journal with entity data.

    Usage:
        def test_example(entity_env):
            entity_env(attached=[
                {"type": "Person", "name": "Alice", "description": "Friend"}
            ])
            # _SOLSTONE_JOURNAL_OVERRIDE is set, entity files exist
    """
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    clear_journal_entity_cache()
    clear_entity_loading_cache()
    clear_relationship_caches()
    clear_observation_cache()
    import think.utils

    think.utils._journal_path_cache = None

    def _create(
        attached: list[dict] | None = None,
        detected: list[dict] | None = None,
        day: str | None = None,
        facet: str = "personal",
        observations: list[str] | None = None,
        observation_entity: str | None = None,
    ):
        if attached:
            save_entities(facet, attached, day=None)
        if detected and day:
            save_entities(facet, detected, day=day)
        if observations and observation_entity:
            for i, content in enumerate(observations, 1):
                add_observation(facet, observation_entity, content, i)
        return tmp_path

    yield _create
    clear_journal_entity_cache()
    clear_entity_loading_cache()
    clear_relationship_caches()
    clear_observation_cache()
    import think.utils

    think.utils._journal_path_cache = None


@pytest.fixture
def entity_move_env(tmp_path, monkeypatch):
    """Create a two-facet environment for entity move tests."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    clear_journal_entity_cache()
    clear_entity_loading_cache()
    clear_relationship_caches()
    clear_observation_cache()
    import think.utils

    think.utils._journal_path_cache = None

    def _create(
        entity_name: str = "Alice Johnson",
        src_facet: str = "work",
        dst_facet: str = "personal",
        src_observations: list[dict] | None = None,
        dst_observations: list[dict] | None = None,
        create_dst_entity: bool = False,
    ):
        for facet in [src_facet, dst_facet]:
            facet_dir = tmp_path / "facets" / facet
            facet_dir.mkdir(parents=True, exist_ok=True)
            (facet_dir / "facet.json").write_text(
                json.dumps({"title": f"Test {facet}", "description": "Test facet"}),
                encoding="utf-8",
            )

        entity = {
            "type": "Person",
            "name": entity_name,
            "description": "Friend",
            "attached_at": 1000,
            "updated_at": 1000,
        }
        save_entities(src_facet, [entity], day=None)

        if src_observations:
            save_observations(src_facet, entity_name, src_observations)

        if create_dst_entity:
            save_entities(dst_facet, [entity], day=None)

        if dst_observations:
            save_observations(dst_facet, entity_name, dst_observations)

        return tmp_path, src_facet, dst_facet, entity_name

    yield _create
    clear_journal_entity_cache()
    clear_entity_loading_cache()
    clear_relationship_caches()
    clear_observation_cache()
    import think.utils

    think.utils._journal_path_cache = None
