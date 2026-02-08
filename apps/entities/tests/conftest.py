# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for entities app tests."""

from __future__ import annotations

import pytest

from think.entities.observations import add_observation
from think.entities.saving import save_entities


@pytest.fixture
def entity_env(tmp_path, monkeypatch):
    """Create a temporary journal with entity data.

    Usage:
        def test_example(entity_env):
            entity_env(attached=[
                {"type": "Person", "name": "Alice", "description": "Friend"}
            ])
            # JOURNAL_PATH is set, entity files exist
    """
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

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

    return _create
