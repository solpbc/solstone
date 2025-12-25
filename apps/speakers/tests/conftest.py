"""Self-contained fixtures for speakers app tests.

These fixtures are fully standalone and only depend on pytest builtins.
No shared dependencies from the root conftest.py are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from think.entities import normalize_entity_name


@pytest.fixture
def speakers_env(tmp_path, monkeypatch):
    """Create a temporary journal environment for speaker tests.

    Provides helpers to create:
    - Day directories with segment speaker embeddings
    - Facets with entities and voiceprints

    Usage:
        def test_example(speakers_env):
            env = speakers_env()
            env.create_segment("20240101", "143022_300", ["Speaker 1", "Speaker 2"])
            env.create_entity("test", "Alice Test")
            # Now JOURNAL_PATH is set and data exists
    """

    class SpeakersEnv:
        def __init__(self, journal_path: Path):
            self.journal = journal_path
            monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        def create_segment(
            self, day: str, segment_key: str, speakers: list[str]
        ) -> Path:
            """Create a segment with speaker embedding files."""
            audio_dir = self.journal / day / segment_key / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            for speaker in speakers:
                emb = np.random.randn(256).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                np.savez_compressed(audio_dir / f"{speaker}.npz", embedding=emb)

            return audio_dir

        def create_embedding(self, vector: list[float] | None = None) -> np.ndarray:
            """Create a normalized 256-dim embedding."""
            if vector is None:
                emb = np.random.randn(256).astype(np.float32)
            else:
                emb = np.array(vector + [0.0] * (256 - len(vector)), dtype=np.float32)
            return emb / np.linalg.norm(emb)

        def create_entity(
            self,
            facet: str,
            name: str,
            voiceprints: list[tuple[str, str]] | None = None,
        ) -> Path:
            """Create an entity with optional voiceprint files.

            Args:
                facet: Facet name
                name: Entity name
                voiceprints: Optional list of (day, segment_key) tuples for voiceprints
            """
            facet_dir = self.journal / "facets" / facet
            facet_dir.mkdir(parents=True, exist_ok=True)

            # Create entities.jsonl
            entities_file = facet_dir / "entities.jsonl"
            entity_data = {"type": "Person", "name": name, "description": "Test entity"}
            with open(entities_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entity_data) + "\n")

            # Create entity folder with voiceprints if specified
            if voiceprints:
                entity_dir = facet_dir / "entities" / normalize_entity_name(name)
                entity_dir.mkdir(parents=True, exist_ok=True)

                for day, segment_key in voiceprints:
                    emb = self.create_embedding()
                    np.savez_compressed(
                        entity_dir / f"{day}_{segment_key}.npz", embedding=emb
                    )

            return facet_dir

    def _create():
        return SpeakersEnv(tmp_path)

    return _create
