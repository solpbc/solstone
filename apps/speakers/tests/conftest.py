# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for speakers app tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from think.entities import entity_slug


@pytest.fixture
def speakers_env(tmp_path, monkeypatch):
    """Create a temporary journal environment for speaker tests.

    Provides helpers to create:
    - Day directories with sentence embeddings
    - Journal-level entities with voiceprints

    Usage:
        def test_example(speakers_env):
            env = speakers_env()
            env.create_segment("20240101", "143022_300", ["mic_audio"])
            env.create_entity("Alice Test")
            # Now JOURNAL_PATH is set and data exists
    """

    class SpeakersEnv:
        def __init__(self, journal_path: Path):
            self.journal = journal_path
            monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        def create_segment(
            self, day: str, segment_key: str, sources: list[str], num_sentences: int = 5
        ) -> Path:
            """Create a segment with sentence embeddings.

            Creates both JSONL transcripts and NPZ embedding files.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                sources: List of audio sources (e.g., ["mic_audio", "sys_audio"])
                num_sentences: Number of sentences to create
            """
            segment_dir = self.journal / day / segment_key
            segment_dir.mkdir(parents=True, exist_ok=True)

            for source in sources:
                # Create JSONL transcript
                jsonl_path = segment_dir / f"{source}.jsonl"
                lines = [json.dumps({"raw": f"{source}.flac", "model": "medium.en"})]

                # Parse segment_key to get base time (e.g., "143022_300" -> 14:30:22)
                # This matches real transcriber output which uses absolute timestamps
                time_part = segment_key.split("_")[0]
                base_h = int(time_part[0:2])
                base_m = int(time_part[2:4])
                base_s = int(time_part[4:6])
                base_seconds = base_h * 3600 + base_m * 60 + base_s

                for i in range(num_sentences):
                    offset = i * 5  # 5 seconds per sentence
                    abs_seconds = base_seconds + offset
                    h = (abs_seconds // 3600) % 24
                    m = (abs_seconds % 3600) // 60
                    s = abs_seconds % 60
                    lines.append(
                        json.dumps(
                            {
                                "start": f"{h:02d}:{m:02d}:{s:02d}",
                                "text": f"This is sentence {i + 1}.",
                            }
                        )
                    )
                jsonl_path.write_text("\n".join(lines) + "\n")

                # Create NPZ embeddings
                npz_path = segment_dir / f"{source}.npz"
                embeddings = np.random.randn(num_sentences, 256).astype(np.float32)
                # Normalize each embedding
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                statement_ids = np.arange(1, num_sentences + 1, dtype=np.int32)
                np.savez_compressed(
                    npz_path, embeddings=embeddings, statement_ids=statement_ids
                )

                # Create dummy audio file
                audio_path = segment_dir / f"{source}.flac"
                audio_path.write_bytes(b"")  # Empty placeholder

            return segment_dir

        def create_embedding(self, vector: list[float] | None = None) -> np.ndarray:
            """Create a normalized 256-dim embedding."""
            if vector is None:
                emb = np.random.randn(256).astype(np.float32)
            else:
                emb = np.array(vector + [0.0] * (256 - len(vector)), dtype=np.float32)
            return emb / np.linalg.norm(emb)

        def create_entity(
            self,
            name: str,
            voiceprints: list[tuple[str, str, str, int]] | None = None,
        ) -> Path:
            """Create a journal-level entity with optional voiceprint files.

            Args:
                name: Entity name
                voiceprints: Optional list of (day, segment_key, source, sentence_id)
                            tuples for voiceprints
            """
            # Create journal-level entity
            entity_id = entity_slug(name)
            journal_entity_dir = self.journal / "entities" / entity_id
            journal_entity_dir.mkdir(parents=True, exist_ok=True)
            journal_entity = {
                "id": entity_id,
                "name": name,
                "type": "Person",
                "created_at": 1700000000000,
            }
            with open(journal_entity_dir / "entity.json", "w", encoding="utf-8") as f:
                json.dump(journal_entity, f)

            # Create voiceprints.npz at journal level if specified
            if voiceprints:
                all_embeddings = []
                all_metadata = []
                for day, segment_key, source, sentence_id in voiceprints:
                    emb = self.create_embedding()
                    all_embeddings.append(emb)
                    metadata = {
                        "day": day,
                        "segment_key": segment_key,
                        "source": source,
                        "sentence_id": sentence_id,
                        "added_at": 1700000000000,
                    }
                    all_metadata.append(json.dumps(metadata))

                np.savez_compressed(
                    journal_entity_dir / "voiceprints.npz",
                    embeddings=np.array(all_embeddings, dtype=np.float32),
                    metadata=np.array(all_metadata, dtype=str),
                )

            return journal_entity_dir

        def create_speakers_json(
            self, day: str, segment_key: str, speakers: list[str]
        ) -> Path:
            """Create a speakers.json file in a segment directory.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                speakers: List of speaker names
            """
            segment_dir = self.journal / day / segment_key
            segment_dir.mkdir(parents=True, exist_ok=True)

            speakers_path = segment_dir / "speakers.json"
            with open(speakers_path, "w", encoding="utf-8") as f:
                json.dump(speakers, f)

            return speakers_path

    def _create():
        return SpeakersEnv(tmp_path)

    return _create
