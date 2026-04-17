# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for speakers app tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from think.entities import entity_slug

# Default stream name for test fixtures
STREAM = "test"


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
            # Now _SOLSTONE_JOURNAL_OVERRIDE is set and data exists
    """

    class SpeakersEnv:
        def __init__(self, journal_path: Path):
            self.journal = journal_path
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_path))

        def create_segment(
            self,
            day: str,
            segment_key: str,
            sources: list[str],
            num_sentences: int = 5,
            *,
            stream: str | None = None,
            embeddings: np.ndarray | None = None,
        ) -> Path:
            """Create a segment with sentence embeddings.

            Creates both JSONL transcripts and NPZ embedding files.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                sources: List of audio sources (e.g., ["mic_audio", "sys_audio"])
                num_sentences: Number of sentences to create
            """
            segment_dir = self.journal / day / (stream or STREAM) / segment_key
            segment_dir.mkdir(parents=True, exist_ok=True)

            sentence_count = (
                embeddings.shape[0] if embeddings is not None else num_sentences
            )

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

                for i in range(sentence_count):
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
                if embeddings is None:
                    source_embeddings = np.random.randn(sentence_count, 256).astype(
                        np.float32
                    )
                    norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
                    source_embeddings = source_embeddings / norms
                else:
                    source_embeddings = embeddings.astype(np.float32)
                statement_ids = np.arange(1, sentence_count + 1, dtype=np.int32)
                np.savez_compressed(
                    npz_path,
                    embeddings=source_embeddings,
                    statement_ids=statement_ids,
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
            is_principal: bool = False,
        ) -> Path:
            """Create a journal-level entity with optional voiceprint files.

            Args:
                name: Entity name
                voiceprints: Optional list of (day, segment_key, source, sentence_id)
                            tuples for voiceprints
                is_principal: If True, mark this entity as the principal (self)
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
            if is_principal:
                journal_entity["is_principal"] = True
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
            agents_dir = self.journal / day / STREAM / segment_key / "talents"
            agents_dir.mkdir(parents=True, exist_ok=True)

            speakers_path = agents_dir / "speakers.json"
            with open(speakers_path, "w", encoding="utf-8") as f:
                json.dump(speakers, f)

            return speakers_path

        def create_speaker_labels(
            self,
            day: str,
            segment_key: str,
            labels: list[dict],
            metadata: dict | None = None,
        ) -> Path:
            """Create a speaker_labels.json file in a segment directory.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                labels: List of label dicts with sentence_id, speaker, confidence,
                    method
                metadata: Optional extra metadata (owner_centroid_version,
                    voiceprint_versions)
            """
            agents_dir = self.journal / day / STREAM / segment_key / "talents"
            agents_dir.mkdir(parents=True, exist_ok=True)

            data = {"labels": labels}
            if metadata:
                data.update(metadata)
            else:
                data["owner_centroid_version"] = None
                data["voiceprint_versions"] = {}

            labels_path = agents_dir / "speaker_labels.json"
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            return labels_path

        def create_speaker_corrections(
            self,
            day: str,
            segment_key: str,
            corrections: list[dict],
            *,
            stream: str | None = None,
        ) -> Path:
            """Create a speaker_corrections.json file in a segment directory.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                corrections: List of correction dicts with sentence_id,
                    original_speaker, corrected_speaker, timestamp
                stream: Optional stream name (defaults to STREAM)
            """
            agents_dir = (
                self.journal / day / (stream or STREAM) / segment_key / "talents"
            )
            agents_dir.mkdir(parents=True, exist_ok=True)

            data = {"corrections": corrections}
            corrections_path = agents_dir / "speaker_corrections.json"
            with open(corrections_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            return corrections_path

        def create_facet_relationship(
            self,
            facet: str,
            entity_id: str,
            *,
            description: str = "",
            attached_at: int = 1700000000000,
            updated_at: int | None = None,
            last_seen: str | None = None,
            observations: list[str] | None = None,
        ) -> Path:
            """Create a facet relationship for an entity.

            Args:
                facet: Facet name (e.g., "work", "personal")
                entity_id: Entity ID (slug)
                description: Relationship description
                attached_at: When the relationship was created
                updated_at: Last update timestamp
                last_seen: Last seen day string (YYYYMMDD)
                observations: Optional list of observation strings
            """
            rel_dir = self.journal / "facets" / facet / "entities" / entity_id
            rel_dir.mkdir(parents=True, exist_ok=True)

            relationship: dict = {
                "entity_id": entity_id,
                "attached_at": attached_at,
            }
            if description:
                relationship["description"] = description
            if updated_at is not None:
                relationship["updated_at"] = updated_at
            if last_seen is not None:
                relationship["last_seen"] = last_seen

            with open(rel_dir / "entity.json", "w", encoding="utf-8") as f:
                json.dump(relationship, f, indent=2)

            if observations:
                with open(rel_dir / "observations.jsonl", "w", encoding="utf-8") as f:
                    for obs in observations:
                        f.write(
                            json.dumps({"content": obs, "observed_at": 1700000000000})
                            + "\n"
                        )

            return rel_dir

        def create_import_segment(
            self,
            day: str,
            segment_key: str,
            speakers: list[tuple[str, str]],
            *,
            stream: str = "import.granola",
            embeddings: np.ndarray | None = None,
        ) -> Path:
            """Create an import segment with conversation_transcript and embeddings.

            Creates both a conversation_transcript.jsonl (with speaker labels) and
            imported_audio.{jsonl,npz,flac} (with aligned embeddings) in the
            same segment directory.

            Args:
                day: Day string (YYYYMMDD)
                segment_key: Segment key (HHMMSS_LEN)
                speakers: List of (speaker_name, text) tuples for each sentence
                stream: Import stream name (default: import.granola)
                embeddings: Optional pre-built embeddings array (num_sentences x 256)
            """
            segment_dir = self.journal / day / stream / segment_key
            segment_dir.mkdir(parents=True, exist_ok=True)

            num_sentences = len(speakers)

            time_part = segment_key.split("_")[0]
            base_h = int(time_part[0:2])
            base_m = int(time_part[2:4])
            base_s = int(time_part[4:6])
            base_seconds = base_h * 3600 + base_m * 60 + base_s

            ct_lines = [
                json.dumps({"imported": {"id": "test-import"}, "topics": "test"})
            ]
            for i, (speaker, text) in enumerate(speakers):
                offset = i * 5
                abs_seconds = base_seconds + offset
                h = (abs_seconds // 3600) % 24
                m = (abs_seconds % 3600) // 60
                s = abs_seconds % 60
                ct_lines.append(
                    json.dumps(
                        {
                            "start": f"{h:02d}:{m:02d}:{s:02d}",
                            "speaker": speaker,
                            "text": text,
                            "source": "import",
                        }
                    )
                )
            ct_path = segment_dir / "conversation_transcript.jsonl"
            ct_path.write_text("\n".join(ct_lines) + "\n")

            audio_lines = [
                json.dumps({"raw": "imported_audio.flac", "model": "medium.en"})
            ]
            for i, (_speaker, text) in enumerate(speakers):
                offset = i * 5
                abs_seconds = base_seconds + offset
                h = (abs_seconds // 3600) % 24
                m = (abs_seconds % 3600) // 60
                s = abs_seconds % 60
                audio_lines.append(
                    json.dumps(
                        {
                            "start": f"{h:02d}:{m:02d}:{s:02d}",
                            "text": text,
                        }
                    )
                )
            audio_jsonl_path = segment_dir / "imported_audio.jsonl"
            audio_jsonl_path.write_text("\n".join(audio_lines) + "\n")

            if embeddings is None:
                source_embeddings = np.random.randn(num_sentences, 256).astype(
                    np.float32
                )
                norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
                source_embeddings = source_embeddings / norms
            else:
                source_embeddings = embeddings.astype(np.float32)
            statement_ids = np.arange(1, num_sentences + 1, dtype=np.int32)
            np.savez_compressed(
                segment_dir / "imported_audio.npz",
                embeddings=source_embeddings,
                statement_ids=statement_ids,
            )

            (segment_dir / "imported_audio.flac").write_bytes(b"")

            return segment_dir

    def _create():
        return SpeakersEnv(tmp_path)

    return _create
