# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Voiceprint bootstrap and name resolution for speaker attribution.

Bootstrap: Scans the journal for segments where exactly one non-owner
speaker is listed in speakers.json. In those segments, all non-owner
sentence embeddings belong to that speaker. Uses the owner centroid to
subtract the owner's sentences, then saves the remaining embeddings as
that speaker's voiceprint.

Import seeding: Scans import-stream segments for conversation transcripts
with per-line speaker attribution, maps speakers to journal entities, and
saves corresponding embeddings as voiceprints.

Name resolution: Compares voiceprint centroids between name variants.
Pairs with cosine similarity > 0.90 are the same person. Unambiguous
variants are auto-merged by adding the short name as an aka on the
canonical entity.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np

from apps.speakers.owner import load_owner_centroid
from think.entities import entity_slug, find_matching_entity
from think.entities.journal import (
    ensure_journal_entity_memory,
    get_or_create_journal_entity,
    load_all_journal_entities,
    load_journal_entity,
    save_journal_entity,
)
from think.utils import day_dirs, iter_segments, now_ms, segment_path

logger = logging.getLogger(__name__)

# Cosine similarity threshold for name variant merging (validated in Experiment B)
NAME_MERGE_THRESHOLD = 0.90


def _routes_helpers():
    """Load speakers route helpers lazily to avoid import cycles."""
    from apps.speakers.routes import (
        _load_embeddings_file,
        _load_entity_voiceprints_file,
        _normalize_embedding,
        _scan_segment_embeddings,
    )

    return (
        _load_embeddings_file,
        _normalize_embedding,
        _scan_segment_embeddings,
        _load_entity_voiceprints_file,
    )


def _load_existing_voiceprint_keys(entity_id: str) -> set[tuple]:
    """Load already-saved voiceprint keys for idempotency.

    Returns set of (day, segment_key, source, sentence_id) tuples.
    """
    _, _, _, load_entity_voiceprints_file = _routes_helpers()

    result = load_entity_voiceprints_file(entity_id)
    if result is None:
        return set()

    _, metadata_list = result
    return {
        (m.get("day"), m.get("segment_key"), m.get("source"), m.get("sentence_id"))
        for m in metadata_list
    }


def _save_voiceprints_batch(
    entity_id: str,
    new_items: list[tuple[np.ndarray, dict]],
) -> int:
    """Save multiple voiceprints to an entity's voiceprints.npz in one write.

    Args:
        entity_id: Entity ID (slug)
        new_items: List of (normalized_embedding, metadata_dict) tuples

    Returns:
        Number of embeddings saved
    """
    if not new_items:
        return 0

    folder = ensure_journal_entity_memory(entity_id)
    npz_path = folder / "voiceprints.npz"

    # Load existing voiceprints
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            existing_emb = data["embeddings"]
            existing_meta = list(data["metadata"])
        except Exception:
            existing_emb = np.empty((0, 256), dtype=np.float32)
            existing_meta = []
    else:
        existing_emb = np.empty((0, 256), dtype=np.float32)
        existing_meta = []

    # Build new arrays
    new_emb = np.vstack([emb.reshape(1, -1).astype(np.float32) for emb, _ in new_items])
    new_meta = [json.dumps(m) for _, m in new_items]

    combined_emb = (
        np.vstack([existing_emb, new_emb]) if len(existing_emb) > 0 else new_emb
    )
    combined_meta = np.array(existing_meta + new_meta, dtype=str)

    np.savez_compressed(npz_path, embeddings=combined_emb, metadata=combined_meta)
    return len(new_items)


def bootstrap_voiceprints(dry_run: bool = False) -> dict[str, Any]:
    """Bootstrap voiceprints from 1-listed-speaker segments across the full journal.

    For each segment where speakers.json lists exactly one name, all
    non-owner sentence embeddings belong to that speaker. The owner
    centroid is used to classify sentences — embeddings with cosine
    similarity >= OWNER_THRESHOLD to the owner centroid are excluded
    (contamination guard). Remaining embeddings are saved as voiceprints
    for the matched or newly created entity.

    The operation is idempotent: existing voiceprint entries are checked
    by (day, segment_key, source, sentence_id) key and skipped if already
    present.

    Args:
        dry_run: If True, report what would be saved without saving

    Returns:
        Dict with statistics about the bootstrap run
    """
    (
        load_embeddings_file,
        normalize_embedding,
        scan_segment_embeddings,
        _,
    ) = _routes_helpers()

    # Load owner centroid — required for owner subtraction
    centroid_data = load_owner_centroid()
    if centroid_data is None:
        return {"error": "No confirmed owner centroid. Run owner detection first."}

    owner_centroid, owner_threshold = centroid_data

    # Load all journal entities for speaker name matching
    journal_entities = load_all_journal_entities()
    entities_list = [e for e in journal_entities.values() if not e.get("blocked")]

    stats: dict[str, Any] = {
        "segments_scanned": 0,
        "single_speaker_segments": 0,
        "speakers_found": {},
        "entities_created": 0,
        "embeddings_saved": 0,
        "embeddings_skipped_owner": 0,
        "embeddings_skipped_duplicate": 0,
        "errors": [],
    }

    # Collect embeddings per entity for efficient batch saves
    entity_embeddings: dict[str, list[tuple[np.ndarray, dict]]] = defaultdict(list)
    entity_existing: dict[str, set] = {}
    entity_names: dict[str, str] = {}

    days = sorted(day_dirs().keys())

    for day_idx, day in enumerate(days):
        segments = scan_segment_embeddings(day)

        for segment in segments:
            stats["segments_scanned"] += 1
            stream = segment["stream"]
            seg_key = segment["key"]
            speakers = segment["speakers"]

            # Only process segments with exactly 1 listed speaker
            if len(speakers) != 1:
                continue

            stats["single_speaker_segments"] += 1
            speaker_name = speakers[0]

            # Match speaker name to an existing journal entity
            entity = find_matching_entity(speaker_name, entities_list)
            if entity:
                entity_id = entity["id"]
                entity_name = entity.get("name", speaker_name)
            else:
                # Create a new entity for this speaker
                entity_id = entity_slug(speaker_name)
                if not dry_run:
                    entity = get_or_create_journal_entity(
                        entity_id=entity_id,
                        name=speaker_name,
                        entity_type="Person",
                    )
                    # Add to list so subsequent segments find this entity
                    entities_list.append(entity)
                    stats["entities_created"] += 1
                entity_name = speaker_name

            entity_names[entity_id] = entity_name
            stats["speakers_found"].setdefault(entity_name, 0)

            # Load existing voiceprint keys for idempotency (once per entity)
            if entity_id not in entity_existing:
                entity_existing[entity_id] = _load_existing_voiceprint_keys(entity_id)

            existing_keys = entity_existing[entity_id]
            seg_dir = segment_path(day, seg_key, stream)

            for source in segment["sources"]:
                emb_data = load_embeddings_file(seg_dir / f"{source}.npz")
                if emb_data is None:
                    continue

                embeddings, statement_ids = emb_data

                for embedding, sid in zip(embeddings, statement_ids):
                    sentence_id = int(sid)
                    vp_key = (day, seg_key, source, sentence_id)

                    # Idempotency: skip if already saved
                    if vp_key in existing_keys:
                        stats["embeddings_skipped_duplicate"] += 1
                        continue

                    normalized = normalize_embedding(embedding)
                    if normalized is None:
                        continue

                    # Contamination guard: reject embeddings too similar to owner
                    owner_score = float(np.dot(normalized, owner_centroid))
                    if owner_score >= owner_threshold:
                        stats["embeddings_skipped_owner"] += 1
                        continue

                    # Non-owner embedding — belongs to the listed speaker
                    metadata = {
                        "day": day,
                        "segment_key": seg_key,
                        "source": source,
                        "stream": stream,
                        "sentence_id": sentence_id,
                        "added_at": now_ms(),
                    }

                    entity_embeddings[entity_id].append((normalized, metadata))
                    existing_keys.add(vp_key)
                    stats["speakers_found"][entity_name] += 1

        if (day_idx + 1) % 10 == 0:
            logger.info(
                "Bootstrap progress: %d/%d days, %d segments, %d embeddings queued",
                day_idx + 1,
                len(days),
                stats["segments_scanned"],
                sum(len(v) for v in entity_embeddings.values()),
            )

    # Batch save all collected embeddings
    if not dry_run:
        for entity_id, emb_list in entity_embeddings.items():
            try:
                saved = _save_voiceprints_batch(entity_id, emb_list)
                stats["embeddings_saved"] += saved
            except Exception as e:
                name = entity_names.get(entity_id, entity_id)
                stats["errors"].append(f"Failed to save for {name}: {e}")
                logger.exception("Failed to save voiceprints for %s", entity_id)
    else:
        stats["embeddings_saved"] = sum(len(v) for v in entity_embeddings.values())

    return stats


# Generic speaker names to skip (AI conversation imports)
_GENERIC_SPEAKERS = frozenset({"Human", "Assistant", "human", "assistant", ""})


def seed_from_imports(dry_run: bool = False) -> dict[str, Any]:
    """Seed voiceprints from speaker-attributed import transcripts.

    Scans import-stream segments for conversation_transcript.jsonl files
    with per-line speaker attribution. Maps speaker names to journal
    entities and saves corresponding embeddings as voiceprints.

    Unlike bootstrap_voiceprints(), this does NOT create new entities —
    unmatched speaker names are skipped.

    Args:
        dry_run: If True, report what would be saved without saving

    Returns:
        Dict with statistics about the seed run
    """
    (
        load_embeddings_file,
        normalize_embedding,
        _,
        _,
    ) = _routes_helpers()

    # Load owner centroid — required for owner subtraction
    centroid_data = load_owner_centroid()
    if centroid_data is None:
        return {"error": "No confirmed owner centroid. Run owner detection first."}

    owner_centroid, owner_threshold = centroid_data

    # Load all journal entities for speaker name matching
    journal_entities = load_all_journal_entities()
    entities_list = [e for e in journal_entities.values() if not e.get("blocked")]

    stats: dict[str, Any] = {
        "segments_scanned": 0,
        "segments_with_speakers": 0,
        "speakers_found": {},
        "embeddings_saved": 0,
        "embeddings_skipped_owner": 0,
        "embeddings_skipped_duplicate": 0,
        "errors": [],
    }

    # Collect embeddings per entity for efficient batch saves
    entity_embeddings: dict[str, list[tuple[np.ndarray, dict]]] = defaultdict(list)
    entity_existing: dict[str, set] = {}
    entity_names: dict[str, str] = {}

    days = sorted(day_dirs().keys())

    for day in days:
        for stream, seg_key, seg_dir in iter_segments(day):
            # Only process import streams
            if not stream.startswith("import."):
                continue

            stats["segments_scanned"] += 1

            # Read conversation_transcript.jsonl
            jsonl_path = seg_dir / "conversation_transcript.jsonl"
            if not jsonl_path.exists():
                continue

            try:
                lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            except OSError as e:
                stats["errors"].append(f"Failed to read {jsonl_path}: {e}")
                continue

            if len(lines) < 2:
                continue

            # Build sentence_id -> speaker mapping
            # Line 0 is metadata header. Lines 1+ are entries.
            # sentence_id is 1-based, matching statement_ids in NPZ.
            sid_to_speaker: dict[int, str] = {}
            has_real_speakers = False
            for line_idx in range(1, len(lines)):
                try:
                    entry = json.loads(lines[line_idx])
                except (json.JSONDecodeError, IndexError):
                    continue
                speaker = entry.get("speaker", "")
                if speaker and speaker not in _GENERIC_SPEAKERS:
                    sid_to_speaker[line_idx] = speaker
                    has_real_speakers = True

            if not has_real_speakers:
                continue

            stats["segments_with_speakers"] += 1

            # Find audio embedding NPZ files in this segment
            # Accept both "<source>_audio" pattern and plain "audio"
            npz_files = list(seg_dir.glob("*.npz"))
            sources = [
                f.stem
                for f in npz_files
                if f.stem.endswith("_audio") or f.stem == "audio"
            ]
            if not sources:
                continue

            for source in sources:
                emb_data = load_embeddings_file(seg_dir / f"{source}.npz")
                if emb_data is None:
                    continue

                embeddings, statement_ids = emb_data

                for embedding, sid in zip(embeddings, statement_ids):
                    sentence_id = int(sid)
                    speaker_name = sid_to_speaker.get(sentence_id)
                    if not speaker_name:
                        continue

                    # Match speaker to entity — skip if no match
                    entity = find_matching_entity(speaker_name, entities_list)
                    if not entity:
                        continue

                    entity_id = entity["id"]
                    entity_name = entity.get("name", speaker_name)
                    entity_names[entity_id] = entity_name
                    stats["speakers_found"].setdefault(entity_name, 0)

                    # Load existing voiceprint keys for idempotency (once per entity)
                    if entity_id not in entity_existing:
                        entity_existing[entity_id] = _load_existing_voiceprint_keys(
                            entity_id
                        )

                    existing_keys = entity_existing[entity_id]
                    vp_key = (day, seg_key, source, sentence_id)

                    # Idempotency: skip if already saved
                    if vp_key in existing_keys:
                        stats["embeddings_skipped_duplicate"] += 1
                        continue

                    normalized = normalize_embedding(embedding)
                    if normalized is None:
                        continue

                    # Contamination guard: reject embeddings too similar to owner
                    owner_score = float(np.dot(normalized, owner_centroid))
                    if owner_score >= owner_threshold:
                        stats["embeddings_skipped_owner"] += 1
                        continue

                    metadata = {
                        "day": day,
                        "segment_key": seg_key,
                        "source": source,
                        "stream": stream,
                        "sentence_id": sentence_id,
                        "added_at": now_ms(),
                    }

                    entity_embeddings[entity_id].append((normalized, metadata))
                    existing_keys.add(vp_key)
                    stats["speakers_found"][entity_name] += 1

    # Batch save all collected embeddings
    if not dry_run:
        for entity_id, emb_list in entity_embeddings.items():
            try:
                saved = _save_voiceprints_batch(entity_id, emb_list)
                stats["embeddings_saved"] += saved
            except Exception as e:
                name = entity_names.get(entity_id, entity_id)
                stats["errors"].append(f"Failed to save for {name}: {e}")
                logger.exception("Failed to save voiceprints for %s", entity_id)
    else:
        stats["embeddings_saved"] = sum(len(v) for v in entity_embeddings.values())

    return stats


def merge_names(alias_name: str, canonical_name: str) -> dict[str, Any]:
    """Merge a speaker name variant into a canonical entity.

    Finds both entities by name, adds the alias name as an aka on the
    canonical entity, and merges voiceprint embeddings with deduplication.

    Args:
        alias_name: The alias/variant name to merge from
        canonical_name: The canonical/full name to merge into

    Returns:
        Dict with merge statistics or error
    """
    _, normalize_embedding, _, load_entity_voiceprints_file = _routes_helpers()

    journal_entities = load_all_journal_entities()
    entities_list = [e for e in journal_entities.values() if not e.get("blocked")]

    alias_entity = find_matching_entity(alias_name, entities_list)
    if not alias_entity:
        return {"error": f"No entity found for alias: {alias_name}"}

    canonical_entity = find_matching_entity(canonical_name, entities_list)
    if not canonical_entity:
        return {"error": f"No entity found for canonical: {canonical_name}"}

    alias_id = alias_entity["id"]
    canonical_id = canonical_entity["id"]

    if alias_id == canonical_id:
        return {"error": "Alias and canonical resolve to the same entity."}

    # Add alias name as aka on canonical entity
    canonical = load_journal_entity(canonical_id)
    if not canonical:
        return {"error": f"Failed to load canonical entity: {canonical_id}"}

    alias_display = alias_entity.get("name", alias_name)
    existing_aka = set(canonical.get("aka", []))
    if alias_display not in existing_aka:
        existing_aka.add(alias_display)
        canonical["aka"] = sorted(existing_aka)
        canonical["updated_at"] = now_ms()
        save_journal_entity(canonical)

    # Merge voiceprint embeddings
    alias_vp = load_entity_voiceprints_file(alias_id)
    embeddings_merged = 0

    if alias_vp is not None:
        alias_embeddings, alias_metadata = alias_vp
        existing_keys = _load_existing_voiceprint_keys(canonical_id)

        new_items: list[tuple[np.ndarray, dict]] = []
        for emb, meta in zip(alias_embeddings, alias_metadata):
            key = (
                meta.get("day"),
                meta.get("segment_key"),
                meta.get("source"),
                meta.get("sentence_id"),
            )
            if key in existing_keys:
                continue
            normalized = normalize_embedding(emb)
            if normalized is not None:
                new_items.append((normalized, meta))
                existing_keys.add(key)

        if new_items:
            embeddings_merged = _save_voiceprints_batch(canonical_id, new_items)

    canonical_vp = load_entity_voiceprints_file(canonical_id)
    total = len(canonical_vp[0]) if canonical_vp else 0

    return {
        "merged": True,
        "alias": alias_display,
        "canonical_entity_id": canonical_id,
        "canonical_name": canonical.get("name", canonical_name),
        "embeddings_merged": embeddings_merged,
        "total_embeddings": total,
    }


def resolve_name_variants(dry_run: bool = False) -> dict[str, Any]:
    """Find and merge speaker name variants using voiceprint similarity.

    Computes a centroid for each entity's voiceprints and compares all
    pairs. Pairs with cosine similarity > NAME_MERGE_THRESHOLD (0.90)
    are flagged as the same person.

    Auto-merge criteria (all must be true):
    - Both entities have exactly one high-similarity match (mutual exclusivity)
    - The shorter name is the first word of the longer name (name variant pattern)

    When auto-merging, the short name is added as an aka on the canonical
    (longer-name) entity. Ambiguous cases are logged but not applied.

    Args:
        dry_run: If True, report merges without applying them

    Returns:
        Dict with merge statistics
    """
    _, normalize_embedding, _, load_entity_voiceprints_file = _routes_helpers()

    journal_entities = load_all_journal_entities()

    stats: dict[str, Any] = {
        "entities_with_voiceprints": 0,
        "pairs_compared": 0,
        "matches_found": [],
        "auto_merged": [],
        "ambiguous": [],
        "errors": [],
    }

    # Compute centroid for each non-principal entity with voiceprints
    centroids: dict[str, tuple[np.ndarray, str]] = {}  # entity_id -> (centroid, name)

    for entity_id, entity in journal_entities.items():
        if entity.get("blocked") or entity.get("is_principal"):
            continue

        name = entity.get("name", "")
        if not name:
            continue

        result = load_entity_voiceprints_file(entity_id)
        if result is None:
            continue

        embeddings, _ = result
        if len(embeddings) == 0:
            continue

        normalized_list = []
        for emb in embeddings:
            n = normalize_embedding(emb)
            if n is not None:
                normalized_list.append(n)

        if not normalized_list:
            continue

        centroid = normalize_embedding(np.mean(normalized_list, axis=0))
        if centroid is None:
            continue

        centroids[entity_id] = (centroid, name)
        stats["entities_with_voiceprints"] += 1

    # Compare all pairs and build match graph
    ids = list(centroids.keys())
    match_graph: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            stats["pairs_compared"] += 1
            id_a, id_b = ids[i], ids[j]
            cent_a, name_a = centroids[id_a]
            cent_b, name_b = centroids[id_b]

            similarity = float(np.dot(cent_a, cent_b))

            if similarity >= NAME_MERGE_THRESHOLD:
                stats["matches_found"].append(
                    {
                        "name_a": name_a,
                        "name_b": name_b,
                        "similarity": round(similarity, 4),
                    }
                )
                match_graph[id_a].append((id_b, similarity))
                match_graph[id_b].append((id_a, similarity))

    # Process matches — determine auto-merges vs ambiguous
    processed: set[tuple[str, str]] = set()

    for eid, matches in match_graph.items():
        if len(matches) > 1:
            # Multiple high-similarity matches — ambiguous
            _, name = centroids[eid]
            # Deduplicate: only report once per entity
            if any(tuple(sorted([eid, m[0]])) in processed for m in matches):
                continue
            for m in matches:
                processed.add(tuple(sorted([eid, m[0]])))

            stats["ambiguous"].append(
                {
                    "name": name,
                    "candidates": [
                        {"name": centroids[m[0]][1], "similarity": round(m[1], 4)}
                        for m in matches
                    ],
                }
            )
            continue

        # Single match — candidate for auto-merge
        other_id, similarity = matches[0]
        pair = tuple(sorted([eid, other_id]))
        if pair in processed:
            continue
        processed.add(pair)

        # Both sides must have exactly one match (mutual exclusivity)
        if len(match_graph.get(other_id, [])) != 1:
            continue  # Other side has multiple matches, handled above

        _, name_a = centroids[eid]
        _, name_b = centroids[other_id]

        # Determine canonical (longer name) vs alias (shorter name)
        if len(name_a) >= len(name_b):
            canonical_name, alias_name = name_a, name_b
            canonical_id = eid
        else:
            canonical_name, alias_name = name_b, name_a
            canonical_id = other_id

        # Check name variant pattern: alias is the first word of canonical
        canonical_first = canonical_name.split()[0].lower()
        alias_lower = alias_name.strip().lower()

        if canonical_first != alias_lower:
            stats["ambiguous"].append(
                {
                    "name": name_a,
                    "candidates": [
                        {"name": name_b, "similarity": round(similarity, 4)}
                    ],
                }
            )
            continue

        # Auto-merge: add alias as aka on canonical entity
        if not dry_run:
            try:
                je = load_journal_entity(canonical_id)
                if je:
                    existing_aka = set(je.get("aka", []))
                    if alias_name not in existing_aka:
                        existing_aka.add(alias_name)
                        je["aka"] = sorted(existing_aka)
                        je["updated_at"] = now_ms()
                        save_journal_entity(je)
            except Exception as e:
                stats["errors"].append(
                    f"Failed to merge {alias_name} -> {canonical_name}: {e}"
                )
                continue

        stats["auto_merged"].append(
            {
                "canonical": canonical_name,
                "alias": alias_name,
                "similarity": round(similarity, 4),
            }
        )

    return stats
