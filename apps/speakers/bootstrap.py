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

import bisect
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from apps.speakers.owner import load_owner_centroid
from think.entities import entity_slug, find_matching_entity, is_name_variant_match
from think.entities.journal import (
    ensure_journal_entity_memory,
    get_or_create_journal_entity,
    load_all_journal_entities,
    load_journal_entity,
    save_journal_entity,
)
from think.utils import day_dirs, get_journal, iter_segments, now_ms, segment_path

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
            # Use np.load with allow_pickle=False for safety, adjust if metadata requires it.
            with np.load(npz_path, allow_pickle=False) as data:
                existing_emb = data["embeddings"]
                # Existing metadata was likely saved as JSON strings. Deserialize them.
                # Assuming np.load returns an array of strings if saved as dtype=str.
                existing_meta_strings = data["metadata"]
                existing_meta_dicts = [json.loads(m) for m in existing_meta_strings]
        except (FileNotFoundError, ValueError, np.lib.npyio.NpzFile) as e:
            logger.warning(
                f"Failed to load existing voiceprints for {entity_id} from {npz_path}: {e}. Starting fresh."
            )
            existing_emb = np.empty((0, 256), dtype=np.float32)
            existing_meta_dicts = []
        except Exception as e:  # Catch other potential errors during loading
            logger.error(
                f"Unexpected error loading existing voiceprints for {entity_id} from {npz_path}: {e}"
            )
            raise
    else:
        existing_emb = np.empty((0, 256), dtype=np.float32)
        existing_meta_dicts = []

    # Prepare new embeddings and metadata dicts
    new_emb_list = []
    new_meta_dicts = []
    for emb, meta_dict in new_items:
        new_emb_list.append(emb.reshape(1, -1).astype(np.float32))
        new_meta_dicts.append(meta_dict)

    # Combine existing and new data
    if new_emb_list:
        new_emb_np = np.vstack(new_emb_list)
        combined_emb = (
            np.vstack([existing_emb, new_emb_np])
            if len(existing_emb) > 0
            else new_emb_np
        )
        # Combine the metadata dictionaries
        combined_meta_dicts = existing_meta_dicts + new_meta_dicts
    else:  # Should not happen if new_items is not empty, but for safety
        combined_emb = existing_emb
        combined_meta_dicts = existing_meta_dicts

    # Use the new safe saving utility
    try:
        # Import the utility function
        from apps.speakers.voiceprint_io import save_voiceprints_safely

        save_voiceprints_safely(
            npz_path=npz_path,
            embeddings=combined_emb,
            metadata=combined_meta_dicts,  # Pass metadata as a list of dicts
        )
        return len(new_items)
    except Exception as e:
        logger.error(f"Failed to safely save voiceprints for {entity_id}: {e}")
        # The save_voiceprints_safely function already logs critical errors and re-raises.
        # We re-raise here to propagate the failure.
        raise


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


def merge_names(alias_name: str, canonical_name: str) -> dict[str, Any]:
    """Deep merge a speaker entity into a canonical entity.

    Performs a phased deep merge: identity data, voiceprints, facet
    relationships, speaker references, then deletes the alias entity.
    Designed for interrupt safety — delete-last ordering ensures the
    system is never in an unrecoverable state. Every phase is idempotent.

    Args:
        alias_name: The alias/variant name to merge from
        canonical_name: The canonical/full name to merge into

    Returns:
        Dict with merge statistics or error
    """
    _, normalize_embedding, _, load_entity_voiceprints_file = _routes_helpers()

    journal_entities = load_all_journal_entities()
    entities_list = list(journal_entities.values())

    # --- Phase 0: Resolve and validate ---
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

    alias = load_journal_entity(alias_id)
    if not alias:
        return {"error": f"Failed to load alias entity: {alias_id}"}

    canonical = load_journal_entity(canonical_id)
    if not canonical:
        return {"error": f"Failed to load canonical entity: {canonical_id}"}

    if alias.get("is_principal") or canonical.get("is_principal"):
        return {"error": "Cannot merge the principal entity."}
    if alias.get("blocked"):
        return {"error": f"Cannot merge blocked entity: {alias_id}"}
    if canonical.get("blocked"):
        return {"error": f"Cannot merge blocked entity: {canonical_id}"}

    alias_display = alias.get("name", alias_name)

    # Set merged_into resume marker on alias
    alias["merged_into"] = canonical_id
    alias["updated_at"] = now_ms()
    save_journal_entity(alias)

    # --- Phase 1: Merge identity data ---
    akas_added: list[str] = []
    existing_aka = set(canonical.get("aka", []))
    canonical_name_val = canonical.get("name", "")

    # Add alias display name as aka
    if alias_display not in existing_aka and alias_display != canonical_name_val:
        existing_aka.add(alias_display)
        akas_added.append(alias_display)

    # Merge alias's akas
    for aka in alias.get("aka", []):
        if aka not in existing_aka and aka != canonical_name_val:
            existing_aka.add(aka)
            akas_added.append(aka)

    canonical["aka"] = sorted(existing_aka)

    # Merge emails
    canonical_emails = {e.lower() for e in canonical.get("emails", [])}
    for email in alias.get("emails", []):
        canonical_emails.add(email.lower())
    if canonical_emails:
        canonical["emails"] = sorted(canonical_emails)

    canonical["updated_at"] = now_ms()
    save_journal_entity(canonical)

    # --- Phase 2: Merge voiceprints ---
    alias_vp = load_entity_voiceprints_file(alias_id)
    voiceprints_merged = 0

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
            voiceprints_merged = _save_voiceprints_batch(canonical_id, new_items)

    canonical_vp = load_entity_voiceprints_file(canonical_id)
    voiceprints_total = len(canonical_vp[0]) if canonical_vp else 0

    # --- Phase 3: Merge facet relationships ---
    facets_merged: list[str] = []
    facets_moved: list[str] = []
    journal = get_journal()
    facets_dir = Path(journal) / "facets"

    if facets_dir.exists():
        for facet_entry in sorted(facets_dir.iterdir()):
            if not facet_entry.is_dir():
                continue
            facet_name = facet_entry.name
            alias_rel_dir = facet_entry / "entities" / alias_id
            alias_rel_path = alias_rel_dir / "entity.json"
            if not alias_rel_path.is_file():
                continue

            canonical_rel_dir = facet_entry / "entities" / canonical_id
            canonical_rel_path = canonical_rel_dir / "entity.json"

            if not canonical_rel_path.is_file():
                # Move: rename alias relationship dir to canonical
                if canonical_rel_dir.exists():
                    shutil.rmtree(canonical_rel_dir)
                alias_rel_dir.rename(canonical_rel_dir)
                # Update entity_id inside the moved entity.json
                moved_path = canonical_rel_dir / "entity.json"
                try:
                    with open(moved_path, encoding="utf-8") as f:
                        rel_data = json.load(f)
                    rel_data["entity_id"] = canonical_id
                    tmp = moved_path.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(rel_data, f, ensure_ascii=False, indent=2)
                        f.write("\n")
                    tmp.rename(moved_path)
                except (json.JSONDecodeError, OSError):
                    pass
                facets_moved.append(facet_name)
            else:
                # Both have relationships: merge timestamps and data
                try:
                    with open(alias_rel_path, encoding="utf-8") as f:
                        alias_rel = json.load(f)
                    with open(canonical_rel_path, encoding="utf-8") as f:
                        canonical_rel = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                # Merge timestamps: earliest attached_at
                alias_attached = alias_rel.get("attached_at")
                canonical_attached = canonical_rel.get("attached_at")
                if alias_attached and (
                    not canonical_attached or alias_attached < canonical_attached
                ):
                    canonical_rel["attached_at"] = alias_attached

                # Latest updated_at and last_seen
                for ts_field in ("updated_at", "last_seen"):
                    alias_ts = alias_rel.get(ts_field)
                    canonical_ts = canonical_rel.get(ts_field)
                    if alias_ts and (not canonical_ts or alias_ts > canonical_ts):
                        canonical_rel[ts_field] = alias_ts

                # Merge description: keep canonical's if non-empty
                if not canonical_rel.get("description") and alias_rel.get(
                    "description"
                ):
                    canonical_rel["description"] = alias_rel["description"]

                # Save merged relationship (atomic write)
                canonical_rel["entity_id"] = canonical_id
                content = json.dumps(canonical_rel, ensure_ascii=False, indent=2) + "\n"
                tmp = canonical_rel_path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(content)
                tmp.rename(canonical_rel_path)

                # Merge observations: append alias's to canonical's
                alias_obs_path = alias_rel_dir / "observations.jsonl"
                if alias_obs_path.exists():
                    alias_obs = alias_obs_path.read_text(encoding="utf-8")
                    if alias_obs.strip():
                        canonical_obs_path = canonical_rel_dir / "observations.jsonl"
                        existing_obs = ""
                        if canonical_obs_path.exists():
                            existing_obs = canonical_obs_path.read_text(
                                encoding="utf-8"
                            )
                        with open(canonical_obs_path, "a", encoding="utf-8") as f:
                            if existing_obs and not existing_obs.endswith("\n"):
                                f.write("\n")
                            f.write(alias_obs)
                            if not alias_obs.endswith("\n"):
                                f.write("\n")

                # Delete alias relationship directory
                shutil.rmtree(alias_rel_dir)
                facets_merged.append(facet_name)

    # --- Phase 4: Rewrite speaker references ---
    segments_scanned = 0
    labels_rewritten = 0
    corrections_rewritten = 0
    errors: list[str] = []
    alias_id_bytes = alias_id.encode("utf-8")

    for day in sorted(day_dirs().keys()):
        for _stream, _seg_key, seg_path in iter_segments(day):
            segments_scanned += 1
            agents_dir = seg_path / "agents"

            # Rewrite speaker_labels.json
            labels_path = agents_dir / "speaker_labels.json"
            if labels_path.is_file():
                try:
                    raw = labels_path.read_bytes()
                    if alias_id_bytes in raw:
                        data = json.loads(raw)
                        changed = False
                        for label in data.get("labels", []):
                            if label.get("speaker") == alias_id:
                                label["speaker"] = canonical_id
                                changed = True
                        if changed:
                            tmp = labels_path.with_suffix(".tmp")
                            with open(tmp, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2)
                            tmp.rename(labels_path)
                            labels_rewritten += 1
                except Exception as e:
                    errors.append(f"{labels_path}: {e}")

            # Rewrite speaker_corrections.json
            corrections_path = agents_dir / "speaker_corrections.json"
            if corrections_path.is_file():
                try:
                    raw = corrections_path.read_bytes()
                    if alias_id_bytes in raw:
                        data = json.loads(raw)
                        changed = False
                        for correction in data.get("corrections", []):
                            if correction.get("original_speaker") == alias_id:
                                correction["original_speaker"] = canonical_id
                                changed = True
                            if correction.get("corrected_speaker") == alias_id:
                                correction["corrected_speaker"] = canonical_id
                                changed = True
                        if changed:
                            tmp = corrections_path.with_suffix(".tmp")
                            with open(tmp, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2)
                            tmp.rename(corrections_path)
                            corrections_rewritten += 1
                except Exception as e:
                    errors.append(f"{corrections_path}: {e}")

    # --- Phase 5: Cleanup ---
    alias_entity_dir = Path(journal) / "entities" / alias_id
    if alias_entity_dir.exists():
        shutil.rmtree(alias_entity_dir)

    discovery_cache = Path(journal) / "awareness" / "discovery_clusters.json"
    if discovery_cache.exists():
        discovery_cache.unlink()

    return {
        "merged": True,
        "alias": alias_display,
        "alias_id": alias_id,
        "canonical_name": canonical.get("name", canonical_name),
        "canonical_id": canonical_id,
        "akas_added": akas_added,
        "voiceprints_merged": voiceprints_merged,
        "voiceprints_total": voiceprints_total,
        "facets_merged": facets_merged,
        "facets_moved": facets_moved,
        "segments_scanned": segments_scanned,
        "labels_rewritten": labels_rewritten,
        "corrections_rewritten": corrections_rewritten,
        "errors": errors,
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
        else:
            canonical_name, alias_name = name_b, name_a

        # Check name variant pattern: first-word, token-subset, or prefix-token
        if not is_name_variant_match(alias_name, canonical_name):
            stats["ambiguous"].append(
                {
                    "name": name_a,
                    "candidates": [
                        {"name": name_b, "similarity": round(similarity, 4)}
                    ],
                }
            )
            continue

        # Auto-merge: call deep merge
        if not dry_run:
            try:
                result = merge_names(alias_name, canonical_name)
                if result.get("error"):
                    stats["errors"].append(
                        f"Failed to merge {alias_name} -> {canonical_name}: "
                        f"{result['error']}"
                    )
                    continue
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


# Import streams that contain AI chat (no real speakers to seed)
_AI_CHAT_STREAMS = frozenset({"import.chatgpt", "import.claude", "import.gemini"})


def _time_str_to_seconds(time_str: str) -> int:
    """Parse HH:MM:SS to total seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _parse_conversation_speakers(seg_dir: Path) -> list[tuple[int, str]]:
    """Parse speaker assignments from conversation_transcript.jsonl.

    Returns sorted list of (start_seconds, speaker_name) tuples.
    Skips the metadata header line and entries with empty speaker fields.
    """
    ct_path = seg_dir / "conversation_transcript.jsonl"
    if not ct_path.exists():
        return []

    entries: list[tuple[int, str]] = []
    try:
        with open(ct_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []

    # Skip header (line 0), parse entry lines
    for line in lines[1:]:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        speaker = entry.get("speaker", "")
        start = entry.get("start", "")
        if not speaker or not start:
            continue
        try:
            seconds = _time_str_to_seconds(start)
        except (ValueError, IndexError):
            continue
        entries.append((seconds, speaker))

    entries.sort(key=lambda x: x[0])
    return entries


def _find_speaker_at_time(
    speaker_entries: list[tuple[int, str]], target_seconds: int
) -> str | None:
    """Find the speaker at a given time using binary search.

    Returns the speaker whose entry starts at or before target_seconds,
    or None if no entry precedes the target time.
    """
    if not speaker_entries:
        return None
    # bisect on just the time values
    times = [e[0] for e in speaker_entries]
    idx = bisect.bisect_right(times, target_seconds) - 1
    if idx < 0:
        return None
    return speaker_entries[idx][1]


def link_import(name: str, entity_id: str) -> dict[str, Any]:
    """Link an import participant name as an aka on an existing entity.

    Args:
        name: The participant name from an import transcript
        entity_id: The entity ID to link to

    Returns:
        Dict with link result or error
    """
    entity = load_journal_entity(entity_id)
    if not entity:
        return {"error": f"Entity not found: {entity_id}"}

    # Check if the name conflicts with another entity
    all_entities = load_all_journal_entities()
    others = [e for eid, e in all_entities.items() if eid != entity_id]
    conflict = find_matching_entity(name, others)
    if conflict:
        return {"error": f"Name '{name}' conflicts with entity '{conflict['id']}'"}

    existing_aka = set(entity.get("aka", []))
    already_present = name in existing_aka

    if not already_present:
        existing_aka.add(name)
        entity["aka"] = sorted(existing_aka)
        entity["updated_at"] = now_ms()
        save_journal_entity(entity)

    return {
        "linked": True,
        "entity_id": entity_id,
        "name_added": name,
        "already_present": already_present,
    }


def seed_from_imports(dry_run: bool = False) -> dict[str, Any]:
    """Seed voiceprints from import segments with speaker-attributed transcripts."""
    (
        load_embeddings_file,
        normalize_embedding,
        scan_segment_embeddings,
        _,
    ) = _routes_helpers()

    centroid_data = load_owner_centroid()
    if centroid_data is None:
        return {"error": "No confirmed owner centroid. Run owner detection first."}

    owner_centroid, owner_threshold = centroid_data

    journal_entities = load_all_journal_entities()
    entities_list = [e for e in journal_entities.values() if not e.get("blocked")]

    stats: dict[str, Any] = {
        "segments_scanned": 0,
        "segments_with_speakers": 0,
        "speakers_found": {},
        "embeddings_saved": 0,
        "embeddings_skipped_owner": 0,
        "embeddings_skipped_duplicate": 0,
        "speakers_unmatched": [],
        "errors": [],
    }

    entity_embeddings: dict[str, list[tuple[np.ndarray, dict]]] = defaultdict(list)
    entity_existing: dict[str, set] = {}
    unmatched_set: set[str] = set()
    speaker_entity_cache: dict[str, Any] = {}

    days = sorted(day_dirs().keys())

    for day_idx, day in enumerate(days):
        segments = scan_segment_embeddings(day)

        for segment in segments:
            stream = segment["stream"]
            if not stream.startswith("import.") or stream in _AI_CHAT_STREAMS:
                continue

            stats["segments_scanned"] += 1
            seg_key = segment["key"]
            seg_dir = segment_path(day, seg_key, stream)
            speaker_entries = _parse_conversation_speakers(seg_dir)
            if not speaker_entries:
                continue

            stats["segments_with_speakers"] += 1

            for source in segment["sources"]:
                source_jsonl = seg_dir / f"{source}.jsonl"
                stmt_times: dict[int, int] = {}
                if source_jsonl.exists():
                    try:
                        with open(source_jsonl, encoding="utf-8") as f:
                            src_lines = f.readlines()
                    except OSError as exc:
                        stats["errors"].append(
                            f"Failed to read source transcript {day}/{seg_key}/{source}: {exc}"
                        )
                        continue

                    for i, line in enumerate(src_lines[1:], start=1):
                        try:
                            entry = json.loads(line)
                            start = entry.get("start", "")
                            if start:
                                stmt_times[i] = _time_str_to_seconds(start)
                        except (json.JSONDecodeError, ValueError, IndexError):
                            continue

                emb_data = load_embeddings_file(seg_dir / f"{source}.npz")
                if emb_data is None:
                    continue

                embeddings, statement_ids = emb_data

                for embedding, sid in zip(embeddings, statement_ids):
                    sentence_id = int(sid)

                    target_time = stmt_times.get(sentence_id)
                    if target_time is None:
                        continue

                    speaker_name = _find_speaker_at_time(speaker_entries, target_time)
                    if not speaker_name:
                        continue

                    if speaker_name not in speaker_entity_cache:
                        entity = find_matching_entity(speaker_name, entities_list)
                        speaker_entity_cache[speaker_name] = entity

                    entity = speaker_entity_cache[speaker_name]
                    if entity is None:
                        if speaker_name not in unmatched_set:
                            unmatched_set.add(speaker_name)
                            stats["speakers_unmatched"].append(speaker_name)
                        continue

                    entity_id = entity["id"]
                    entity_name = entity.get("name", speaker_name)
                    stats["speakers_found"].setdefault(entity_name, 0)

                    if entity_id not in entity_existing:
                        entity_existing[entity_id] = _load_existing_voiceprint_keys(
                            entity_id
                        )

                    existing_keys = entity_existing[entity_id]
                    vp_key = (day, seg_key, source, sentence_id)

                    if vp_key in existing_keys:
                        stats["embeddings_skipped_duplicate"] += 1
                        continue

                    normalized = normalize_embedding(embedding)
                    if normalized is None:
                        continue

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

        if (day_idx + 1) % 10 == 0:
            logger.info(
                "Import seeding progress: %d/%d days, %d segments, %d embeddings queued",
                day_idx + 1,
                len(days),
                stats["segments_scanned"],
                sum(len(v) for v in entity_embeddings.values()),
            )

    if not dry_run:
        for entity_id, emb_list in entity_embeddings.items():
            try:
                saved = _save_voiceprints_batch(entity_id, emb_list)
                stats["embeddings_saved"] += saved
            except Exception as e:
                stats["errors"].append(f"Failed to save for {entity_id}: {e}")
                logger.exception(
                    "Failed to save import-seeded voiceprints for %s", entity_id
                )
    else:
        stats["embeddings_saved"] = sum(len(v) for v in entity_embeddings.values())

    return stats
