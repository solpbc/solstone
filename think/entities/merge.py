# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Journal-entity merge primitive."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from think.entities.journal import (
    clear_journal_entity_cache,
    load_journal_entity,
    save_journal_entity,
    scan_journal_entities,
)
from think.entities.loading import clear_entity_loading_cache
from think.entities.observations import (
    clear_observation_cache,
    clear_observation_count_cache,
    save_observations,
)
from think.entities.relationships import (
    clear_relationship_caches,
    save_facet_relationship,
)
from think.entities.voiceprints import (
    load_entity_voiceprints_file,
    load_existing_voiceprint_keys,
    normalize_embedding,
    save_voiceprints_batch,
)
from think.utils import day_dirs, get_journal, iter_segments, now_ms


def _dedupe_akas(target_values: list[Any], source_values: list[Any]) -> list[str]:
    """Case-insensitive aka dedup, preserving first-seen spelling."""
    aka_by_lower: dict[str, str] = {}
    for values in (target_values, source_values):
        if not isinstance(values, list):
            continue
        for value in values:
            if not value:
                continue
            key = str(value).lower()
            if key not in aka_by_lower:
                aka_by_lower[key] = str(value)
    return sorted(aka_by_lower.values(), key=str.lower)


def _dedupe_emails(target_values: list[Any], source_values: list[Any]) -> list[str]:
    """Case-insensitive email dedup, preserving first-seen order/spelling."""
    merged_emails: list[str] = []
    seen_emails: set[str] = set()
    for values in (target_values, source_values):
        if not isinstance(values, list):
            continue
        for value in values:
            if not value:
                continue
            email = str(value)
            key = email.lower()
            if key in seen_emails:
                continue
            seen_emails.add(key)
            merged_emails.append(email)
    return merged_emails


def _dedupe_observations(
    source_observations: list[dict[str, Any]],
    target_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate observations on (content, observed_at)."""
    seen = {
        (item.get("content", ""), item.get("observed_at"))
        for item in target_observations
    }
    merged_observations = list(target_observations)
    for item in source_observations:
        key = (item.get("content", ""), item.get("observed_at"))
        if key in seen:
            continue
        seen.add(key)
        merged_observations.append(item)
    return merged_observations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _identity_section(
    akas_added: list[str],
    emails_added: list[str],
    principal_transferred: bool,
) -> dict[str, Any]:
    return {
        "akas_added": akas_added,
        "akas_added_count": len(akas_added),
        "emails_added": emails_added,
        "emails_added_count": len(emails_added),
        "principal_transferred": principal_transferred,
    }


def _voiceprint_section(
    added: int, skipped_duplicate: int, target_total: int
) -> dict[str, Any]:
    return {
        "added": added,
        "skipped_duplicate": skipped_duplicate,
        "target_total": target_total,
    }


def _facet_section(
    moved: list[str],
    merged: list[str],
    observations_appended: int,
) -> dict[str, Any]:
    return {
        "moved": moved,
        "moved_count": len(moved),
        "merged": merged,
        "merged_count": len(merged),
        "observations_appended": observations_appended,
    }


def _segment_section(
    labels_rewritten: int,
    corrections_rewritten: int,
    files_scanned: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "labels_rewritten": labels_rewritten,
        "corrections_rewritten": corrections_rewritten,
        "files_scanned": files_scanned,
        "errors": errors,
    }


def _empty_result_section() -> dict[str, Any]:
    return {
        "identity": _identity_section([], [], False),
        "voiceprints": _voiceprint_section(0, 0, 0),
        "facets": _facet_section([], [], 0),
        "segments": _segment_section(0, 0, 0, []),
    }


def _is_missing_value(value: Any) -> bool:
    return value in (None, "", [], {})


def _plan_resume_marker(
    source_entity: dict[str, Any],
    target_id: str,
    *,
    principal_transferred: bool,
) -> dict[str, Any]:
    updated = dict(source_entity)
    updated["merged_into"] = target_id
    updated["updated_at"] = now_ms()
    if principal_transferred:
        updated.pop("is_principal", None)
    return updated


def _plan_identity_merge(
    source_entity: dict[str, Any],
    target_entity: dict[str, Any],
    *,
    keep_source_as_aka: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_after = dict(target_entity)
    source_display = str(source_entity.get("name", source_entity.get("id", "")))
    target_name = str(target_entity.get("name", ""))

    target_akas = target_entity.get("aka", [])
    if not isinstance(target_akas, list):
        target_akas = []

    source_aka_values = source_entity.get("aka", [])
    if not isinstance(source_aka_values, list):
        source_aka_values = []

    aka_candidates: list[str] = []
    if keep_source_as_aka and source_display and source_display != target_name:
        aka_candidates.append(source_display)
    aka_candidates.extend(str(value) for value in source_aka_values if value)

    target_aka_keys = {str(value).lower() for value in target_akas if value}
    added_akas: list[str] = []
    seen_added_akas: set[str] = set()
    for value in aka_candidates:
        key = value.lower()
        if key in target_aka_keys or key in seen_added_akas:
            continue
        seen_added_akas.add(key)
        added_akas.append(value)

    merged_akas = _dedupe_akas(target_akas, aka_candidates)
    if merged_akas:
        target_after["aka"] = merged_akas

    target_emails = target_entity.get("emails", [])
    if not isinstance(target_emails, list):
        target_emails = []
    source_emails = source_entity.get("emails", [])
    if not isinstance(source_emails, list):
        source_emails = []

    target_email_keys = {str(value).lower() for value in target_emails if value}
    added_emails: list[str] = []
    seen_added_emails: set[str] = set()
    for value in source_emails:
        if not value:
            continue
        email = str(value)
        key = email.lower()
        if key in target_email_keys or key in seen_added_emails:
            continue
        seen_added_emails.add(key)
        added_emails.append(email)

    merged_emails = _dedupe_emails(target_emails, source_emails)
    if merged_emails:
        target_after["emails"] = merged_emails

    principal_transferred = bool(
        source_entity.get("is_principal") and not target_entity.get("is_principal")
    )
    if principal_transferred:
        target_after["is_principal"] = True

    for key, value in source_entity.items():
        if key in {
            "id",
            "name",
            "aka",
            "emails",
            "created_at",
            "updated_at",
            "merged_into",
            "blocked",
            "is_principal",
        }:
            continue
        if _is_missing_value(target_after.get(key)) and not _is_missing_value(value):
            target_after[key] = value

    target_after["updated_at"] = now_ms()
    return target_after, _identity_section(
        added_akas, added_emails, principal_transferred
    )


def _plan_voiceprint_merge(source_id: str, target_id: str) -> dict[str, Any]:
    source_vp = load_entity_voiceprints_file(source_id)
    target_vp = load_entity_voiceprints_file(target_id)
    existing_keys = load_existing_voiceprint_keys(target_id)

    new_items: list[tuple[Any, dict[str, Any]]] = []
    skipped_duplicate = 0
    if source_vp is not None:
        source_embeddings, source_metadata = source_vp
        for emb, meta in zip(source_embeddings, source_metadata):
            key = (
                meta.get("day"),
                meta.get("segment_key"),
                meta.get("source"),
                meta.get("sentence_id"),
            )
            if key in existing_keys:
                skipped_duplicate += 1
                continue
            normalized = normalize_embedding(emb)
            if normalized is None:
                continue
            new_items.append((normalized, meta))
            existing_keys.add(key)

    target_existing_total = len(target_vp[0]) if target_vp else 0
    added = len(new_items)
    return {
        "items": new_items,
        "section": _voiceprint_section(
            added=added,
            skipped_duplicate=skipped_duplicate,
            target_total=target_existing_total + added,
        ),
    }


def _plan_facet_merge(source_id: str, target_id: str) -> dict[str, Any]:
    journal = Path(get_journal())
    facets_dir = journal / "facets"
    operations: list[dict[str, Any]] = []
    moved: list[str] = []
    merged: list[str] = []
    observations_appended = 0

    if not facets_dir.exists():
        return {
            "operations": operations,
            "section": _facet_section(moved, merged, observations_appended),
        }

    for facet_entry in sorted(facets_dir.iterdir()):
        if not facet_entry.is_dir():
            continue
        facet_name = facet_entry.name
        source_rel_dir = facet_entry / "entities" / source_id
        source_rel_path = source_rel_dir / "entity.json"
        if not source_rel_path.is_file():
            continue

        target_rel_dir = facet_entry / "entities" / target_id
        target_rel_path = target_rel_dir / "entity.json"

        if not target_rel_path.is_file():
            operations.append(
                {
                    "kind": "move",
                    "facet": facet_name,
                    "source_rel_dir": source_rel_dir,
                    "target_rel_dir": target_rel_dir,
                }
            )
            moved.append(facet_name)
            continue

        try:
            with open(source_rel_path, encoding="utf-8") as handle:
                source_rel = json.load(handle)
            with open(target_rel_path, encoding="utf-8") as handle:
                target_rel = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        merged_rel = dict(target_rel)
        source_attached = source_rel.get("attached_at")
        target_attached = merged_rel.get("attached_at")
        if source_attached and (
            not target_attached or source_attached < target_attached
        ):
            merged_rel["attached_at"] = source_attached

        for field in ("updated_at", "last_seen"):
            source_ts = source_rel.get(field)
            target_ts = merged_rel.get(field)
            if source_ts and (not target_ts or source_ts > target_ts):
                merged_rel[field] = source_ts

        if not merged_rel.get("description") and source_rel.get("description"):
            merged_rel["description"] = source_rel["description"]

        source_obs = _read_jsonl(source_rel_dir / "observations.jsonl")
        target_obs = _read_jsonl(target_rel_dir / "observations.jsonl")
        merged_obs = _dedupe_observations(source_obs, target_obs)
        observations_added = len(merged_obs) - len(target_obs)
        observations_appended += observations_added

        operations.append(
            {
                "kind": "merge",
                "facet": facet_name,
                "source_rel_dir": source_rel_dir,
                "target_rel_dir": target_rel_dir,
                "relationship": merged_rel,
                "observations": merged_obs,
                "observations_added": observations_added,
            }
        )
        merged.append(facet_name)

    return {
        "operations": operations,
        "section": _facet_section(moved, merged, observations_appended),
    }


def _plan_segment_rewrites(source_id: str, target_id: str) -> dict[str, Any]:
    labels_rewritten = 0
    corrections_rewritten = 0
    files_scanned = 0
    errors: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    source_id_bytes = source_id.encode("utf-8")

    for day_path in _segment_day_dirs():
        for _stream, _seg_key, seg_path in iter_segments(day_path):
            files_scanned += 1
            talents_dir = seg_path / "talents"

            labels_path = talents_dir / "speaker_labels.json"
            if labels_path.is_file():
                try:
                    raw = labels_path.read_bytes()
                    if source_id_bytes in raw:
                        data = json.loads(raw)
                        changed = False
                        for label in data.get("labels", []):
                            if label.get("speaker") == source_id:
                                label["speaker"] = target_id
                                changed = True
                        if changed:
                            labels_rewritten += 1
                            operations.append(
                                {
                                    "kind": "speaker_labels",
                                    "path": labels_path,
                                    "data": data,
                                }
                            )
                except Exception as exc:
                    errors.append(
                        {
                            "kind": "speaker_labels",
                            "path": str(labels_path),
                            "message": str(exc),
                        }
                    )

            corrections_path = talents_dir / "speaker_corrections.json"
            if corrections_path.is_file():
                try:
                    raw = corrections_path.read_bytes()
                    if source_id_bytes in raw:
                        data = json.loads(raw)
                        changed = False
                        for correction in data.get("corrections", []):
                            if correction.get("original_speaker") == source_id:
                                correction["original_speaker"] = target_id
                                changed = True
                            if correction.get("corrected_speaker") == source_id:
                                correction["corrected_speaker"] = target_id
                                changed = True
                        if changed:
                            corrections_rewritten += 1
                            operations.append(
                                {
                                    "kind": "speaker_corrections",
                                    "path": corrections_path,
                                    "data": data,
                                }
                            )
                except Exception as exc:
                    errors.append(
                        {
                            "kind": "speaker_corrections",
                            "path": str(corrections_path),
                            "message": str(exc),
                        }
                    )

    return {
        "operations": operations,
        "section": _segment_section(
            labels_rewritten=labels_rewritten,
            corrections_rewritten=corrections_rewritten,
            files_scanned=files_scanned,
            errors=errors,
        ),
    }


def _segment_day_dirs() -> list[Path]:
    chronicle_days = [Path(path) for _, path in sorted(day_dirs().items())]
    journal = Path(get_journal())
    flat_days = sorted(
        entry
        for entry in journal.iterdir()
        if entry.is_dir() and entry.name.isdigit() and len(entry.name) == 8
    )
    return chronicle_days or flat_days


def _check_aka_cross_references(
    source_id: str, source_display: str, target_id: str
) -> list[str]:
    offenders: list[str] = []
    for entity_id in scan_journal_entities():
        if entity_id in {source_id, target_id}:
            continue
        entity = load_journal_entity(entity_id)
        if not entity:
            continue
        aka_values = entity.get("aka", [])
        if not isinstance(aka_values, list):
            continue
        if source_id in aka_values or source_display in aka_values:
            offenders.append(entity_id)
    offenders.sort()
    return offenders


def _apply_facet_plan(operations: list[dict[str, Any]], target_id: str) -> None:
    for operation in operations:
        if operation["kind"] == "move":
            source_rel_dir = operation["source_rel_dir"]
            target_rel_dir = operation["target_rel_dir"]
            if target_rel_dir.exists():
                shutil.rmtree(target_rel_dir)
            source_rel_dir.rename(target_rel_dir)
            moved_path = target_rel_dir / "entity.json"
            try:
                with open(moved_path, encoding="utf-8") as handle:
                    rel_data = json.load(handle)
                rel_data["entity_id"] = target_id
                save_facet_relationship(operation["facet"], target_id, rel_data)
            except (json.JSONDecodeError, OSError):
                pass
            continue

        save_facet_relationship(
            operation["facet"], target_id, operation["relationship"]
        )
        save_observations(operation["facet"], target_id, operation["observations"])
        shutil.rmtree(operation["source_rel_dir"])


def _apply_segment_plan(operations: list[dict[str, Any]]) -> None:
    for operation in operations:
        out_path = operation["path"]
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(operation["data"], handle, indent=2)
        tmp_path.rename(out_path)


def _clear_merge_caches() -> list[str]:
    clear_journal_entity_cache()
    clear_relationship_caches()
    clear_observation_cache()
    clear_observation_count_cache()
    clear_entity_loading_cache()
    return [
        "journal_entity_cache",
        "relationship_caches",
        "observation_cache",
        "observation_count_cache",
        "entity_loading_cache",
    ]


def _audit_counts(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": {
            "akas_added": result["identity"]["akas_added_count"],
            "emails_added": result["identity"]["emails_added_count"],
            "principal_transferred": result["identity"]["principal_transferred"],
        },
        "voiceprints": {
            "added": result["voiceprints"]["added"],
            "skipped_duplicate": result["voiceprints"]["skipped_duplicate"],
            "target_total": result["voiceprints"]["target_total"],
        },
        "facets": {
            "moved": result["facets"]["moved_count"],
            "merged": result["facets"]["merged_count"],
            "observations_appended": result["facets"]["observations_appended"],
        },
        "segments": {
            "labels_rewritten": result["segments"]["labels_rewritten"],
            "corrections_rewritten": result["segments"]["corrections_rewritten"],
            "files_scanned": result["segments"]["files_scanned"],
            "errors": len(result["segments"]["errors"]),
        },
    }


def _append_audit_log(
    *,
    source_id: str,
    source_display_name: str,
    target_id: str,
    target_display_name: str,
    result: dict[str, Any],
    caller: str,
) -> str:
    logs_dir = Path(get_journal()) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    audit_path = logs_dir / "entity-merges.jsonl"
    payload = {
        "ts": now_ms(),
        "source_id": source_id,
        "source_display_name": source_display_name,
        "target_id": target_id,
        "target_display_name": target_display_name,
        "principal_transferred": result["identity"]["principal_transferred"],
        "counts": _audit_counts(result),
        "caller": caller,
    }
    with open(audit_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(audit_path)


def merge_entity(
    source_id: str,
    target_id: str,
    *,
    keep_source_as_aka: bool = True,
    commit: bool = False,
    caller: str = "entities.merge",
) -> dict[str, Any]:
    if source_id == target_id:
        return {"error": "Source and target must be different entities."}

    source_entity = load_journal_entity(source_id)
    if not source_entity:
        return {"error": f"Source entity not found: {source_id}"}

    target_entity = load_journal_entity(target_id)
    if not target_entity:
        return {"error": f"Target entity not found: {target_id}"}

    if source_entity.get("blocked"):
        return {"error": f"Cannot merge blocked entity: {source_id}"}
    if target_entity.get("blocked"):
        return {"error": f"Cannot merge blocked entity: {target_id}"}
    if source_entity.get("is_principal") and target_entity.get("is_principal"):
        return {"error": "Cannot merge two principal entities."}

    source_display = str(source_entity.get("name", source_id))
    target_display = str(target_entity.get("name", target_id))

    offenders = _check_aka_cross_references(source_id, source_display, target_id)
    if offenders:
        offender_str = ", ".join(offenders)
        return {
            "error": f"Cannot merge '{source_id}': referenced in aka lists of entity ids: {offender_str}"
        }

    planned_target, identity_plan = _plan_identity_merge(
        source_entity,
        target_entity,
        keep_source_as_aka=keep_source_as_aka,
    )
    resume_source = _plan_resume_marker(
        source_entity,
        target_id,
        principal_transferred=identity_plan["principal_transferred"],
    )
    voiceprint_plan = _plan_voiceprint_merge(source_id, target_id)
    facet_plan = _plan_facet_merge(source_id, target_id)
    segment_plan = _plan_segment_rewrites(source_id, target_id)

    zero = _empty_result_section()
    result: dict[str, Any] = {
        "merged": commit,
        "source_id": source_id,
        "target_id": target_id,
        "identity": identity_plan if commit else zero["identity"],
        "voiceprints": voiceprint_plan["section"] if commit else zero["voiceprints"],
        "facets": facet_plan["section"] if commit else zero["facets"],
        "segments": segment_plan["section"] if commit else zero["segments"],
        "caches_cleared": [],
        "audit_log_path": None,
        "would_identity": None if commit else identity_plan,
        "would_voiceprints": None if commit else voiceprint_plan["section"],
        "would_facets": None if commit else facet_plan["section"],
        "would_segments": None if commit else segment_plan["section"],
    }

    if not commit:
        return result

    try:
        save_journal_entity(resume_source)
        save_journal_entity(planned_target)

        if voiceprint_plan["items"]:
            save_voiceprints_batch(target_id, voiceprint_plan["items"])

        _apply_facet_plan(facet_plan["operations"], target_id)
        _apply_segment_plan(segment_plan["operations"])

        discovery_cache = Path(get_journal()) / "awareness" / "discovery_clusters.json"
        caches_cleared = _clear_merge_caches()
        if discovery_cache.exists():
            discovery_cache.unlink()
            caches_cleared.append("discovery_clusters")

        source_entity_dir = Path(get_journal()) / "entities" / source_id
        if source_entity_dir.exists():
            shutil.rmtree(source_entity_dir)

        result["caches_cleared"] = caches_cleared

        try:
            result["audit_log_path"] = _append_audit_log(
                source_id=source_id,
                source_display_name=source_display,
                target_id=target_id,
                target_display_name=str(planned_target.get("name", target_display)),
                result=result,
                caller=caller,
            )
        except OSError as exc:
            result["segments"]["errors"].append(
                {
                    "kind": "audit_log",
                    "path": str(Path(get_journal()) / "logs" / "entity-merges.jsonl"),
                    "message": str(exc),
                }
            )

        return result
    except Exception as exc:
        return {"error": str(exc)}
