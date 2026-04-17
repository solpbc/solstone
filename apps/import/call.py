# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for import review and resolution.

Auto-discovered by ``think.call`` and mounted as ``sol call import ...``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import typer

from think.entities.core import EntityDict, entity_slug
from think.entities.journal import (
    has_journal_principal,
    journal_entity_path,
    load_all_journal_entities,
    save_journal_entity,
)
from think.entities.observations import load_observations, save_observations
from think.entities.relationships import (
    load_facet_relationship,
    save_facet_relationship,
)
from think.utils import get_journal, require_solstone

app = typer.Typer(help="Import review and resolution.")


@app.callback()
def _require_up() -> None:
    require_solstone()


ingest = import_module("apps.import.ingest")
journal_sources = import_module("apps.import.journal_sources")

_ENTITY_FILE_TYPES = {
    "entity_relationship",
    "entity_observations",
    "detected_entities",
    "activity_records",
}
_append_decision = ingest._append_decision
_categorize_field = ingest._categorize_field
_write_state_atomic = ingest._write_state_atomic
find_journal_source_by_name = journal_sources.find_journal_source_by_name
get_state_directory = journal_sources.get_state_directory


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _resolve_source(name: str) -> tuple[dict, str, Path]:
    source = find_journal_source_by_name(name)
    if not source:
        _fail(
            f"Import source '{name}' not found. Check available sources in "
            "~/.local/share/solstone/app-storage/import/journal_sources/."
        )

    key = source.get("key")
    if not isinstance(key, str) or len(key) < 8:
        _fail(f"Import source '{name}' has an invalid key.")

    key_prefix = key[:8]
    state_dir = get_state_directory(key_prefix)
    return source, key_prefix, state_dir


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = cfg
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _write_config(config: dict) -> None:
    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.chmod(config_path, 0o600)


def merge_entity_fields(
    target: EntityDict, source: EntityDict
) -> tuple[EntityDict, list[str]]:
    merged: EntityDict = dict(target)
    pre_merge_snapshot = dict(merged)

    aka_by_lower: dict[str, str] = {}
    for values in (merged.get("aka", []), source.get("aka", [])):
        if not isinstance(values, list):
            continue
        for value in values:
            if not value:
                continue
            key = str(value).lower()
            if key not in aka_by_lower:
                aka_by_lower[key] = str(value)
    if aka_by_lower:
        merged["aka"] = sorted(aka_by_lower.values(), key=str.lower)

    merged_emails: list[str] = []
    seen_emails: set[str] = set()
    for values in (merged.get("emails", []), source.get("emails", [])):
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
    if merged_emails:
        merged["emails"] = merged_emails

    source_created = source.get("created_at")
    target_created = merged.get("created_at")
    if source_created is not None and target_created is not None:
        merged["created_at"] = min(source_created, target_created)
    elif source_created is not None:
        merged["created_at"] = source_created

    fields_changed = sorted(
        key
        for key in set(pre_merge_snapshot) | set(merged)
        if pre_merge_snapshot.get(key) != merged.get(key)
    )
    return merged, fields_changed


def _allocate_slug(name: str) -> str | None:
    base_slug = entity_slug(name)
    if not base_slug:
        return None

    for attempt in range(1, 102):
        candidate = base_slug if attempt == 1 else f"{base_slug}_{attempt}"
        if not journal_entity_path(candidate).exists():
            return candidate
    return None


def _log_resolution(
    log_path: Path,
    action: str,
    item_type: str,
    item_id: str,
    reason: str,
    **extra: Any,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "item_type": item_type,
        "item_id": item_id,
        "reason": reason,
        "resolved_by": "talent",
    }
    entry.update(extra)
    _append_decision(log_path, entry)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_entity_state(state_path: Path) -> dict[str, dict[str, Any]]:
    entity_state = _load_json(state_path, {})
    if not isinstance(entity_state, dict):
        entity_state = {}

    id_map = entity_state.get("id_map")
    received = entity_state.get("received")
    if not isinstance(id_map, dict) or not isinstance(received, dict):
        return {"id_map": {}, "received": {}}

    return {"id_map": dict(id_map), "received": dict(received)}


def _parse_jsonl_text(source_data: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_data.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid JSONL at line {line_number}: item must be an object"
            )
        items.append(item)
    return items


def _append_jsonl_items(target_path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    facet_ingest = import_module("apps.import.facet_ingest")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "ab") as handle:
        handle.write(facet_ingest._serialize_jsonl(items))


def _load_config_diff(diff_path: Path) -> dict[str, dict[str, Any]]:
    diff = _load_json(diff_path, {})
    if not isinstance(diff, dict):
        _fail("Config diff is invalid.")
    return diff


def _write_config_diff(diff_path: Path, diff: dict[str, dict[str, Any]]) -> None:
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(
        json.dumps(diff, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_config_field(state_dir: Path, field: str, action: str) -> None:
    diff_path = state_dir / "config" / "diff.json"
    if not diff_path.exists():
        _fail("No staged config diff found.")

    diff = _load_config_diff(diff_path)
    if field not in diff:
        _fail(f"Config field '{field}' is not staged.")

    if action not in {"apply", "keep"}:
        _fail("Action must be 'apply' or 'keep'.")

    diff_entry = diff[field]
    if not isinstance(diff_entry, dict):
        _fail(f"Config field '{field}' has invalid diff data.")

    if action == "apply":
        from think.utils import get_config

        config = get_config()
        _set_nested(config, field, diff_entry.get("source"))
        _write_config(config)
        log_action = "config_field_applied"
        reason = "review_apply"
    else:
        log_action = "config_field_kept"
        reason = "review_keep"

    diff.pop(field)
    if diff:
        _write_config_diff(diff_path, diff)
    else:
        diff_path.unlink(missing_ok=True)
        (state_dir / "config" / "source_config.json").unlink(missing_ok=True)

    _log_resolution(
        state_dir / "config" / "log.jsonl",
        action=log_action,
        item_type="config",
        item_id=field,
        reason=reason,
        category=diff_entry.get("category", _categorize_field(field)),
        source=diff_entry.get("source"),
        target_previous=diff_entry.get("target"),
    )


@app.command("list-staged")
def list_staged(
    source: str = typer.Option(..., "--source", help="Import source name."),
    area: str | None = typer.Option(
        None, "--area", help="Area: entities, facets, or config."
    ),
) -> None:
    _, _, state_dir = _resolve_source(source)

    if area is not None and area not in {"entities", "facets", "config"}:
        _fail("Area must be one of: entities, facets, config.")

    if area in {None, "entities"}:
        staged_dir = state_dir / "entities" / "staged"
        for staged_path in sorted(staged_dir.glob("*.json")):
            payload = _load_json(staged_path, {})
            if not isinstance(payload, dict):
                continue
            line = {
                "area": "entities",
                "source_id": staged_path.stem,
                "reason": payload.get("reason"),
                "source_entity": payload.get("source_entity"),
                "match_candidates": payload.get("match_candidates"),
                "staged_at": payload.get("staged_at"),
            }
            typer.echo(json.dumps(line, ensure_ascii=False))

    if area in {None, "facets"}:
        staged_dir = state_dir / "facets" / "staged"
        for staged_path in sorted(staged_dir.glob("**/*.staged.json")):
            payload = _load_json(staged_path, {})
            if not isinstance(payload, dict):
                continue
            relative_path = staged_path.relative_to(staged_dir)
            parts = relative_path.parts
            if len(parts) < 3:
                continue
            line = {
                "area": "facets",
                "staged_file": relative_path.as_posix(),
                "facet": parts[0],
                "file_type": parts[1],
            }
            line.update(payload)
            typer.echo(json.dumps(line, ensure_ascii=False))

    if area in {None, "config"}:
        diff_path = state_dir / "config" / "diff.json"
        if diff_path.exists():
            diff = _load_config_diff(diff_path)
            typer.echo(json.dumps({"area": "config", "diff": diff}, ensure_ascii=False))


@app.command("resolve-entity")
def resolve_entity(
    source_id: str = typer.Argument(help="Source entity ID."),
    action: str = typer.Argument(help="Action: merge, create, or skip."),
    source: str = typer.Option(..., "--source", help="Import source name."),
    target: str | None = typer.Option(
        None, "--target", help="Target entity ID for merge."
    ),
) -> None:
    _, _, state_dir = _resolve_source(source)

    if action not in {"merge", "create", "skip"}:
        _fail("Action must be 'merge', 'create', or 'skip'.")

    staged_path = state_dir / "entities" / "staged" / f"{source_id}.json"
    if not staged_path.exists():
        _fail(f"Staged entity '{source_id}' not found.")

    payload = _load_json(staged_path, {})
    if not isinstance(payload, dict):
        _fail(f"Staged entity '{source_id}' is invalid.")

    source_entity = payload.get("source_entity")
    if not isinstance(source_entity, dict):
        _fail(f"Staged entity '{source_id}' is missing source_entity.")

    log_path = state_dir / "entities" / "log.jsonl"
    state_path = state_dir / "entities" / "state.json"
    entity_state = _load_entity_state(state_path)
    reason = str(payload.get("reason", ""))
    match_candidates = payload.get("match_candidates")
    match_tier = None
    if isinstance(match_candidates, list) and match_candidates:
        first_candidate = match_candidates[0]
        if isinstance(first_candidate, dict):
            match_tier = first_candidate.get("tier")

    if action == "merge":
        if not target:
            _fail("--target is required for merge.")

        target_entities = load_all_journal_entities()
        target_entity = target_entities.get(target)
        if target_entity is None:
            _fail(
                f"Target entity '{target}' not found. Use "
                "'list-staged --source SOURCE --area entities' to check "
                "match candidates, or use 'create' instead of 'merge'."
            )

        merged, fields_changed = merge_entity_fields(target_entity, source_entity)
        save_journal_entity(merged)
        entity_state["id_map"][source_id] = target
        _write_state_atomic(state_path, entity_state)
        staged_path.unlink()
        _log_resolution(
            log_path,
            action="resolved_merge",
            item_type="entity",
            item_id=source_id,
            reason=reason,
            source=source_entity,
            target=merged,
            fields_changed=fields_changed,
            match_tier=match_tier,
        )
        typer.echo(f"Merged {source_id} into {target}.")
        return

    if target is not None:
        _fail("--target is only valid for merge.")

    if action == "create":
        created_entity = dict(source_entity)
        final_id = str(created_entity.get("id") or source_id)
        if reason == "id_collision" or journal_entity_path(final_id).exists():
            allocated = _allocate_slug(str(created_entity.get("name", "")))
            if allocated is None:
                _fail(
                    f"Unable to allocate a slug for '{created_entity.get('name', '')}'."
                )
            final_id = allocated
        created_entity["id"] = final_id

        if reason == "principal_conflict" and has_journal_principal():
            created_entity["is_principal"] = False

        save_journal_entity(created_entity)
        entity_state["id_map"][source_id] = final_id
        _write_state_atomic(state_path, entity_state)
        staged_path.unlink()
        _log_resolution(
            log_path,
            action="resolved_create",
            item_type="entity",
            item_id=source_id,
            reason=reason,
            source=source_entity,
            target=created_entity,
            match_tier=match_tier,
            fields_changed=[],
        )
        typer.echo(f"Created entity {final_id} from {source_id}.")
        return

    staged_path.unlink()
    _log_resolution(
        log_path,
        action="resolved_skip",
        item_type="entity",
        item_id=source_id,
        reason=reason,
        source=source_entity,
        target=None,
        match_tier=match_tier,
        fields_changed=[],
    )
    typer.echo(f"Skipped staged entity {source_id}.")


@app.command("resolve-facet")
def resolve_facet(
    staged_file: str = typer.Argument(
        help="Staged file path relative to facets/staged/."
    ),
    action: str = typer.Argument(help="Action: apply or skip."),
    source: str = typer.Option(..., "--source", help="Import source name."),
) -> None:
    _, _, state_dir = _resolve_source(source)

    if action not in {"apply", "skip"}:
        _fail("Action must be 'apply' or 'skip'.")

    staged_dir = state_dir / "facets" / "staged"
    staged_path = staged_dir / staged_file
    if not staged_path.exists():
        _fail(f"Staged facet file '{staged_file}' not found.")

    payload = _load_json(staged_path, {})
    if not isinstance(payload, dict):
        _fail(f"Staged facet file '{staged_file}' is invalid.")

    parts = Path(staged_file).parts
    if len(parts) < 3:
        _fail(f"Staged facet file '{staged_file}' has an invalid path.")

    facet_name = parts[0]
    file_type = parts[1]
    reason = str(payload.get("reason", ""))
    log_path = state_dir / "facets" / "log.jsonl"

    if reason == "facet_json_conflict":
        item_id = f"{facet_name}/facet.json"
    else:
        item_id = f"{facet_name}/{payload.get('source_path', staged_file)}"

    if action == "skip":
        staged_path.unlink()
        _log_resolution(
            log_path,
            action="resolved_skip",
            item_type=file_type,
            item_id=item_id,
            reason=reason,
            facet=facet_name,
            staged_path=str(staged_path),
        )
        typer.echo(f"Skipped staged facet file {staged_file}.")
        return

    if reason == "unmapped_entity":
        if file_type not in _ENTITY_FILE_TYPES:
            _fail(f"Unsupported staged facet file type '{file_type}'.")

        facet_ingest = import_module("apps.import.facet_ingest")
        entities_state = _load_entity_state(state_dir / "entities" / "state.json")
        id_map = entities_state.get("id_map", {})
        source_entity_id = str(payload.get("source_entity_id", ""))
        if source_entity_id not in id_map:
            _fail(
                f"Entity {source_entity_id} has no mapping yet. Run entity review first."
            )

        source_path = str(payload.get("source_path", ""))
        source_data = str(payload.get("source_data", ""))

        normalized_path, path_info = facet_ingest._parse_path(source_path, file_type)
        if file_type == "entity_relationship":
            parsed_data: Any = json.loads(source_data)
        else:
            parsed_data = _parse_jsonl_text(source_data)

        remapped_data, remapped_path_info = facet_ingest._remap_entity_ids(
            parsed_data, id_map, file_type, path_info
        )
        target_path = Path(get_journal()) / "facets" / facet_name / normalized_path

        if file_type == "entity_relationship":
            entity_id = remapped_path_info["entity_id"]
            source_relationship = dict(remapped_data)
            source_relationship["entity_id"] = entity_id
            target_relationship = load_facet_relationship(facet_name, entity_id) or {}
            merged_relationship = {**source_relationship, **target_relationship}
            save_facet_relationship(facet_name, entity_id, merged_relationship)
        elif file_type == "entity_observations":
            entity_id = remapped_path_info["entity_id"]
            target_observations = load_observations(facet_name, entity_id)
            seen = {
                (item.get("content", ""), item.get("observed_at"))
                for item in target_observations
            }
            merged_observations = list(target_observations)
            for item in remapped_data:
                key = (item.get("content", ""), item.get("observed_at"))
                if key in seen:
                    continue
                seen.add(key)
                merged_observations.append(item)
            save_observations(facet_name, entity_id, merged_observations)
        elif file_type in {"detected_entities", "activity_records"}:
            existing_items = (
                _parse_jsonl_text(target_path.read_text(encoding="utf-8"))
                if target_path.exists()
                else []
            )
            existing_ids = {item.get("id") for item in existing_items}
            new_items = [
                item for item in remapped_data if item.get("id") not in existing_ids
            ]
            _append_jsonl_items(target_path, new_items)
        else:
            _fail(f"Unsupported staged facet file type '{file_type}'.")

        staged_path.unlink()
        _log_resolution(
            log_path,
            action="resolved_apply",
            item_type=file_type,
            item_id=item_id,
            reason=reason,
            facet=facet_name,
            staged_path=str(staged_path),
            target_path=str(target_path),
        )
        typer.echo(f"Applied staged facet file {staged_file}.")
        return

    if reason == "facet_json_conflict":
        target_path = Path(get_journal()) / "facets" / facet_name / "facet.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(payload.get("source_content"), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        staged_path.unlink()
        _log_resolution(
            log_path,
            action="resolved_apply",
            item_type=file_type,
            item_id=item_id,
            reason=reason,
            facet=facet_name,
            staged_path=str(staged_path),
            target_path=str(target_path),
        )
        typer.echo(f"Applied staged facet file {staged_file}.")
        return

    _fail(f"Unsupported staged facet reason '{reason}'.")


@app.command("resolve-config")
def resolve_config(
    field: str = typer.Argument(help="Dotted config field path."),
    action: str = typer.Argument(help="Action: apply or keep."),
    source: str = typer.Option(..., "--source", help="Import source name."),
) -> None:
    _, _, state_dir = _resolve_source(source)
    _resolve_config_field(state_dir, field, action)
    typer.echo(f"Resolved config field {field} with action {action}.")


@app.command("resolve-config-all")
def resolve_config_all(
    source: str = typer.Option(..., "--source", help="Import source name."),
    category: str = typer.Option(
        ..., "--category", help="Category: transferable or preference."
    ),
) -> None:
    _, _, state_dir = _resolve_source(source)

    if category not in {"transferable", "preference"}:
        _fail("Category must be 'transferable' or 'preference'.")

    diff_path = state_dir / "config" / "diff.json"
    if not diff_path.exists():
        _fail("No staged config diff found.")

    diff = _load_config_diff(diff_path)
    fields = [
        field
        for field, diff_entry in diff.items()
        if isinstance(diff_entry, dict)
        and diff_entry.get("category", _categorize_field(field)) == category
    ]
    for field in list(fields):
        _resolve_config_field(state_dir, field, "apply")

    typer.echo(f"Applied {len(fields)} {category} config field(s).")
