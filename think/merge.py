# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Journal merge engine - one-shot merge of a source journal into the target."""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from think.entities.core import entity_slug
from think.entities.journal import (
    load_all_journal_entities,
    save_journal_entity,
)
from think.entities.matching import find_matching_entity
from think.entities.observations import save_observations
from think.entities.relationships import save_facet_relationship
from think.utils import CHRONICLE_DIR, iter_segments

DATE_RE = re.compile(r"^\d{8}$")
logger = logging.getLogger(__name__)


@dataclass
class MergeSummary:
    segments_copied: int = 0
    segments_skipped: int = 0
    segments_errored: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    entities_skipped: int = 0
    entities_staged: int = 0
    facets_created: int = 0
    facets_merged: int = 0
    imports_copied: int = 0
    imports_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def merge_journals(
    source: Path,
    target: Path,
    dry_run: bool = False,
    log_path: Path | None = None,
    staging_path: Path | None = None,
) -> MergeSummary:
    summary = MergeSummary()
    target_entities = load_all_journal_entities()

    _merge_segments(source, target, summary, dry_run, log_path=log_path)
    _merge_entities(
        source,
        summary,
        dry_run,
        target_entities,
        log_path=log_path,
        staging_path=staging_path,
    )
    _merge_facets(source, target, summary, dry_run, log_path=log_path)
    _merge_imports(source, target, summary, dry_run, log_path=log_path)

    return summary


def _log_decision(log_path: Path | None, entry: dict[str, Any]) -> None:
    if log_path is None:
        return

    payload = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _source_day_dirs(source: Path) -> dict[str, Path]:
    chronicle_dir = source / CHRONICLE_DIR
    chronicle_days = {}
    if chronicle_dir.is_dir():
        chronicle_days = {
            entry.name: entry
            for entry in chronicle_dir.iterdir()
            if entry.is_dir() and DATE_RE.match(entry.name)
        }
    flat_days = {
        entry.name: entry
        for entry in source.iterdir()
        if entry.is_dir() and DATE_RE.match(entry.name)
    }
    if chronicle_days and flat_days:
        logger.warning(
            "Merge source has both flat and %s/ day dirs; preferring %s/.",
            CHRONICLE_DIR,
            CHRONICLE_DIR,
        )
        return chronicle_days
    return chronicle_days or flat_days


def _merge_segments(
    source: Path,
    target: Path,
    summary: MergeSummary,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    target_chronicle = target / CHRONICLE_DIR
    if not dry_run:
        target_chronicle.mkdir(parents=True, exist_ok=True)

    for day_name, source_day in sorted(_source_day_dirs(source).items()):
        target_day = target_chronicle / day_name
        for stream, seg_key, seg_path in iter_segments(source_day):
            if stream == "_default":
                target_path = target_day / seg_key
            else:
                target_path = target_day / stream / seg_key

            item_id = f"{day_name}/{stream}/{seg_key}"
            try:
                if target_path.exists():
                    summary.segments_skipped += 1
                    _log_decision(
                        log_path,
                        {
                            "action": "segment_skipped",
                            "item_type": "segment",
                            "item_id": item_id,
                            "reason": "target_exists",
                        },
                    )
                    continue

                if dry_run:
                    summary.segments_copied += 1
                    _log_decision(
                        log_path,
                        {
                            "action": "segment_copied",
                            "item_type": "segment",
                            "item_id": item_id,
                            "reason": "new",
                        },
                    )
                    continue

                shutil.copytree(seg_path, target_path, copy_function=shutil.copy2)
                summary.segments_copied += 1
                _log_decision(
                    log_path,
                    {
                        "action": "segment_copied",
                        "item_type": "segment",
                        "item_id": item_id,
                        "reason": "new",
                    },
                )
            except Exception as exc:
                summary.segments_errored += 1
                summary.errors.append(f"segment {day_name}/{stream}/{seg_key}: {exc}")


def _merge_entities(
    source: Path,
    summary: MergeSummary,
    dry_run: bool,
    target_entities: dict[str, dict[str, Any]],
    log_path: Path | None = None,
    staging_path: Path | None = None,
) -> None:

    target_has_principal = any(
        bool(entity.get("is_principal")) for entity in target_entities.values()
    )
    source_entities_dir = source / "entities"
    if not source_entities_dir.is_dir():
        return

    for entity_dir in sorted(source_entities_dir.iterdir()):
        entity_path = entity_dir / "entity.json"
        if not entity_dir.is_dir() or not entity_path.is_file():
            continue

        try:
            source_entity = json.loads(entity_path.read_text(encoding="utf-8"))
            source_name = str(source_entity.get("name", "")).strip()
            if not source_name:
                raise ValueError("missing entity name")

            entity_id = str(
                source_entity.get("id") or entity_dir.name or entity_slug(source_name)
            )
            if not entity_id:
                raise ValueError("missing entity id")
            source_entity["id"] = entity_id

            match = find_matching_entity(source_name, list(target_entities.values()))
            if match is None:
                if entity_id in target_entities:
                    if staging_path is not None:
                        summary.entities_staged += 1
                        if not dry_run:
                            staged_dir = staging_path / entity_id
                            staged_dir.mkdir(parents=True, exist_ok=True)
                            (staged_dir / "entity.json").write_text(
                                json.dumps(
                                    source_entity,
                                    indent=2,
                                    ensure_ascii=False,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                        _log_decision(
                            log_path,
                            {
                                "action": "entity_staged",
                                "item_type": "entity",
                                "item_id": entity_id,
                                "reason": "id_collision_no_match",
                                "source": source_entity,
                                "target": dict(target_entities[entity_id]),
                            },
                        )
                    else:
                        summary.entities_skipped += 1
                        _log_decision(
                            log_path,
                            {
                                "action": "entity_skipped",
                                "item_type": "entity",
                                "item_id": entity_id,
                                "reason": "id_collision_no_staging",
                                "source": source_entity,
                                "target": dict(target_entities[entity_id]),
                            },
                        )
                    continue

                if source_entity.get("is_principal") and target_has_principal:
                    source_entity["is_principal"] = False
                elif source_entity.get("is_principal"):
                    target_has_principal = True

                if not dry_run:
                    save_journal_entity(source_entity)
                summary.entities_created += 1
                target_entities[source_entity["id"]] = source_entity
                _log_decision(
                    log_path,
                    {
                        "action": "entity_created",
                        "item_type": "entity",
                        "item_id": source_entity["id"],
                        "reason": "no_match",
                    },
                )
                continue

            target_id = str(match.get("id", ""))
            if not target_id:
                raise ValueError("matched target entity missing id")

            target_entity = dict(target_entities.get(target_id, match))
            pre_merge_snapshot = dict(target_entity)

            aka_by_lower: dict[str, str] = {}
            for values in (target_entity.get("aka", []), source_entity.get("aka", [])):
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not value:
                        continue
                    key = str(value).lower()
                    if key not in aka_by_lower:
                        aka_by_lower[key] = str(value)
            if aka_by_lower:
                target_entity["aka"] = sorted(aka_by_lower.values(), key=str.lower)

            merged_emails: list[str] = []
            seen_emails: set[str] = set()
            for values in (
                target_entity.get("emails", []),
                source_entity.get("emails", []),
            ):
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
                target_entity["emails"] = merged_emails

            if not dry_run:
                save_journal_entity(target_entity)
            summary.entities_merged += 1
            target_entities[target_id] = target_entity
            fields_changed = sorted(
                key
                for key in set(pre_merge_snapshot) | set(target_entity)
                if pre_merge_snapshot.get(key) != target_entity.get(key)
            )
            _log_decision(
                log_path,
                {
                    "action": "entity_merged",
                    "item_type": "entity",
                    "item_id": target_id,
                    "reason": "name_match",
                    "source": source_entity,
                    "target": pre_merge_snapshot,
                    "fields_changed": fields_changed,
                },
            )
        except Exception as exc:
            summary.errors.append(f"entity {entity_dir.name}: {exc}")


def _merge_facets(
    source: Path,
    target: Path,
    summary: MergeSummary,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:

    source_facets_dir = source / "facets"
    if not source_facets_dir.is_dir():
        return

    for source_facet_dir in sorted(source_facets_dir.iterdir()):
        facet_json = source_facet_dir / "facet.json"
        if not source_facet_dir.is_dir() or not facet_json.is_file():
            continue

        facet_name = source_facet_dir.name
        target_facet_dir = target / "facets" / facet_name

        try:
            if not target_facet_dir.exists():
                if not dry_run:
                    shutil.copytree(
                        source_facet_dir,
                        target_facet_dir,
                        copy_function=shutil.copy2,
                    )
                summary.facets_created += 1
                _log_decision(
                    log_path,
                    {
                        "action": "facet_created",
                        "item_type": "facet",
                        "item_id": facet_name,
                        "reason": "new",
                    },
                )
                continue

            _merge_overlapping_facet(
                facet_name,
                source_facet_dir,
                target_facet_dir,
                summary,
                dry_run,
                log_path=log_path,
            )
            summary.facets_merged += 1
            _log_decision(
                log_path,
                {
                    "action": "facet_merged",
                    "item_type": "facet",
                    "item_id": facet_name,
                    "reason": "overlap",
                },
            )
        except Exception as exc:
            summary.errors.append(f"facet {facet_name}: {exc}")


def _merge_overlapping_facet(
    facet_name: str,
    source_facet_dir: Path,
    target_facet_dir: Path,
    summary: MergeSummary,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    source_entities_dir = source_facet_dir / "entities"
    if source_entities_dir.is_dir():
        for source_entity_dir in sorted(source_entities_dir.iterdir()):
            source_entity_json = source_entity_dir / "entity.json"
            if not source_entity_dir.is_dir() or not source_entity_json.is_file():
                continue

            entity_id = source_entity_dir.name
            target_entity_dir = target_facet_dir / "entities" / entity_id
            try:
                if target_entity_dir.exists():
                    source_relationship = json.loads(
                        source_entity_json.read_text(encoding="utf-8")
                    )
                    target_relationship_path = target_entity_dir / "entity.json"
                    target_relationship: dict[str, Any] = {}
                    if target_relationship_path.is_file():
                        target_relationship = json.loads(
                            target_relationship_path.read_text(encoding="utf-8")
                        )
                    merged_relationship = {**source_relationship, **target_relationship}

                    source_observations = _read_jsonl(
                        source_entity_dir / "observations.jsonl"
                    )
                    target_observations = _read_jsonl(
                        target_entity_dir / "observations.jsonl"
                    )
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

                    if not dry_run:
                        save_facet_relationship(
                            facet_name, entity_id, merged_relationship
                        )
                        save_observations(facet_name, entity_id, merged_observations)
                    _log_decision(
                        log_path,
                        {
                            "action": "facet_entity_merged",
                            "item_type": "facet_entity",
                            "item_id": f"{facet_name}/entities/{entity_id}",
                            "reason": "overlap",
                        },
                    )
                    continue

                if not dry_run:
                    shutil.copytree(
                        source_entity_dir,
                        target_entity_dir,
                        copy_function=shutil.copy2,
                    )
                _log_decision(
                    log_path,
                    {
                        "action": "facet_entity_copied",
                        "item_type": "facet_entity",
                        "item_id": f"{facet_name}/entities/{entity_id}",
                        "reason": "new",
                    },
                )
            except Exception as exc:
                summary.errors.append(f"facet {facet_name} entity {entity_id}: {exc}")

    if source_entities_dir.is_dir():
        for source_det_file in sorted(source_entities_dir.glob("*.jsonl")):
            try:
                target_det_file = target_facet_dir / "entities" / source_det_file.name
                target_items = _read_jsonl(target_det_file)
                seen_ids = {item.get("id") for item in target_items if item.get("id")}
                source_items = _read_jsonl(source_det_file)
                new_items = []
                for item in source_items:
                    item_id = item.get("id", "")
                    log_id = f"{facet_name}/entities/{source_det_file.name}/{item_id}"
                    if item_id in seen_ids:
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_detected_entity_merged",
                                "item_type": "facet_detected_entity",
                                "item_id": log_id,
                                "reason": "duplicate_skip",
                            },
                        )
                    else:
                        new_items.append(item)
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_detected_entity_merged",
                                "item_type": "facet_detected_entity",
                                "item_id": log_id,
                                "reason": "appended",
                            },
                        )
                if new_items and not dry_run:
                    _append_jsonl(target_det_file, new_items)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} detected entities {source_det_file.name}: {exc}"
                )

    source_todos_dir = source_facet_dir / "todos"
    if source_todos_dir.is_dir():
        for source_todo_file in sorted(source_todos_dir.glob("*.jsonl")):
            try:
                target_todo_file = target_facet_dir / "todos" / source_todo_file.name
                target_items = _read_jsonl(target_todo_file)
                seen = {(item["text"], item.get("created_at")) for item in target_items}
                new_items = []
                for item in _read_jsonl(source_todo_file):
                    log_id = f"{facet_name}/todos/{source_todo_file.name}/{item.get('text', '')}"
                    if (item["text"], item.get("created_at")) in seen:
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_todo_merged",
                                "item_type": "todo",
                                "item_id": log_id,
                                "reason": "duplicate_skip",
                            },
                        )
                    else:
                        new_items.append(item)
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_todo_merged",
                                "item_type": "todo",
                                "item_id": log_id,
                                "reason": "appended",
                            },
                        )
                if new_items and not dry_run:
                    _append_jsonl(target_todo_file, new_items)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} todo {source_todo_file.name}: {exc}"
                )

    source_calendar_dir = source_facet_dir / "calendar"
    if source_calendar_dir.is_dir():
        for source_calendar_file in sorted(source_calendar_dir.glob("*.jsonl")):
            try:
                target_calendar_file = (
                    target_facet_dir / "calendar" / source_calendar_file.name
                )
                target_items = _read_jsonl(target_calendar_file)
                seen = {(item["title"], item.get("start")) for item in target_items}
                new_items = []
                for item in _read_jsonl(source_calendar_file):
                    log_id = f"{facet_name}/calendar/{source_calendar_file.name}/{item.get('title', '')}"
                    if (item["title"], item.get("start")) in seen:
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_calendar_merged",
                                "item_type": "calendar",
                                "item_id": log_id,
                                "reason": "duplicate_skip",
                            },
                        )
                    else:
                        new_items.append(item)
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_calendar_merged",
                                "item_type": "calendar",
                                "item_id": log_id,
                                "reason": "appended",
                            },
                        )
                if new_items and not dry_run:
                    _append_jsonl(target_calendar_file, new_items)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} calendar {source_calendar_file.name}: {exc}"
                )

    source_activities_dir = source_facet_dir / "activities"
    if source_activities_dir.is_dir():
        source_config_file = source_activities_dir / "activities.jsonl"
        target_config_file = target_facet_dir / "activities" / "activities.jsonl"
        if source_config_file.is_file():
            try:
                target_config = _read_jsonl(target_config_file)
                existing_ids = {item.get("id") for item in target_config}
                source_config = _read_jsonl(source_config_file)
                new_config = []
                for item in source_config:
                    log_id = f"{facet_name}/activities/{item.get('id', '')}"
                    if item.get("id") in existing_ids:
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_activities_config_merged",
                                "item_type": "activity_config",
                                "item_id": log_id,
                                "reason": "duplicate_skip",
                            },
                        )
                    else:
                        new_config.append(item)
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_activities_config_merged",
                                "item_type": "activity_config",
                                "item_id": log_id,
                                "reason": "appended",
                            },
                        )
                if new_config and not dry_run:
                    _append_jsonl(target_config_file, new_config)
            except Exception as exc:
                summary.errors.append(f"facet {facet_name} activities config: {exc}")

        for source_day_file in sorted(source_activities_dir.glob("*.jsonl")):
            if source_day_file.name == "activities.jsonl":
                continue
            try:
                target_day_file = target_facet_dir / "activities" / source_day_file.name
                target_records = _read_jsonl(target_day_file)
                existing_ids = {item.get("id") for item in target_records}
                source_records = _read_jsonl(source_day_file)
                new_records = []
                for item in source_records:
                    log_id = f"{facet_name}/activities/{source_day_file.name}/{item.get('id', '')}"
                    if item.get("id") in existing_ids:
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_activities_record_merged",
                                "item_type": "activity_record",
                                "item_id": log_id,
                                "reason": "duplicate_skip",
                            },
                        )
                    else:
                        new_records.append(item)
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_activities_record_merged",
                                "item_type": "activity_record",
                                "item_id": log_id,
                                "reason": "appended",
                            },
                        )
                if new_records and not dry_run:
                    _append_jsonl(target_day_file, new_records)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} activities {source_day_file.name}: {exc}"
                )

        for source_day_dir in sorted(source_activities_dir.iterdir()):
            if not source_day_dir.is_dir() or not DATE_RE.match(source_day_dir.name):
                continue
            for source_output_dir in sorted(source_day_dir.iterdir()):
                if not source_output_dir.is_dir():
                    continue
                target_output_dir = (
                    target_facet_dir
                    / "activities"
                    / source_day_dir.name
                    / source_output_dir.name
                )
                try:
                    if target_output_dir.exists():
                        _log_decision(
                            log_path,
                            {
                                "action": "facet_activities_output_copied",
                                "item_type": "activity_output",
                                "item_id": (
                                    f"{facet_name}/activities/{source_day_dir.name}/"
                                    f"{source_output_dir.name}"
                                ),
                                "reason": "target_exists_skip",
                            },
                        )
                        continue
                    if not dry_run:
                        shutil.copytree(
                            source_output_dir,
                            target_output_dir,
                            copy_function=shutil.copy2,
                        )
                    _log_decision(
                        log_path,
                        {
                            "action": "facet_activities_output_copied",
                            "item_type": "activity_output",
                            "item_id": (
                                f"{facet_name}/activities/{source_day_dir.name}/"
                                f"{source_output_dir.name}"
                            ),
                            "reason": "copied",
                        },
                    )
                except Exception as exc:
                    summary.errors.append(
                        "facet "
                        f"{facet_name} activities output "
                        f"{source_day_dir.name}/{source_output_dir.name}: {exc}"
                    )

    source_logs_dir = source_facet_dir / "logs"
    if source_logs_dir.is_dir():
        for source_log_file in sorted(source_logs_dir.glob("*.jsonl")):
            try:
                source_items = _read_jsonl(source_log_file)
                if source_items and not dry_run:
                    target_log_file = target_facet_dir / "logs" / source_log_file.name
                    _append_jsonl(target_log_file, source_items)
                for item in source_items:
                    _log_decision(
                        log_path,
                        {
                            "action": "facet_logs_appended",
                            "item_type": "facet_log",
                            "item_id": f"{facet_name}/logs/{source_log_file.name}",
                            "reason": "appended",
                        },
                    )
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} logs {source_log_file.name}: {exc}"
                )

    source_news_dir = source_facet_dir / "news"
    if source_news_dir.is_dir():
        target_news_dir = target_facet_dir / "news"
        for source_news_file in sorted(source_news_dir.glob("*.md")):
            try:
                target_news_file = target_news_dir / source_news_file.name
                if target_news_file.exists():
                    _log_decision(
                        log_path,
                        {
                            "action": "facet_news_skipped",
                            "item_type": "news",
                            "item_id": f"{facet_name}/news/{source_news_file.name}",
                            "reason": "target_exists",
                        },
                    )
                    continue
                if not dry_run:
                    target_news_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_news_file, target_news_file)
                _log_decision(
                    log_path,
                    {
                        "action": "facet_news_copied",
                        "item_type": "news",
                        "item_id": f"{facet_name}/news/{source_news_file.name}",
                        "reason": "new",
                    },
                )
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} news {source_news_file.name}: {exc}"
                )


def _merge_imports(
    source: Path,
    target: Path,
    summary: MergeSummary,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:

    source_imports_dir = source / "imports"
    if not source_imports_dir.is_dir():
        return

    for source_import_dir in sorted(source_imports_dir.iterdir()):
        if not source_import_dir.is_dir():
            continue

        target_import_dir = target / "imports" / source_import_dir.name
        try:
            if target_import_dir.exists():
                summary.imports_skipped += 1
                _log_decision(
                    log_path,
                    {
                        "action": "import_skipped",
                        "item_type": "import",
                        "item_id": source_import_dir.name,
                        "reason": "target_exists",
                    },
                )
                continue
            if not dry_run:
                shutil.copytree(
                    source_import_dir,
                    target_import_dir,
                    copy_function=shutil.copy2,
                )
            summary.imports_copied += 1
            _log_decision(
                log_path,
                {
                    "action": "import_copied",
                    "item_type": "import",
                    "item_id": source_import_dir.name,
                    "reason": "new",
                },
            )
        except Exception as exc:
            summary.errors.append(f"import {source_import_dir.name}: {exc}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


__all__ = ["MergeSummary", "merge_journals"]
