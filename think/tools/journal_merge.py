# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Journal merge engine - one-shot merge of a source journal into the target."""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

from think.entities.core import entity_slug
from think.entities.journal import (
    load_all_journal_entities,
    save_journal_entity,
)
from think.entities.matching import find_matching_entity
from think.entities.observations import save_observations
from think.entities.relationships import save_facet_relationship
from think.utils import get_journal, iter_segments

DATE_RE = re.compile(r"^\d{8}$")


@dataclass
class MergeSummary:
    segments_copied: int = 0
    segments_skipped: int = 0
    segments_errored: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    entities_skipped: int = 0
    facets_created: int = 0
    facets_merged: int = 0
    imports_copied: int = 0
    imports_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def merge_journals(source: Path, dry_run: bool = False) -> MergeSummary:
    target = Path(get_journal())
    summary = MergeSummary()
    target_entities = load_all_journal_entities()

    _merge_segments(source, target, summary, dry_run)
    _merge_entities(source, summary, dry_run, target_entities)
    _merge_facets(source, target, summary, dry_run)
    _merge_imports(source, target, summary, dry_run)

    return summary


def _source_day_dirs(source: Path) -> dict[str, Path]:
    days: dict[str, Path] = {}
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and DATE_RE.match(entry.name):
            days[entry.name] = entry
    return days


def _merge_segments(
    source: Path, target: Path, summary: MergeSummary, dry_run: bool
) -> None:
    for day_name, source_day in sorted(_source_day_dirs(source).items()):
        target_day = target / day_name
        for stream, seg_key, seg_path in iter_segments(source_day):
            if stream == "_default":
                target_path = target_day / seg_key
            else:
                target_path = target_day / stream / seg_key

            try:
                if target_path.exists():
                    summary.segments_skipped += 1
                    continue

                typer.echo(f"  Segment {day_name}/{stream}/{seg_key}", err=True)
                if dry_run:
                    summary.segments_copied += 1
                    continue

                shutil.copytree(seg_path, target_path, copy_function=shutil.copy2)
                summary.segments_copied += 1
            except Exception as exc:
                summary.segments_errored += 1
                summary.errors.append(f"segment {day_name}/{stream}/{seg_key}: {exc}")


def _merge_entities(
    source: Path,
    summary: MergeSummary,
    dry_run: bool,
    target_entities: dict[str, dict[str, Any]],
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

            typer.echo(f"  Entity {entity_id}", err=True)
            match = find_matching_entity(source_name, list(target_entities.values()))
            if match is None:
                base_id = entity_id
                next_id = base_id
                suffix = 2
                while next_id in target_entities:
                    next_id = f"{base_id}_{suffix}"
                    suffix += 1
                source_entity["id"] = next_id

                if source_entity.get("is_principal") and target_has_principal:
                    source_entity["is_principal"] = False
                elif source_entity.get("is_principal"):
                    target_has_principal = True

                if not dry_run:
                    save_journal_entity(source_entity)
                summary.entities_created += 1
                target_entities[source_entity["id"]] = source_entity
                continue

            target_id = str(match.get("id", ""))
            if not target_id:
                raise ValueError("matched target entity missing id")

            target_entity = dict(target_entities.get(target_id, match))

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
        except Exception as exc:
            summary.errors.append(f"entity {entity_dir.name}: {exc}")


def _merge_facets(
    source: Path, target: Path, summary: MergeSummary, dry_run: bool
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
            typer.echo(f"  Facet {facet_name}", err=True)
            if not target_facet_dir.exists():
                if not dry_run:
                    shutil.copytree(
                        source_facet_dir,
                        target_facet_dir,
                        copy_function=shutil.copy2,
                    )
                summary.facets_created += 1
                continue

            _merge_overlapping_facet(
                facet_name,
                source_facet_dir,
                target_facet_dir,
                summary,
                dry_run,
            )
            summary.facets_merged += 1
        except Exception as exc:
            summary.errors.append(f"facet {facet_name}: {exc}")


def _merge_overlapping_facet(
    facet_name: str,
    source_facet_dir: Path,
    target_facet_dir: Path,
    summary: MergeSummary,
    dry_run: bool,
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
                    continue

                if not dry_run:
                    shutil.copytree(
                        source_entity_dir,
                        target_entity_dir,
                        copy_function=shutil.copy2,
                    )
            except Exception as exc:
                summary.errors.append(f"facet {facet_name} entity {entity_id}: {exc}")

    source_todos_dir = source_facet_dir / "todos"
    if source_todos_dir.is_dir():
        for source_todo_file in sorted(source_todos_dir.glob("*.jsonl")):
            try:
                target_todo_file = target_facet_dir / "todos" / source_todo_file.name
                target_items = _read_jsonl(target_todo_file)
                seen = {(item["text"], item.get("created_at")) for item in target_items}
                new_items = [
                    item
                    for item in _read_jsonl(source_todo_file)
                    if (item["text"], item.get("created_at")) not in seen
                ]
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
                new_items = [
                    item
                    for item in _read_jsonl(source_calendar_file)
                    if (item["title"], item.get("start")) not in seen
                ]
                if new_items and not dry_run:
                    _append_jsonl(target_calendar_file, new_items)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} calendar {source_calendar_file.name}: {exc}"
                )

    source_news_dir = source_facet_dir / "news"
    if source_news_dir.is_dir():
        target_news_dir = target_facet_dir / "news"
        for source_news_file in sorted(source_news_dir.glob("*.md")):
            try:
                target_news_file = target_news_dir / source_news_file.name
                if target_news_file.exists():
                    continue
                if not dry_run:
                    target_news_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_news_file, target_news_file)
            except Exception as exc:
                summary.errors.append(
                    f"facet {facet_name} news {source_news_file.name}: {exc}"
                )


def _merge_imports(
    source: Path, target: Path, summary: MergeSummary, dry_run: bool
) -> None:
    source_imports_dir = source / "imports"
    if not source_imports_dir.is_dir():
        return

    for source_import_dir in sorted(source_imports_dir.iterdir()):
        if not source_import_dir.is_dir():
            continue

        target_import_dir = target / "imports" / source_import_dir.name
        try:
            typer.echo(f"  Import {source_import_dir.name}", err=True)
            if target_import_dir.exists():
                summary.imports_skipped += 1
                continue
            if not dry_run:
                shutil.copytree(
                    source_import_dir,
                    target_import_dir,
                    copy_function=shutil.copy2,
                )
            summary.imports_copied += 1
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
