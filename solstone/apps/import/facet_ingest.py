# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from solstone.think.entities.observations import load_observations, save_observations
from solstone.think.entities.relationships import (
    load_facet_relationship,
    save_facet_relationship,
)

from .ingest import _append_decision

_DAY_JSONL_RE = re.compile(r"^\d{8}\.jsonl$")
_DAY_MD_RE = re.compile(r"^\d{8}\.md$")
_FACET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENTITY_FILE_TYPES = {
    "entity_relationship",
    "entity_observations",
    "detected_entities",
    "activity_records",
}


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


def _remap_entity_id(entity_id: str, id_map: dict[str, str]) -> str | None:
    return id_map.get(entity_id)


def _parse_path(path_str: str, file_type: str) -> tuple[PurePosixPath, dict[str, str]]:
    path = PurePosixPath(path_str)
    if not path_str or path.is_absolute():
        raise ValueError("Invalid path")

    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Invalid path")

    parts = path.parts

    if file_type == "facet_json":
        if parts != ("facet.json",):
            raise ValueError("facet_json path must be facet.json")
        return path, {}

    if file_type == "entity_relationship":
        if len(parts) != 3 or parts[0] != "entities" or parts[2] != "entity.json":
            raise ValueError(
                "entity_relationship path must be entities/<id>/entity.json"
            )
        return path, {"entity_id": parts[1]}

    if file_type == "entity_observations":
        if (
            len(parts) != 3
            or parts[0] != "entities"
            or parts[2] != "observations.jsonl"
        ):
            raise ValueError(
                "entity_observations path must be entities/<id>/observations.jsonl"
            )
        return path, {"entity_id": parts[1]}

    if file_type == "detected_entities":
        if (
            len(parts) != 2
            or parts[0] != "entities"
            or not _DAY_JSONL_RE.match(parts[1])
        ):
            raise ValueError("detected_entities path must be entities/YYYYMMDD.jsonl")
        return path, {"day_file": parts[1]}

    if file_type == "activity_config":
        if parts != ("activities", "activities.jsonl"):
            raise ValueError("activity_config path must be activities/activities.jsonl")
        return path, {}

    if file_type == "activity_records":
        if (
            len(parts) != 2
            or parts[0] != "activities"
            or not _DAY_JSONL_RE.match(parts[1])
        ):
            raise ValueError("activity_records path must be activities/YYYYMMDD.jsonl")
        return path, {"day_file": parts[1]}

    if file_type == "activity_output":
        if (
            len(parts) < 4
            or parts[0] != "activities"
            or not re.match(r"^\d{8}$", parts[1])
        ):
            raise ValueError(
                "activity_output path must be activities/YYYYMMDD/<activity_id>/..."
            )
        return path, {"output_dir": str(PurePosixPath(*parts[:3]))}

    if file_type == "todos":
        if len(parts) != 2 or parts[0] != "todos" or not _DAY_JSONL_RE.match(parts[1]):
            raise ValueError("todos path must be todos/YYYYMMDD.jsonl")
        return path, {"day_file": parts[1]}

    if file_type == "news":
        if len(parts) != 2 or parts[0] != "news" or not _DAY_MD_RE.match(parts[1]):
            raise ValueError("news path must be news/YYYYMMDD.md")
        return path, {"news_file": parts[1]}

    if file_type == "logs":
        if len(parts) != 2 or parts[0] != "logs" or not _DAY_JSONL_RE.match(parts[1]):
            raise ValueError("logs path must be logs/YYYYMMDD.jsonl")
        return path, {"day_file": parts[1]}

    raise ValueError(f"Unsupported file type: {file_type}")


def _decode_text(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8")


def _parse_json_bytes(raw_bytes: bytes) -> Any:
    return json.loads(_decode_text(raw_bytes))


def _parse_jsonl_bytes(raw_bytes: bytes) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(_decode_text(raw_bytes).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid JSONL at line {line_number}: item must be an object"
            )
        items.append(value)
    return items


def _check_unmapped_entities(
    data: Any,
    id_map: dict[str, str],
    file_type: str,
    path_info: dict[str, str],
) -> list[str]:
    unmapped: list[str] = []

    def add(entity_id: str) -> None:
        if (
            entity_id
            and _remap_entity_id(entity_id, id_map) is None
            and entity_id not in unmapped
        ):
            unmapped.append(entity_id)

    if file_type in {"entity_relationship", "entity_observations"}:
        add(path_info["entity_id"])
    elif file_type == "detected_entities":
        for item in data:
            entity_id = item.get("id")
            if entity_id:
                add(str(entity_id))
    elif file_type == "activity_records":
        for item in data:
            active_entities = item.get("active_entities")
            if not isinstance(active_entities, list):
                continue
            for entity_id in active_entities:
                if entity_id:
                    add(str(entity_id))

    return unmapped


def _sanitize_stage_name(relative_path: str) -> str:
    return relative_path.replace("/", "__") + ".staged.json"


def _stage_unmapped_entity(
    staged_dir: Path,
    facet_name: str,
    file_type: str,
    relative_path: str,
    entity_id: str,
    source_data: str,
) -> Path:
    target_path = (
        staged_dir / facet_name / file_type / _sanitize_stage_name(relative_path)
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": "unmapped_entity",
        "source_entity_id": entity_id,
        "explanation": (
            f"Entity '{entity_id}' has no mapping in entities/state.json id_map"
        ),
        "source_path": relative_path,
        "source_data": source_data,
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def _stage_facet_json_conflict(
    staged_dir: Path,
    facet_name: str,
    relative_path: str,
    source_content: Any,
    target_content: Any,
) -> Path:
    target_path = (
        staged_dir / facet_name / "facet_json" / _sanitize_stage_name(relative_path)
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": "facet_json_conflict",
        "source_content": source_content,
        "target_content": target_content,
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def _write_bytes(path: Path, raw_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)


def _merge_facet_json(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
    staged_dir: Path,
    facet_name: str,
    relative_path: str,
) -> dict[str, Any]:
    source_content = _parse_json_bytes(raw_bytes)
    if not target_path.exists() or new_facet:
        _write_bytes(target_path, raw_bytes)
        return {
            "status": "written",
            "reason": "new_facet" if new_facet else "overlap_merged",
        }

    target_content = json.loads(target_path.read_text(encoding="utf-8"))
    if target_content == source_content:
        return {"status": "skipped", "reason": "facet_json_match"}

    staged_path = _stage_facet_json_conflict(
        staged_dir, facet_name, relative_path, source_content, target_content
    )
    return {
        "status": "staged",
        "reason": "facet_json_conflict",
        "staged_path": str(staged_path),
    }


def _merge_entity_relationship(
    facet_name: str,
    entity_id: str,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_relationship = _parse_json_bytes(raw_bytes)
    if not isinstance(source_relationship, dict):
        raise ValueError("entity_relationship content must be a JSON object")
    source_relationship["entity_id"] = entity_id

    target_relationship = {}
    if not new_facet:
        loaded = load_facet_relationship(facet_name, entity_id)
        if loaded is not None:
            target_relationship = loaded

    merged_relationship = {**source_relationship, **target_relationship}
    save_facet_relationship(facet_name, entity_id, merged_relationship)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_observations(
    facet_name: str,
    entity_id: str,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_observations = _parse_jsonl_bytes(raw_bytes)
    target_observations = [] if new_facet else load_observations(facet_name, entity_id)
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

    save_observations(facet_name, entity_id, merged_observations)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_detected_entities(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_items = _parse_jsonl_bytes(raw_bytes)
    target_items = [] if new_facet else _read_jsonl(target_path)
    seen_ids = {item.get("id") for item in target_items if item.get("id")}
    new_items = []
    for item in source_items:
        item_id = item.get("id", "")
        if item_id in seen_ids:
            continue
        new_items.append(item)
    _append_jsonl(target_path, new_items)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_activity_config(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_items = _parse_jsonl_bytes(raw_bytes)
    target_items = [] if new_facet else _read_jsonl(target_path)
    existing_ids = {item.get("id") for item in target_items}
    new_items = [item for item in source_items if item.get("id") not in existing_ids]
    _append_jsonl(target_path, new_items)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_activity_records(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_items = _parse_jsonl_bytes(raw_bytes)
    target_items = [] if new_facet else _read_jsonl(target_path)
    existing_ids = {item.get("id") for item in target_items}
    new_items = [item for item in source_items if item.get("id") not in existing_ids]
    _append_jsonl(target_path, new_items)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_activity_output(
    target_path: Path,
    raw_bytes: bytes,
    output_dir: Path,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        return {"status": "skipped", "reason": "output_dir_exists"}
    _write_bytes(target_path, raw_bytes)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_todos(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_items = _parse_jsonl_bytes(raw_bytes)
    target_items = [] if new_facet else _read_jsonl(target_path)
    seen = {(item["text"], item.get("created_at")) for item in target_items}
    new_items = [
        item
        for item in source_items
        if (item["text"], item.get("created_at")) not in seen
    ]
    _append_jsonl(target_path, new_items)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_news(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    if target_path.exists():
        return {"status": "skipped", "reason": "news_exists"}
    _write_bytes(target_path, raw_bytes)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _merge_logs(
    target_path: Path,
    raw_bytes: bytes,
    *,
    new_facet: bool,
) -> dict[str, Any]:
    source_items = _parse_jsonl_bytes(raw_bytes)
    _append_jsonl(target_path, source_items)
    return {
        "status": "written",
        "reason": "new_facet" if new_facet else "overlap_merged",
    }


def _remap_entity_ids(
    data: Any,
    id_map: dict[str, str],
    file_type: str,
    path_info: dict[str, str],
) -> tuple[Any, dict[str, str]]:
    updated_path_info = dict(path_info)

    if file_type == "entity_relationship":
        entity_id = path_info["entity_id"]
        mapped_id = _remap_entity_id(entity_id, id_map)
        if mapped_id is None:
            raise ValueError(f"Unmapped entity id: {entity_id}")
        if not isinstance(data, dict):
            raise ValueError("entity_relationship content must be a JSON object")
        updated_path_info["entity_id"] = mapped_id
        remapped = dict(data)
        remapped["entity_id"] = mapped_id
        return remapped, updated_path_info

    if file_type == "entity_observations":
        entity_id = path_info["entity_id"]
        mapped_id = _remap_entity_id(entity_id, id_map)
        if mapped_id is None:
            raise ValueError(f"Unmapped entity id: {entity_id}")
        updated_path_info["entity_id"] = mapped_id
        return data, updated_path_info

    if file_type == "detected_entities":
        remapped_items = []
        for item in data:
            updated = dict(item)
            entity_id = updated.get("id")
            if entity_id:
                mapped_id = _remap_entity_id(str(entity_id), id_map)
                if mapped_id is None:
                    raise ValueError(f"Unmapped entity id: {entity_id}")
                updated["id"] = mapped_id
            remapped_items.append(updated)
        return remapped_items, updated_path_info

    if file_type == "activity_records":
        remapped_items = []
        for item in data:
            updated = dict(item)
            active_entities = updated.get("active_entities")
            if isinstance(active_entities, list):
                remapped_entities = []
                for entity_id in active_entities:
                    mapped_id = _remap_entity_id(str(entity_id), id_map)
                    if mapped_id is None:
                        raise ValueError(f"Unmapped entity id: {entity_id}")
                    remapped_entities.append(mapped_id)
                updated["active_entities"] = remapped_entities
            remapped_items.append(updated)
        return remapped_items, updated_path_info

    return data, updated_path_info


def _serialize_jsonl(items: list[dict[str, Any]]) -> bytes:
    if not items:
        return b""
    return "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in items
    ).encode("utf-8")


def process_facet(
    facet_name: str,
    files: list[dict],
    file_data: list[bytes],
    journal_root: Path,
    id_map: dict[str, str],
    log_path: Path,
    staged_dir: Path,
    received: dict[str, str],
) -> dict:
    if not _FACET_NAME_RE.match(facet_name):
        raise ValueError("Invalid facet name")

    facet_dir = journal_root / "facets" / facet_name
    new_facet = not facet_dir.exists()

    result = {
        "created": 0,
        "merged": 0,
        "skipped": 0,
        "staged": 0,
        "errors": [],
        "wrote_files": False,
    }

    for metadata, raw_bytes in zip(files, file_data, strict=True):
        raw_path = str(metadata.get("path", "")).strip()
        file_type = str(metadata.get("type", "")).strip()

        try:
            normalized_path, path_info = _parse_path(raw_path, file_type)
            relative_path = normalized_path.as_posix()
            item_id = f"{facet_name}/{relative_path}"
            content_hash = hashlib.sha256(raw_bytes).hexdigest()

            if received.get(item_id) == content_hash:
                result["skipped"] += 1
                _append_decision(
                    log_path,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "action": "facet_file_skipped",
                        "item_type": file_type,
                        "item_id": item_id,
                        "facet": facet_name,
                        "reason": "idempotent",
                    },
                )
                continue

            parsed_data: Any = raw_bytes
            if file_type == "facet_json":
                parsed_data = _parse_json_bytes(raw_bytes)
            elif file_type in {
                "entity_observations",
                "detected_entities",
                "activity_config",
                "activity_records",
                "todos",
                "logs",
            }:
                parsed_data = _parse_jsonl_bytes(raw_bytes)
            elif file_type == "entity_relationship":
                parsed_data = _parse_json_bytes(raw_bytes)

            if file_type in _ENTITY_FILE_TYPES:
                unmapped = _check_unmapped_entities(
                    parsed_data, id_map, file_type, path_info
                )
                if unmapped:
                    staged_path = _stage_unmapped_entity(
                        staged_dir,
                        facet_name,
                        file_type,
                        relative_path,
                        unmapped[0],
                        _decode_text(raw_bytes),
                    )
                    result["staged"] += 1
                    _append_decision(
                        log_path,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "action": "facet_file_staged",
                            "item_type": file_type,
                            "item_id": item_id,
                            "facet": facet_name,
                            "reason": "unmapped_entity",
                            "staged_path": str(staged_path),
                        },
                    )
                    received[item_id] = content_hash
                    continue

                parsed_data, path_info = _remap_entity_ids(
                    parsed_data, id_map, file_type, path_info
                )
                if file_type == "entity_relationship":
                    raw_bytes = (
                        json.dumps(parsed_data, ensure_ascii=False, indent=2) + "\n"
                    ).encode("utf-8")
                elif file_type in {"detected_entities", "activity_records"}:
                    raw_bytes = _serialize_jsonl(parsed_data)

            target_path = facet_dir / normalized_path

            if file_type == "facet_json":
                merge_result = _merge_facet_json(
                    target_path,
                    raw_bytes,
                    new_facet=new_facet,
                    staged_dir=staged_dir,
                    facet_name=facet_name,
                    relative_path=relative_path,
                )
            elif file_type == "entity_relationship":
                merge_result = _merge_entity_relationship(
                    facet_name,
                    path_info["entity_id"],
                    raw_bytes,
                    new_facet=new_facet,
                )
            elif file_type == "entity_observations":
                merge_result = _merge_observations(
                    facet_name,
                    path_info["entity_id"],
                    raw_bytes,
                    new_facet=new_facet,
                )
            elif file_type == "detected_entities":
                merge_result = _merge_detected_entities(
                    target_path, raw_bytes, new_facet=new_facet
                )
            elif file_type == "activity_config":
                merge_result = _merge_activity_config(
                    target_path, raw_bytes, new_facet=new_facet
                )
            elif file_type == "activity_records":
                merge_result = _merge_activity_records(
                    target_path, raw_bytes, new_facet=new_facet
                )
            elif file_type == "activity_output":
                output_dir = facet_dir / PurePosixPath(path_info["output_dir"])
                merge_result = _merge_activity_output(
                    target_path,
                    raw_bytes,
                    output_dir,
                    new_facet=new_facet,
                )
            elif file_type == "todos":
                merge_result = _merge_todos(target_path, raw_bytes, new_facet=new_facet)
            elif file_type == "news":
                merge_result = _merge_news(target_path, raw_bytes, new_facet=new_facet)
            elif file_type == "logs":
                merge_result = _merge_logs(target_path, raw_bytes, new_facet=new_facet)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            status = merge_result["status"]
            if status == "written":
                received[item_id] = content_hash
                bucket = "created" if new_facet else "merged"
                result[bucket] += 1
                result["wrote_files"] = True
                action = "facet_file_created" if new_facet else "facet_file_merged"
            elif status == "staged":
                received[item_id] = content_hash
                result["staged"] += 1
                action = "facet_file_staged"
            else:
                received[item_id] = content_hash
                result["skipped"] += 1
                action = "facet_file_skipped"

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "item_type": file_type,
                "item_id": item_id,
                "facet": facet_name,
                "reason": merge_result["reason"],
            }
            if "staged_path" in merge_result:
                entry["staged_path"] = merge_result["staged_path"]
            _append_decision(log_path, entry)
        except Exception as exc:
            _append_decision(
                log_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action": "facet_file_error",
                    "item_type": file_type,
                    "item_id": f"{facet_name}/{raw_path}",
                    "facet": facet_name,
                    "reason": str(exc),
                },
            )
            result["errors"].append(
                {
                    "facet": facet_name,
                    "path": raw_path,
                    "error": str(exc),
                }
            )

    return result


__all__ = ["process_facet"]
