# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Segment ingest endpoint for journal source imports."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import abort, g, jsonify, request
from werkzeug.utils import secure_filename

from convey import emit, state
from observe.utils import (
    compute_bytes_sha256,
    compute_file_sha256,
    find_available_segment,
)

from .journal_sources import (
    get_state_directory,
    require_journal_source,
    save_journal_source,
)

logger = logging.getLogger(__name__)

_DAY_RE = re.compile(r"^\d{8}$")
_SEGMENT_RE = re.compile(r"^\d{6}_\d+$")
_STREAM_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _append_decision(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_state_atomic(state_path: Path, state_data: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=state_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state_data, handle, indent=2)
        Path(tmp_path).rename(state_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def register_ingest_routes(bp) -> None:
    @bp.route("/journal/<key_prefix>/ingest/segments", methods=["POST"])
    @require_journal_source
    def ingest_segments(key_prefix: str):
        if g.journal_source["key"][:8] != key_prefix:
            abort(403, description="Key prefix mismatch")

        metadata_raw = request.form.get("metadata")
        if not metadata_raw:
            return jsonify({"error": "Missing metadata"}), 400

        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid metadata JSON"}), 400

        if not isinstance(metadata, dict):
            return jsonify({"error": "Invalid metadata JSON"}), 400

        segments = metadata.get("segments")
        if not isinstance(segments, list):
            return jsonify({"error": "Missing segments array"}), 400

        journal_root = Path(state.journal_root)
        log_path = get_state_directory(key_prefix) / "segments" / "log.jsonl"

        copied = 0
        skipped = 0
        deconflicted = 0
        errors: list[dict[str, str]] = []
        new_state = {}

        for idx, segment in enumerate(segments):
            day = ""
            segment_key = ""
            try:
                if not isinstance(segment, dict):
                    raise ValueError("Segment metadata must be an object")

                day = str(segment.get("day", "")).strip()
                stream = str(segment.get("stream", "")).strip()
                segment_key = str(segment.get("segment_key", "")).strip()
                files = segment.get("files")

                if not _DAY_RE.match(day):
                    raise ValueError("Invalid day format")
                if not _STREAM_RE.match(stream):
                    raise ValueError("Invalid stream format")
                if not _SEGMENT_RE.match(segment_key):
                    raise ValueError("Invalid segment_key format")
                if not isinstance(files, list) or not files:
                    raise ValueError("Segment must list at least one file")

                expected_names = []
                for raw_name in files:
                    name = secure_filename(str(raw_name))
                    if not name:
                        raise ValueError("Invalid filename in metadata")
                    expected_names.append(name)

                if len(set(expected_names)) != len(expected_names):
                    raise ValueError("Duplicate filenames in metadata")

                uploaded_files = request.files.getlist(f"files_{idx}")
                file_infos: dict[str, dict[str, str | int | bytes]] = {}
                for upload in uploaded_files:
                    if not upload.filename:
                        continue
                    filename = secure_filename(upload.filename)
                    if not filename:
                        continue
                    content = upload.read()
                    if len(content) == 0:
                        continue
                    if filename in file_infos:
                        raise ValueError(f"Duplicate uploaded filename: {filename}")
                    file_infos[filename] = {
                        "name": filename,
                        "content": content,
                        "sha256": compute_bytes_sha256(content),
                        "size": len(content),
                    }

                expected_set = set(expected_names)
                uploaded_set = set(file_infos.keys())
                if expected_set != uploaded_set:
                    missing = sorted(expected_set - uploaded_set)
                    unexpected = sorted(uploaded_set - expected_set)
                    parts = []
                    if missing:
                        parts.append(f"Missing uploaded files: {', '.join(missing)}")
                    if unexpected:
                        parts.append(
                            f"Unexpected uploaded files: {', '.join(unexpected)}"
                        )
                    raise ValueError("; ".join(parts))

                original_segment_key = segment_key
                arc_key = f"{stream}/{segment_key}"
                day_dir = journal_root / day
                stream_dir = day_dir / stream
                segment_dir = stream_dir / segment_key
                action = "copied"
                reason = "new segment"

                if segment_dir.exists():
                    exact_match = True
                    for name in expected_names:
                        file_path = segment_dir / name
                        if not file_path.is_file():
                            exact_match = False
                            break
                        if compute_file_sha256(file_path) != file_infos[name]["sha256"]:
                            exact_match = False
                            break

                    if exact_match:
                        action = "skipped"
                        reason = "exact match"
                    else:
                        new_key = find_available_segment(stream_dir, segment_key)
                        if new_key is None:
                            raise ValueError("No available segment slot")
                        segment_key = new_key
                        arc_key = f"{stream}/{segment_key}"
                        segment_dir = stream_dir / segment_key
                        action = "deconflicted"
                        reason = "segment key conflict"

                if action in {"copied", "deconflicted"}:
                    segment_dir.mkdir(parents=True, exist_ok=True)
                    for name in expected_names:
                        (segment_dir / name).write_bytes(file_infos[name]["content"])

                    file_records = [
                        {
                            "name": name,
                            "sha256": str(file_infos[name]["sha256"]),
                            "size": int(file_infos[name]["size"]),
                        }
                        for name in expected_names
                    ]
                    new_state.setdefault(day, {})[arc_key] = {"files": file_records}

                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action": action,
                    "item_type": "segment",
                    "item_id": f"{day}/{arc_key}",
                    "reason": reason,
                    "files": expected_names,
                }
                if action == "deconflicted":
                    entry["original_key"] = original_segment_key
                _append_decision(log_path, entry)

                if action == "copied":
                    copied += 1
                elif action == "skipped":
                    skipped += 1
                else:
                    deconflicted += 1
            except Exception as exc:
                errors.append(
                    {
                        "segment_key": segment_key,
                        "day": day,
                        "error": str(exc),
                    }
                )

        if new_state:
            state_path = get_state_directory(key_prefix) / "segments" / "state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                existing = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}

            for day, segments_for_day in new_state.items():
                existing.setdefault(day, {}).update(segments_for_day)

            _write_state_atomic(state_path, existing)

        written = copied + deconflicted
        if written > 0:
            source = g.journal_source
            source.setdefault("stats", {})
            source["stats"]["segments_received"] = (
                source["stats"].get("segments_received", 0) + written
            )
            save_journal_source(source)

            try:
                emit("supervisor", "request", cmd=["sol", "indexer", "--rescan"])
            except Exception:
                logger.warning("Failed to trigger indexer rescan via Callosum")

        return jsonify(
            {
                "segments_received": written,
                "segments_skipped": skipped,
                "segments_deconflicted": deconflicted,
                "errors": errors,
            }
        )
