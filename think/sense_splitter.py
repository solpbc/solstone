# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Split unified Sense agent output into per-agent file locations."""

import json
from datetime import datetime, timezone
from pathlib import Path


def _write_json_atomic(path: Path, data: object) -> None:
    """Atomically write JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically write text data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_sense_outputs(
    sense_json: dict, seg_dir: Path, stream: str | None = None
) -> None:
    """Write unified Sense output into per-agent files."""
    agents_dir = seg_dir / "talents"

    density = sense_json.get("density") or "active"
    activity_summary = sense_json.get("activity_summary") or ""
    entities = sense_json.get("entities") or []
    facets = sense_json.get("facets") or []
    meeting_detected = bool(sense_json.get("meeting_detected"))
    speakers = sense_json.get("speakers") or []

    _write_text_atomic(agents_dir / "activity.md", activity_summary)
    _write_json_atomic(agents_dir / "facets.json", facets)
    _write_json_atomic(
        agents_dir / "density.json",
        {
            "classification": density,
            "transcript_lines": 0,
            "screen_frames": 0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    # Write both structured and human-readable Sense outputs here.
    # think/cluster.py discovers talent outputs by globbing
    # {segment}/talents/**/*.md for load.talents.{name} consumers.
    # Dropping sense.md would silently break downstream talents such as
    # participation that rely on the sense markdown file being present.
    _write_json_atomic(agents_dir / "sense.json", sense_json)

    if entities:
        lines = ["# Sense Entities", ""]
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            lines.append(
                "- "
                f"{entity.get('type', '')} — {entity.get('name', '')} "
                f"(role={entity.get('role', '')}, source={entity.get('source', '')}) "
                f"— {entity.get('context', '')}"
            )
        if len(lines) > 2:
            _write_text_atomic(agents_dir / "sense.md", "\n".join(lines))

    if meeting_detected:
        _write_json_atomic(agents_dir / "speakers.json", speakers)


def write_idle_stubs(seg_dir: Path) -> None:
    """Write minimal idle output files for a segment."""
    _write_json_atomic(
        seg_dir / "talents" / "density.json",
        {
            "classification": "idle",
            "transcript_lines": 0,
            "screen_frames": 0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
