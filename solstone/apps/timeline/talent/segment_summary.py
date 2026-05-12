# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Timeline segment summary talent hooks."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from solstone.think.utils import day_path, iter_segments, segment_path

logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = (
    "Pick the SINGLE MOST IMPORTANT EVENT from this ~5-minute slice of a "
    "personal life-journal and name it. The output is one cell in a multi-scale "
    "timeline UI — each cell shows a 2-line title and a 3-line description, so "
    "brevity matters more than completeness.\n"
    "\n"
    "An EVENT is a discrete thing that happened: a decision made, a problem "
    "solved, a message sent, a system change applied, a person met, a file "
    "shipped, a milestone reached. NOT a topic area, NOT a feeling, NOT a "
    "generic activity descriptor.\n"
    "\n"
    "Anti-patterns to avoid:\n"
    "  BAD: 'Coding Session' (topic, not event)\n"
    "  BAD: 'Working on KDE' (activity descriptor)\n"
    "  BAD: 'System Maintenance' (generic)\n"
    "  GOOD: 'GDM Service Restart' (specific event)\n"
    "  GOOD: 'Trademark Filed' (discrete action)\n"
    "  GOOD: 'Crash Diagnosed' (concrete result)\n"
    "\n"
    "If multiple noteworthy events occurred, pick the one with the highest "
    "consequence — a decision over a routine action, a fix over an "
    "investigation, a shipped artifact over a draft.\n"
    "\n"
    "FIELD RULES (hard caps):\n"
    "- title: max 3 words, max 22 characters, headline case. Name the EVENT "
    "as a noun phrase or past-tense action. Shorthand is encouraged: Dev, "
    "Env, Config, UI, App, Repo, PR, Bug, Cli, Doc, Auth, K8s, KDE, GDM, "
    "Wallet, Plasma, GNOME. Drop articles. Prefer specific over generic "
    "('KDE Panel Fix' beats 'System Config'). Examples: 'Display Reset', "
    "'Trademark Filed', 'Dev Env Debug', 'Sprint Planned', 'Crash Triage'.\n"
    "- description: max 10 words, max 60 characters, ONE sentence, third "
    "person, present tense, verb-led, describing what happened in/around "
    "the event. Examples: 'Restarts display manager to recover desktop "
    "session.' (51c) 'Files trademark application with specimen images.' "
    "(49c) 'Identifies GNOME DBus dependency causing the crash.' (52c). "
    "No first-person ('I', 'me'). No times ('17:02'). No segment IDs.\n"
    "\n"
    "If the input is empty or trivial, still return a plausible compact "
    "{title, description} for whatever did happen."
)


def origin_for_segment(seg_dir):
    """Composite ID '<day>[/<stream>]/<seg>' relative to the chronicle root."""
    parts = seg_dir.parts
    try:
        ci = parts.index("chronicle")
    except ValueError:
        return seg_dir.name
    return "/".join(parts[ci + 1 :])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _candidate_segment_dirs(day: str, segment: str, stream: str | None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if stream:
        add(segment_path(day, segment, stream, create=False))

    for _stream, segment_id, seg_dir in iter_segments(day):
        if segment_id == segment:
            add(seg_dir)

    add(day_path(day, create=False) / segment)
    return candidates


def _find_activity(seg_dir: Path) -> Path | None:
    for rel in ("talents/activity.md", "activity.md"):
        candidate = seg_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _resolve_activity(
    day: str, segment: str, stream: str | None
) -> tuple[Path, Path] | None:
    for seg_dir in _candidate_segment_dirs(day, segment, stream):
        activity_path = _find_activity(seg_dir)
        if activity_path is not None:
            return seg_dir, activity_path
    return None


def _resolve_segment_dir(day: str, segment: str, stream: str | None) -> Path | None:
    for seg_dir in _candidate_segment_dirs(day, segment, stream):
        if seg_dir.exists():
            return seg_dir
    return None


def pre_process(context: dict) -> dict | None:
    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream")
    if not day or not segment:
        return {"skip_reason": "no_activity_md"}

    resolved = _resolve_activity(
        str(day), str(segment), str(stream) if stream else None
    )
    if resolved is None:
        return {"skip_reason": "no_activity_md"}

    seg_dir, activity_path = resolved
    if (seg_dir / "timeline.json").exists() and not context.get("refresh"):
        return {"skip_reason": "timeline_exists"}

    try:
        activity_text = activity_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "timeline segment summary: failed to read %s: %s", activity_path, exc
        )
        return {"skip_reason": "no_activity_md"}

    return {
        "template_vars": {
            "activity_text": activity_text,
            "segment_rel_path": origin_for_segment(seg_dir),
        }
    }


def post_process(result: str, context: dict) -> str | None:
    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream")
    if not day or not segment:
        return None

    seg_dir = _resolve_segment_dir(
        str(day), str(segment), str(stream) if stream else None
    )
    if seg_dir is None:
        logger.warning(
            "timeline segment summary: could not resolve segment dir for %s/%s",
            day,
            segment,
        )
        return None

    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        logger.warning("timeline segment summary: could not parse result as JSON")
        return None

    if not isinstance(parsed, dict):
        logger.warning("timeline segment summary: result is not a JSON object")
        return None

    payload = {
        "title": parsed.get("title", ""),
        "description": parsed.get("description", ""),
        "origin": origin_for_segment(seg_dir),
        "model": MODEL,
        "generated_at": int(time.time()),
    }
    _atomic_write_json(seg_dir / "timeline.json", payload)
    return None
