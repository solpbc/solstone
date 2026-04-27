# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Talents app - browse historical talent uses by day and facet."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey import state
from convey.utils import DATE_RE, format_date
from think.facets import get_facets
from think.models import calc_agent_cost
from think.talent import get_output_path, get_talent_configs
from think.utils import resolve_journal_path, updated_days

sol_bp = Blueprint(
    "app:sol",
    __name__,
    url_prefix="/app/sol",
)


def _resolve_output_path(
    request_event: dict[str, Any], journal_root: str
) -> Path | None:
    """Resolve output file path from an agent request event.

    Uses explicit output_path if present, otherwise derives from
    request fields via get_output_path.

    Returns absolute Path or None if not resolvable.
    """
    # Prefer explicit output_path (set for activity agents, custom paths)
    if request_event.get("output_path"):
        return Path(request_event["output_path"])

    # Derive from request fields
    req_day = request_event.get("day")
    if not req_day:
        return None
    day_dir = Path(journal_root) / req_day
    req_segment = request_event.get("segment")
    req_facet = request_event.get("facet")
    req_name = request_event["name"]
    req_env = request_event.get("env") or {}
    req_stream = req_env.get("SOL_STREAM") if req_env else None
    return get_output_path(
        day_dir,
        req_name,
        segment=req_segment,
        output_format=request_event.get("output"),
        facet=req_facet,
        stream=req_stream,
    )


def _get_facet_filter() -> str | None:
    """Get facet filter from query param or cookie.

    Returns None for all-facet mode.
    """
    facet = request.args.get("facet")
    if facet is None:
        facet = request.cookies.get("selectedFacet") or None
    return facet


def _use_id_to_day(use_id: str) -> str:
    """Convert use_id (millisecond timestamp) to YYYYMMDD day string."""
    try:
        ts = int(use_id) / 1000
        return datetime.fromtimestamp(ts).strftime("%Y%m%d")
    except (ValueError, OSError):
        return ""


def _parse_use_events(
    lines: list[str], *, collect_events: bool = False
) -> dict[str, Any]:
    """Parse use event lines and extract counts and cost data.

    Args:
        lines: List of JSONL lines
        collect_events: If True, include parsed event dicts as "events" key

    Returns:
        Dict with: thinking_count, tool_count, model, provider, usage, finish_ts,
        error_ts, error_message, and optionally events
    """
    result: dict[str, Any] = {
        "thinking_count": 0,
        "tool_count": 0,
        "model": None,
        "provider": None,
        "usage": None,
        "finish_ts": None,
        "error_ts": None,
        "error_message": None,
    }
    events: list[dict] = [] if collect_events else None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if events is not None:
                # Strip bulky provider-native data not used by the frontend
                event.pop("raw", None)
                events.append(event)
            event_type = event.get("event")
            if event_type == "thinking":
                result["thinking_count"] += 1
            elif event_type == "tool_start":
                result["tool_count"] += 1
            elif event_type == "start":
                result["model"] = event.get("model")
                result["provider"] = event.get("provider")
            elif event_type == "finish":
                result["finish_ts"] = event.get("ts", 0)
                result["usage"] = event.get("usage")
            elif event_type == "error":
                result["error_ts"] = event.get("ts", 0)
                msg = event.get("error", "")
                if msg:
                    result["error_message"] = msg[:200]
        except json.JSONDecodeError:
            continue

    if events is not None:
        result["events"] = events

    return result


def _parse_use_file(use_file: Path) -> dict[str, Any] | None:
    """Parse a use JSONL file and extract metadata.

    Returns dict with: id, name, start, status, prompt, facet, failed,
    runtime_seconds, thinking_count, tool_count, cost, model, provider,
    error_message.
    Returns None if file cannot be parsed.
    """
    from think.cortex_client import get_use_end_state

    try:
        with open(use_file, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        first_line = lines[0].strip()
        if not first_line:
            return None

        request_event = json.loads(first_line)
        if request_event.get("event") != "request":
            return None

        is_active = "_active.jsonl" in use_file.name
        use_id = use_file.stem.replace("_active", "")

        # Parse events using shared helper
        event_data = _parse_use_events(lines[1:])

        use_info: dict[str, Any] = {
            "id": use_id,
            "name": request_event["name"],
            "start": request_event.get("ts", 0),
            "status": "running" if is_active else "completed",
            "prompt": request_event.get("prompt", ""),
            "facet": request_event.get("facet"),
            "failed": False,
            "runtime_seconds": None,
            "thinking_count": event_data["thinking_count"],
            "tool_count": event_data["tool_count"],
            "cost": None,
            "model": event_data["model"],
            "provider": request_event.get("provider") or event_data.get("provider"),
            "error_message": event_data["error_message"],
        }

        # Check for output file (generators only)
        output_file = None
        req_output = request_event.get("output")
        if req_output:
            out_path = _resolve_output_path(request_event, state.journal_root)
            if out_path and out_path.exists():
                req_day = request_event.get("day")
                day_dir = Path(state.journal_root) / req_day if req_day else None
                if day_dir and out_path.is_relative_to(day_dir):
                    output_file = str(out_path.relative_to(day_dir))
                else:
                    output_file = str(out_path.relative_to(state.journal_root))
        use_info["output_file"] = output_file

        # For completed uses, determine end state and calculate cost
        if not is_active:
            end_state = get_use_end_state(use_id)
            use_info["failed"] = end_state in ("error", "unknown")

            # Calculate runtime from finish or error timestamp
            end_ts = event_data["finish_ts"] or event_data["error_ts"]
            if end_ts and use_info["start"]:
                use_info["runtime_seconds"] = (end_ts - use_info["start"]) / 1000.0

            # Calculate cost
            use_info["cost"] = calc_agent_cost(event_data["model"], event_data["usage"])

        return use_info
    except (json.JSONDecodeError, IOError):
        return None


def _get_use_day(use_file: Path) -> str:
    """Get the logical day for a use from its request event.

    Prefers the ``day`` field from the request event (the day being processed)
    over the use_id timestamp (when the agent actually ran).  This ensures
    overnight think uses appear under the day they processed.
    """
    use_id = use_file.stem.replace("_active", "")
    try:
        with open(use_file, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                request_event = json.loads(first_line)
                req_day = request_event.get("day")
                if req_day:
                    return req_day
    except (json.JSONDecodeError, IOError):
        pass
    return _use_id_to_day(use_id)


def _get_uses_for_day(day: str, facet_filter: str | None = None) -> list[dict]:
    """Get all talent uses for a specific day.

    Uses the day index file for fast lookup instead of scanning all use files.

    Args:
        day: YYYYMMDD day string
        facet_filter: Optional facet to filter by (None = all facets)

    Returns:
        List of use info dicts sorted by start time (newest first)
    """
    talents_dir = Path(state.journal_root) / "talents"
    if not talents_dir.exists():
        return []

    uses = []

    # Read day index for completed uses
    day_index_path = talents_dir / f"{day}.jsonl"
    if day_index_path.exists():
        try:
            with open(day_index_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by facet if specified
                    if facet_filter is not None and entry.get("facet") != facet_filter:
                        continue

                    # Locate the actual file for full parsing
                    use_id = entry.get("use_id", "")
                    name = entry["name"]
                    safe_name = name.replace(":", "--")
                    use_file = talents_dir / safe_name / f"{use_id}.jsonl"
                    if not use_file.exists():
                        continue

                    use_info = _parse_use_file(use_file)
                    if use_info:
                        uses.append(use_info)
        except OSError as exc:
            logging.warning("Failed to read use day index %s: %s", day_index_path, exc)

    # Also check for running uses (only have _active files, no day index entry yet)
    for use_file in talents_dir.glob("*/*_active.jsonl"):
        if "_pending" in use_file.name:
            continue
        if _get_use_day(use_file) != day:
            continue

        use_info = _parse_use_file(use_file)
        if not use_info:
            continue

        if facet_filter is not None and use_info.get("facet") != facet_filter:
            continue

        uses.append(use_info)

    # Sort by start time (newest first)
    uses.sort(key=lambda x: x["start"], reverse=True)
    return uses


@lru_cache(maxsize=1)
def _build_talents_meta() -> dict[str, dict[str, Any]]:
    """Build talent metadata dict from all talent configs.

    Returns dict mapping talent name to metadata with capability fields
    for frontend display. Cached for process lifetime since talent configs
    are static.
    """
    configs = get_talent_configs(include_disabled=True)
    talents: dict[str, dict[str, Any]] = {}

    for name, config in configs.items():
        talents[name] = {
            "title": config.get("title", name),
            "description": config.get("description"),
            "color": config.get("color", "#6c757d"),
            "source": config.get("source", "system"),
            "app": config.get("app"),
            "schedule": config.get("schedule"),
            "type": config.get("type"),
            "output_format": config.get("output"),
            "multi_facet": bool(config.get("multi_facet")),
        }

    return talents


# =============================================================================
# Page Routes
# =============================================================================


@sol_bp.route("/")
def index() -> Any:
    """Redirect to today's talent history."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:sol.talents_day", day=today))


@sol_bp.route("/<day>")
def talents_day(day: str) -> str:
    """Render talent history viewer for a specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    title = format_date(day)

    return render_template("app.html", title=title)


# =============================================================================
# API Routes
# =============================================================================


@sol_bp.route("/api/talents/<day>")
def api_talents_day(day: str) -> Any:
    """Get talent uses and metadata for a specific day.

    Returns flat data for frontend grouping/rendering.

    Query params:
        facet: Optional facet filter (from cookie if not specified)

    Returns:
        {
            "uses": [use objects...],
            "talents": {name: metadata...},
            "facets": {name: {title, color}...}
        }
    """
    if not DATE_RE.fullmatch(day):
        return jsonify({"error": "Invalid day format"}), 400

    facet_filter = _get_facet_filter()

    uses = _get_uses_for_day(day, facet_filter)
    talents = _build_talents_meta()
    facets = {
        name: {"title": f.get("title", name), "color": f.get("color")}
        for name, f in get_facets().items()
    }

    return jsonify(
        {
            "uses": uses,
            "talents": talents,
            "facets": facets,
        }
    )


@sol_bp.route("/api/run/<use_id>")
def api_agent_run(use_id: str) -> Any:
    """Return full talent-use detail with metadata and parsed events."""
    # Locate the use JSONL file
    journal_path = Path(state.journal_root)
    talents_dir = journal_path / "talents"
    # Search subdirectories for the use file
    use_file = None
    for match in talents_dir.glob(f"*/{use_id}.jsonl"):
        use_file = match
        break

    if not use_file:
        for match in talents_dir.glob(f"*/{use_id}_active.jsonl"):
            return jsonify({"error": "talent run is still in progress"}), 202
        return jsonify({"error": f"talent run {use_id} not found"}), 404

    try:
        from think.cortex_client import get_use_end_state

        with open(use_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return jsonify({"error": f"talent run {use_id} is malformed"}), 500

        first_line = lines[0].strip()
        if not first_line:
            return jsonify({"error": f"talent run {use_id} is malformed"}), 500

        request_event = json.loads(first_line)
        if request_event.get("event") != "request":
            return jsonify({"error": f"talent run {use_id} is malformed"}), 500

        event_data = _parse_use_events(lines[1:], collect_events=True)

        output_file = None
        req_output = request_event.get("output")
        if req_output:
            out_path = _resolve_output_path(request_event, state.journal_root)
            if out_path and out_path.exists():
                req_day = request_event.get("day")
                day_dir = Path(state.journal_root) / req_day if req_day else None
                if day_dir and out_path.is_relative_to(day_dir):
                    output_file = str(out_path.relative_to(day_dir))
                else:
                    output_file = str(out_path.relative_to(state.journal_root))

        start_ts = request_event.get("ts", 0)
        runtime_seconds = None
        end_ts = event_data["finish_ts"] or event_data["error_ts"]
        if end_ts and start_ts:
            runtime_seconds = (end_ts - start_ts) / 1000.0

        end_state = get_use_end_state(use_id)

        run: dict[str, Any] = {
            "id": use_id,
            "name": request_event["name"],
            "start": start_ts,
            "status": "completed",
            "prompt": request_event.get("prompt", ""),
            "facet": request_event.get("facet"),
            "failed": end_state in ("error", "unknown"),
            "runtime_seconds": runtime_seconds,
            "thinking_count": event_data["thinking_count"],
            "tool_count": event_data["tool_count"],
            "cost": calc_agent_cost(event_data["model"], event_data["usage"]),
            "model": event_data["model"],
            "provider": request_event.get("provider") or event_data.get("provider"),
            "error_message": event_data["error_message"],
            "output_file": output_file,
            "events": event_data.get("events", []),
        }
        run["day"] = request_event.get("day") or _use_id_to_day(use_id)
        return jsonify(run)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sol_bp.route("/api/output/<day>/<path:filename>")
def api_output_file(day: str, filename: str) -> Any:
    """Serve output file content for the run detail output tab.

    Returns JSON with content, format, and filename.
    Path is validated to stay within the journal directory.

    Supports two path styles:
    - Day-relative: ``talents/flow.md`` → resolved under ``{day}/``
    - Journal-relative: ``facets/work/activities/...`` → resolved under journal root
    """
    if not DATE_RE.fullmatch(day):
        return jsonify(error="Invalid day format"), 400

    journal_root = Path(state.journal_root).resolve()

    # Journal-relative paths (e.g., activity output under facets/)
    if filename.startswith("facets/"):
        file_path = (journal_root / filename).resolve()
    else:
        try:
            file_path = resolve_journal_path(
                journal_root, f"{day}/{filename}"
            ).resolve()
        except ValueError:
            return jsonify(error="Invalid path"), 403

    # Security: ensure path is within the journal directory
    try:
        file_path.relative_to(journal_root)
    except ValueError:
        return jsonify(error="Invalid path"), 403

    if not file_path.is_file():
        return jsonify(error="File not found"), 404

    ext = file_path.suffix.lower()
    fmt = "json" if ext == ".json" else "md"

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError:
        return jsonify(error="Could not read file"), 500

    return jsonify(content=content, format=fmt, filename=file_path.name)


@sol_bp.route("/api/preview/<path:name>")
def api_preview_prompt(name: str) -> Any:
    """Return the complete rendered prompt for an agent.

    Returns:
        {
            "name": str,
            "title": str,
            "full_prompt": str,
            "multi_facet": bool
        }
    """
    try:
        from think.talent import get_talent

        config = get_talent(name)

        system_instruction = config.get("system_instruction", "")
        extra_context = config.get("extra_context", "")
        user_instruction = config.get("user_instruction", "")
        # Compose full prompt with labeled sections
        labeled = []
        if system_instruction:
            labeled.append(f"## System Instruction\n\n{system_instruction}")
        if extra_context:
            labeled.append(f"## Context\n\n{extra_context}")
        if user_instruction:
            labeled.append(f"## Instructions\n\n{user_instruction}")
        full_prompt = "\n\n".join(labeled)

        return jsonify(
            {
                "name": name,
                "title": config.get("title", name),
                "full_prompt": full_prompt,
                "multi_facet": config.get("multi_facet", False),
            }
        )
    except FileNotFoundError:
        return jsonify({"error": f"Talent '{name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sol_bp.route("/api/stats/<month>")
def api_stats(month: str) -> Any:
    """Return talent-use counts per day per facet for a month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to {facet: count, ...}
        For unfaceted runs, uses "_none" as the key.
    """
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    talents_dir = Path(state.journal_root) / "talents"
    if not talents_dir.exists():
        return jsonify({})

    stats: dict[str, dict[str, int]] = {}

    # Read day index files for the month
    for day_index_file in talents_dir.glob(f"{month}*.jsonl"):
        day = day_index_file.stem
        if not re.fullmatch(r"\d{8}", day):
            continue

        try:
            with open(day_index_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    facet = entry.get("facet") or "_none"
                    if day not in stats:
                        stats[day] = {}
                    stats[day][facet] = stats[day].get(facet, 0) + 1
        except IOError:
            continue

    return jsonify(stats)


@sol_bp.route("/api/badge-count")
def api_badge_count() -> Any:
    """Get count of failed talent runs for today (for app icon badge)."""
    today = date.today().strftime("%Y%m%d")
    uses = _get_uses_for_day(today, facet_filter=None)
    failed_count = sum(1 for a in uses if a.get("failed"))
    return jsonify({"count": failed_count})


@sol_bp.route("/api/updated-days")
def api_updated_days() -> Any:
    """Return journal days with pending reprocessing."""
    today = date.today().strftime("%Y%m%d")
    try:
        return jsonify(updated_days(exclude={today}))
    except Exception:
        logging.exception("api_updated_days failed")
        return jsonify({"error": "Unable to load updated days"}), 500


@sol_bp.route("/api/identity")
def api_identity() -> Any:
    """Return talent identity and thickness signals."""
    try:
        from think.awareness import compute_thickness
        from think.utils import get_config

        config = get_config()
        agent = config.get("agent", {})
        identity = config.get("identity", {})
        thickness = compute_thickness()

        return jsonify(
            {
                "agent": agent,
                "identity": identity,
                "thickness": thickness,
            }
        )
    except Exception:
        return jsonify({"error": "Unable to load identity data"}), 500
