# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Agents app - browse historical agent runs by day and facet."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey import state
from convey.utils import DATE_RE, format_date
from think.facets import get_facets
from think.utils import get_agents

agents_bp = Blueprint(
    "app:agents",
    __name__,
    url_prefix="/app/agents",
)


def _get_facet_filter() -> str | None:
    """Get facet filter from query param or cookie.

    Returns None for all-facet mode.
    """
    facet = request.args.get("facet")
    if facet is None:
        facet = request.cookies.get("selectedFacet") or None
    return facet


def _agent_id_to_day(agent_id: str) -> str:
    """Convert agent_id (millisecond timestamp) to YYYYMMDD day string."""
    try:
        ts = int(agent_id) / 1000
        return datetime.fromtimestamp(ts).strftime("%Y%m%d")
    except (ValueError, OSError):
        return ""


def _parse_agent_file(agent_file: Path) -> dict[str, Any] | None:
    """Parse agent JSONL file and extract metadata.

    Returns dict with: id, persona, start, status, prompt, facet, failed,
    runtime_seconds, thinking_count, tool_count, cost.
    Returns None if file cannot be parsed.
    """
    from muse.cortex_client import get_agent_end_state

    try:
        with open(agent_file, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        first_line = lines[0].strip()
        if not first_line:
            return None

        request_event = json.loads(first_line)
        if request_event.get("event") != "request":
            return None

        # Extract agent ID from filename
        is_active = "_active.jsonl" in agent_file.name
        agent_id = agent_file.stem.replace("_active", "")

        agent_info = {
            "id": agent_id,
            "persona": request_event.get("persona", "default"),
            "start": request_event.get("ts", 0),
            "status": "running" if is_active else "completed",
            "prompt": request_event.get("prompt", ""),
            "facet": request_event.get("facet"),
            "failed": False,
            "runtime_seconds": None,
            "thinking_count": 0,
            "tool_count": 0,
            "cost": None,
        }

        # Count events and extract data for cost calculation in single pass
        finish_ts = None
        model = None
        usage = None
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("event")
                if event_type == "thinking":
                    agent_info["thinking_count"] += 1
                elif event_type == "tool_start":
                    agent_info["tool_count"] += 1
                elif event_type == "start":
                    model = event.get("model")
                elif event_type == "finish":
                    finish_ts = event.get("ts", 0)
                    usage = event.get("usage")
            except json.JSONDecodeError:
                continue

        # For completed agents, determine end state and calculate cost
        if not is_active:
            end_state = get_agent_end_state(agent_id)
            agent_info["failed"] = end_state in ("error", "unknown")

            # Calculate runtime from finish timestamp
            if finish_ts and agent_info["start"]:
                agent_info["runtime_seconds"] = (
                    finish_ts - agent_info["start"]
                ) / 1000.0

            # Calculate cost if we have model and usage
            if model and usage:
                try:
                    from muse.models import calc_token_cost

                    cost_data = calc_token_cost({"model": model, "usage": usage})
                    if cost_data:
                        agent_info["cost"] = cost_data["total_cost"]
                except Exception:
                    pass  # Cost calculation failed, leave as None

        return agent_info
    except (json.JSONDecodeError, IOError):
        return None


def _get_agents_for_day(day: str, facet_filter: str | None = None) -> list[dict]:
    """Get all agent runs for a specific day.

    Args:
        day: YYYYMMDD day string
        facet_filter: Optional facet to filter by (None = all facets)

    Returns:
        List of agent info dicts sorted by start time (newest first)
    """
    agents_dir = Path(state.journal_root) / "agents"
    if not agents_dir.exists():
        return []

    agents = []
    for agent_file in agents_dir.glob("*.jsonl"):
        # Skip pending files
        if "_pending.jsonl" in agent_file.name:
            continue

        # Extract agent ID and check if it's on this day
        agent_id = agent_file.stem.replace("_active", "")
        agent_day = _agent_id_to_day(agent_id)
        if agent_day != day:
            continue

        agent_info = _parse_agent_file(agent_file)
        if not agent_info:
            continue

        # Filter by facet if specified
        if facet_filter is not None and agent_info.get("facet") != facet_filter:
            continue

        agents.append(agent_info)

    # Sort by start time (newest first)
    agents.sort(key=lambda x: x["start"], reverse=True)
    return agents


def _group_agents_by_persona(
    agents: list[dict], agents_meta: dict
) -> dict[str, dict[str, Any]]:
    """Group agents by persona and add metadata.

    Returns dict mapping persona to:
        - title: Display title
        - source: "system" or "app"
        - app: App name (for app agents)
        - run_count: Total runs
        - failed_count: Failed runs
        - thinking_count: Total thinking events across all runs
        - tool_count: Total tool calls across all runs
        - total_cost: Total cost in USD across all runs
        - facets: Set of facets with runs (for color hints)
    """
    groups: dict[str, dict[str, Any]] = {}

    for agent in agents:
        persona = agent["persona"]
        if persona not in groups:
            meta = agents_meta.get(persona, {})
            groups[persona] = {
                "persona": persona,
                "title": meta.get("title", persona),
                "source": meta.get("source", "system"),
                "app": meta.get("app"),
                "run_count": 0,
                "failed_count": 0,
                "thinking_count": 0,
                "tool_count": 0,
                "total_cost": 0.0,
                "facets": set(),
            }

        groups[persona]["run_count"] += 1
        if agent.get("failed"):
            groups[persona]["failed_count"] += 1
        groups[persona]["thinking_count"] += agent.get("thinking_count", 0)
        groups[persona]["tool_count"] += agent.get("tool_count", 0)
        if agent.get("cost") is not None:
            groups[persona]["total_cost"] += agent["cost"]
        if agent.get("facet"):
            groups[persona]["facets"].add(agent["facet"])

    # Convert facet sets to lists for JSON serialization
    for group in groups.values():
        group["facets"] = list(group["facets"])

    return groups


# =============================================================================
# Page Routes
# =============================================================================


@agents_bp.route("/")
def index() -> Any:
    """Redirect to today's agent history."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:agents.agents_day", day=today))


@agents_bp.route("/<day>")
def agents_day(day: str) -> str:
    """Render agent history viewer for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    title = format_date(day)

    return render_template("app.html", title=title)


# =============================================================================
# API Routes
# =============================================================================


@agents_bp.route("/api/agents/<day>")
def api_agents_day(day: str) -> Any:
    """Get agents that ran on a specific day, grouped by persona.

    Query params:
        facet: Optional facet filter (from cookie if not specified)

    Returns:
        {
            "groups": {
                "system": [agent group objects...],
                "apps": {"app_name": [agent group objects...], ...}
            },
            "total_runs": int,
            "failed_runs": int
        }
    """
    if not re.fullmatch(DATE_RE.pattern, day):
        return jsonify({"error": "Invalid day format"}), 400

    facet_filter = _get_facet_filter()

    # Load agent metadata for titles and grouping
    agents_meta = get_agents()
    facets = get_facets()

    # Get agents for this day
    agents = _get_agents_for_day(day, facet_filter)

    # Group by persona
    persona_groups = _group_agents_by_persona(agents, agents_meta)

    # Organize into system vs app groups
    system_groups = []
    app_groups: dict[str, list] = {}

    for persona, group in persona_groups.items():
        # Add facet colors for display
        group["facet_colors"] = {}
        for facet_name in group["facets"]:
            if facet_name in facets:
                group["facet_colors"][facet_name] = facets[facet_name].get("color")

        if group["source"] == "system":
            system_groups.append(group)
        else:
            app_name = group["app"] or "unknown"
            if app_name not in app_groups:
                app_groups[app_name] = []
            app_groups[app_name].append(group)

    # Sort groups by title
    system_groups.sort(key=lambda x: x["title"].lower())
    for app_name in app_groups:
        app_groups[app_name].sort(key=lambda x: x["title"].lower())

    # Calculate totals
    total_runs = sum(g["run_count"] for g in persona_groups.values())
    failed_runs = sum(g["failed_count"] for g in persona_groups.values())

    return jsonify(
        {
            "groups": {
                "system": system_groups,
                "apps": app_groups,
            },
            "total_runs": total_runs,
            "failed_runs": failed_runs,
        }
    )


@agents_bp.route("/api/agents/<day>/<path:persona>")
def api_agent_runs(day: str, persona: str) -> Any:
    """Get runs for a specific agent on a specific day.

    Returns list of runs with full details for display.
    """
    if not re.fullmatch(DATE_RE.pattern, day):
        return jsonify({"error": "Invalid day format"}), 400

    facet_filter = _get_facet_filter()

    # Load metadata
    agents_meta = get_agents()
    facets = get_facets()

    # Get all agents for day and filter to this persona
    all_agents = _get_agents_for_day(day, facet_filter)
    runs = [a for a in all_agents if a["persona"] == persona]

    # Add facet color to each run
    for run in runs:
        run_facet = run.get("facet")
        if run_facet and run_facet in facets:
            run["facet_color"] = facets[run_facet].get("color")
            run["facet_title"] = facets[run_facet].get("title", run_facet)

    # Get agent metadata
    meta = agents_meta.get(persona, {})

    return jsonify(
        {
            "persona": persona,
            "title": meta.get("title", persona),
            "source": meta.get("source", "system"),
            "app": meta.get("app"),
            "runs": runs,
            "run_count": len(runs),
            "failed_count": sum(1 for r in runs if r.get("failed")),
        }
    )


@agents_bp.route("/api/run/<agent_id>")
def api_agent_run(agent_id: str) -> Any:
    """Return formatted markdown for a completed agent run.

    Returns:
        {
            "header": str,
            "markdown": str,
            "thinking_count": int,
            "tool_count": int,
            "cost": float | None,
            "error": str | None
        }
    """
    from think.formatters import format_file

    # Locate the agent JSONL file
    journal_path = Path(state.journal_root)
    agent_file = journal_path / "agents" / f"{agent_id}.jsonl"

    if not agent_file.exists():
        return jsonify({"error": f"Agent run {agent_id} not found"}), 404

    try:
        chunks, meta = format_file(agent_file)

        # Count events and extract cost data from the file
        thinking_count = 0
        tool_count = 0
        model = None
        usage = None
        with open(agent_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("event")
                    if event_type == "thinking":
                        thinking_count += 1
                    elif event_type == "tool_start":
                        tool_count += 1
                    elif event_type == "start":
                        model = event.get("model")
                    elif event_type == "finish":
                        usage = event.get("usage")
                except json.JSONDecodeError:
                    continue

        # Calculate cost if we have model and usage
        cost = None
        if model and usage:
            try:
                from muse.models import calc_token_cost

                cost_data = calc_token_cost({"model": model, "usage": usage})
                if cost_data:
                    cost = cost_data["total_cost"]
            except Exception:
                pass

        # Build full markdown: header + all chunks
        parts = []
        header = meta.get("header", "")
        if header:
            parts.append(header)

        for chunk in chunks:
            parts.append(chunk.get("markdown", ""))

        markdown = "\n".join(parts)

        return jsonify(
            {
                "header": header,
                "markdown": markdown,
                "thinking_count": thinking_count,
                "tool_count": tool_count,
                "cost": cost,
                "error": meta.get("error"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@agents_bp.route("/api/preview/<path:persona>")
def api_preview_prompt(persona: str) -> Any:
    """Return the complete rendered prompt for an agent.

    Returns:
        {
            "persona": str,
            "title": str,
            "full_prompt": str,
            "multi_facet": bool
        }
    """
    try:
        from think.utils import get_agent

        config = get_agent(persona)

        instruction = config.get("instruction", "")
        extra_context = config.get("extra_context", "")
        full_prompt = f"{instruction}\n\n---\n\n{extra_context}".strip()

        return jsonify(
            {
                "persona": persona,
                "title": config.get("title", persona),
                "full_prompt": full_prompt,
                "multi_facet": config.get("multi_facet", False),
            }
        )
    except FileNotFoundError:
        return jsonify({"error": f"Agent '{persona}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@agents_bp.route("/api/stats/<month>")
def api_stats(month: str) -> Any:
    """Return agent run counts per day per facet for a month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to {facet: count, ...}
        For unfaceted runs, uses "_none" as the key.
    """
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    agents_dir = Path(state.journal_root) / "agents"
    if not agents_dir.exists():
        return jsonify({})

    stats: dict[str, dict[str, int]] = {}

    for agent_file in agents_dir.glob("*.jsonl"):
        # Skip pending and active files
        if "_pending.jsonl" in agent_file.name or "_active.jsonl" in agent_file.name:
            continue

        # Extract agent ID and check if it's in this month
        agent_id = agent_file.stem
        agent_day = _agent_id_to_day(agent_id)
        if not agent_day.startswith(month):
            continue

        # Parse just enough to get facet
        try:
            with open(agent_file, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    request_event = json.loads(first_line)
                    facet = request_event.get("facet") or "_none"

                    if agent_day not in stats:
                        stats[agent_day] = {}
                    stats[agent_day][facet] = stats[agent_day].get(facet, 0) + 1
        except (json.JSONDecodeError, IOError):
            continue

    return jsonify(stats)


@agents_bp.route("/api/badge-count")
def api_badge_count() -> Any:
    """Get count of failed agent runs for today (for app icon badge)."""
    today = date.today().strftime("%Y%m%d")
    agents = _get_agents_for_day(today, facet_filter=None)
    failed_count = sum(1 for a in agents if a.get("failed"))
    return jsonify({"count": failed_count})
