# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for owner-wide skill patterns and edit requests.

Auto-discovered by ``think.call`` and mounted as ``sol call skills ...``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import typer

from think.skills import (
    find_pattern,
    load_patterns,
    load_profile,
    locked_modify_edit_requests,
    locked_modify_patterns,
    make_request_id,
    observation_key,
    profile_path,
    rename_profile,
    touch_updated,
    utc_now_iso,
)
from think.utils import require_solstone

logger = logging.getLogger(__name__)

app = typer.Typer(help="Owner-wide skill patterns and edit requests.")


class _PatternCommandError(Exception):
    """Internal control-flow error carrying a CLI message and exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


@app.callback()
def _require_up() -> None:
    require_solstone()


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _exit_with_message(message: str, *, code: int) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def _parse_activity_ids(raw_value: str) -> list[str]:
    activity_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not activity_ids:
        typer.echo("Error: --activity-ids requires at least one id.", err=True)
        raise typer.Exit(1) from None
    return activity_ids


def _parse_status_filter(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None
    statuses = {item.strip() for item in raw_value.split(",") if item.strip()}
    return statuses or None


def _pattern_observation_key(
    pattern: dict[str, Any], observation: dict[str, Any]
) -> str:
    return observation_key(
        str(pattern.get("slug") or ""),
        str(observation.get("day") or ""),
        [str(item) for item in observation.get("activity_ids", [])],
    )


def _recompute_derived_fields(pattern: dict[str, Any]) -> None:
    observations = pattern.get("observations", [])
    facets = sorted(
        {
            str(observation.get("facet") or "")
            for observation in observations
            if observation.get("facet")
        }
    )
    days = sorted(
        str(observation.get("day") or "")
        for observation in observations
        if observation.get("day")
    )
    pattern["facets_touched"] = facets
    if days:
        pattern["first_seen"] = days[0]
        pattern["last_seen"] = days[-1]


def _emit_pattern_result(
    pattern: dict[str, Any], *, json_output: bool, text_message: str
) -> None:
    if json_output:
        _echo_json(pattern)
        return
    typer.echo(text_message)


def _locked_update_pattern(
    slug: str, mutate_fn: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    updated_pattern: dict[str, Any] | None = None

    def mutate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal updated_pattern
        pattern = find_pattern(slug, rows)
        if pattern is None:
            raise _PatternCommandError("no such skill", 1)
        mutate_fn(pattern)
        updated_pattern = pattern
        return rows

    try:
        locked_modify_patterns(mutate)
    except _PatternCommandError as exc:
        _exit_with_message(exc.message, code=exc.exit_code)

    if updated_pattern is None:  # pragma: no cover - defensive assertion
        raise RuntimeError(f"pattern mutation produced no row for slug {slug}")
    return updated_pattern


@app.command("list")
def list_skills(
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter by one status or a comma-separated list of statuses.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List owner-wide skill patterns."""
    rows = load_patterns()
    status_filter = _parse_status_filter(status)
    if status_filter is not None:
        rows = [row for row in rows if str(row.get("status") or "") in status_filter]

    if json_output:
        _echo_json(rows)
        return

    for row in rows:
        slug = str(row.get("slug") or "")[:40]
        status_value = str(row.get("status") or "")[:10]
        observations = row.get("observations", [])
        last_seen = str(row.get("last_seen") or "")
        facets = ",".join(str(item) for item in row.get("facets_touched", []))
        typer.echo(
            f"{slug:<40} {status_value:<10} "
            f"obs={len(observations):<3} last={last_seen} facets={facets}"
        )


@app.command("show")
def show_skill(
    slug: str = typer.Argument(help="Skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show one owner-wide skill pattern and its profile."""
    pattern = find_pattern(slug)
    if pattern is None:
        _exit_with_message("no such skill", code=1)

    profile = load_profile(slug)
    if json_output:
        _echo_json({"pattern": pattern, "profile": profile})
        return

    typer.echo(f"name: {pattern.get('name', '')}")
    typer.echo(f"slug: {pattern.get('slug', '')}")
    typer.echo(f"status: {pattern.get('status', '')}")
    typer.echo(f"first_seen: {pattern.get('first_seen', '')}")
    typer.echo(f"last_seen: {pattern.get('last_seen', '')}")
    typer.echo(f"obs_count: {len(pattern.get('observations', []))}")
    typer.echo(f"facets_touched: {','.join(pattern.get('facets_touched', []))}")
    observations = sorted(
        pattern.get("observations", []),
        key=lambda observation: (
            str(observation.get("day", "")),
            str(observation.get("recorded_at", "")),
        ),
    )
    for observation in observations:
        activity_ids = ",".join(
            str(item) for item in observation.get("activity_ids", [])
        )
        notes = str(observation.get("notes") or "")
        typer.echo(
            f"- {observation.get('day', '')} [{observation.get('facet', '')}] "
            f"activity_ids={activity_ids} notes={notes}"
        )
    if profile is not None:
        typer.echo("---")
        typer.echo(profile.rstrip("\n"))


@app.command("observe")
def observe_skill(
    slug: str = typer.Argument(help="Skill slug."),
    day: str = typer.Option(..., "--day", help="Observation day in YYYY-MM-DD format."),
    facet: str = typer.Option(..., "--facet", help="Facet name."),
    activity_ids: str = typer.Option(
        ...,
        "--activity-ids",
        help="Comma-separated activity ids.",
    ),
    notes: str = typer.Option("", "--notes", help="Optional observation notes."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Record one new observation for an existing skill."""
    normalized_activity_ids = _parse_activity_ids(activity_ids)
    target_key = observation_key(slug, day, normalized_activity_ids)

    def mutate(pattern: dict[str, Any]) -> None:
        existing = pattern.get("observations", [])
        if any(
            _pattern_observation_key(pattern, observation) == target_key
            for observation in existing
        ):
            raise _PatternCommandError("already recorded", 0)
        existing.append(
            {
                "day": day,
                "facet": facet,
                "activity_ids": normalized_activity_ids,
                "notes": notes,
                "recorded_at": utc_now_iso(),
            }
        )
        _recompute_derived_fields(pattern)
        if pattern.get("status") == "dormant":
            pattern["status"] = "mature"
        touch_updated(pattern)

    pattern = _locked_update_pattern(slug, mutate)
    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"recorded observation: {slug}",
    )


@app.command("seed")
def seed_skill(
    slug: str = typer.Argument(help="Skill slug."),
    name: str = typer.Option(..., "--name", help="Human-readable skill name."),
    day: str = typer.Option(..., "--day", help="Observation day in YYYY-MM-DD format."),
    facet: str = typer.Option(..., "--facet", help="Facet name."),
    activity_ids: str = typer.Option(
        ...,
        "--activity-ids",
        help="Comma-separated activity ids.",
    ),
    notes: str = typer.Option("", "--notes", help="Optional observation notes."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Seed one new emerging skill pattern."""
    normalized_activity_ids = _parse_activity_ids(activity_ids)
    created_pattern: dict[str, Any] | None = None
    created_at = utc_now_iso()

    def mutate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal created_pattern
        if find_pattern(slug, rows) is not None:
            raise _PatternCommandError("slug already exists", 1)
        created_pattern = {
            "slug": slug,
            "name": name,
            "status": "emerging",
            "observations": [
                {
                    "day": day,
                    "facet": facet,
                    "activity_ids": normalized_activity_ids,
                    "notes": notes,
                    "recorded_at": created_at,
                }
            ],
            "facets_touched": [facet],
            "first_seen": day,
            "last_seen": day,
            "needs_profile": False,
            "needs_refresh": False,
            "profile_generated_at": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        rows = list(rows)
        rows.append(created_pattern)
        return rows

    try:
        locked_modify_patterns(mutate)
    except _PatternCommandError as exc:
        _exit_with_message(exc.message, code=exc.exit_code)

    if created_pattern is None:  # pragma: no cover - defensive assertion
        raise RuntimeError(f"seed did not create pattern {slug}")
    _emit_pattern_result(
        created_pattern,
        json_output=json_output,
        text_message=f"created skill: {slug}",
    )


@app.command("promote")
def promote_skill(
    slug: str = typer.Argument(help="Skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Flag one skill for profile generation."""

    def mutate(pattern: dict[str, Any]) -> None:
        if pattern.get("status") == "mature":
            raise _PatternCommandError("already mature", 0)
        if bool(pattern.get("needs_profile")):
            raise _PatternCommandError("already flagged", 0)
        pattern["needs_profile"] = True
        touch_updated(pattern)

    pattern = _locked_update_pattern(slug, mutate)
    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"flagged for profile: {slug}",
    )


@app.command("refresh")
def refresh_skill(
    slug: str = typer.Argument(help="Skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Flag one mature skill for profile refresh."""

    def mutate(pattern: dict[str, Any]) -> None:
        if pattern.get("status") != "mature":
            raise _PatternCommandError("not mature", 1)
        if bool(pattern.get("needs_refresh")):
            raise _PatternCommandError("already flagged", 0)
        pattern["needs_refresh"] = True
        touch_updated(pattern)

    pattern = _locked_update_pattern(slug, mutate)
    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"flagged for refresh: {slug}",
    )


@app.command("mark-dormant")
def mark_dormant_skill(
    slug: str = typer.Argument(help="Skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Mark one skill dormant."""

    def mutate(pattern: dict[str, Any]) -> None:
        if pattern.get("status") == "dormant":
            raise _PatternCommandError("already flagged", 0)
        pattern["status"] = "dormant"
        touch_updated(pattern)

    pattern = _locked_update_pattern(slug, mutate)
    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"marked dormant: {slug}",
    )


@app.command("retire")
def retire_skill(
    slug: str = typer.Argument(help="Skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Mark one skill retired."""

    def mutate(pattern: dict[str, Any]) -> None:
        if pattern.get("status") == "retired":
            raise _PatternCommandError("already flagged", 0)
        pattern["status"] = "retired"
        touch_updated(pattern)

    pattern = _locked_update_pattern(slug, mutate)
    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"retired skill: {slug}",
    )


@app.command("edit-request")
def edit_request_skill(
    slug: str = typer.Argument(help="Skill slug."),
    instructions: str = typer.Option(..., "--instructions", help="Edit instructions."),
    requested_by: str = typer.Option("chat", "--requested-by", help="Request source."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Append one owner-authored edit request for a skill."""
    if find_pattern(slug) is None:
        _exit_with_message("no such skill", code=1)

    request_id = make_request_id()
    request = {
        "id": request_id,
        "slug": slug,
        "instructions": instructions,
        "requested_at": utc_now_iso(),
        "requested_by": requested_by,
        "processed_at": None,
    }

    def mutate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        next_rows = list(rows)
        next_rows.append(request)
        return next_rows

    locked_modify_edit_requests(mutate)

    if json_output:
        _echo_json({"request_id": request_id, "slug": slug})
        return
    typer.echo(f"request_id: {request_id}")


@app.command("rename")
def rename_skill(
    old_slug: str = typer.Argument(help="Existing skill slug."),
    new_slug: str = typer.Argument(help="New skill slug."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Rename one skill slug and move its profile if present."""
    patterns = load_patterns()
    if find_pattern(old_slug, patterns) is None:
        _exit_with_message("no such skill", code=1)
    if find_pattern(new_slug, patterns) is not None or profile_path(new_slug).exists():
        _exit_with_message("new slug already exists", code=1)

    rename_profile(old_slug, new_slug)

    try:
        pattern = _locked_update_pattern(
            old_slug,
            lambda row: (row.__setitem__("slug", new_slug), touch_updated(row)),
        )
    except Exception:
        logger.error(
            "skills: rename_pattern failed after profile move %s -> %s",
            old_slug,
            new_slug,
            exc_info=True,
        )
        raise

    _emit_pattern_result(
        pattern,
        json_output=json_output,
        text_message=f"renamed skill: {new_slug}",
    )
