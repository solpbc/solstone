# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for managing user-defined routines.

Mounted by ``think.call`` as ``sol call routines ...``.
"""

import sys
import uuid
from datetime import datetime, timezone as dt_tz
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer

from think.routines import _run_routine, cron_matches, get_config, save_config
from think.utils import get_journal

app = typer.Typer(help="Manage custom routines.")


def _resolve_id(config: dict[str, dict], prefix: str) -> str:
    matches = sorted(routine_id for routine_id in config if routine_id.startswith(prefix))
    if not matches:
        typer.echo(f"Error: routine '{prefix}' not found.", err=True)
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(f"Error: routine id '{prefix}' is ambiguous.", err=True)
        raise typer.Exit(1)
    return matches[0]


def _format_last_run(value: str | None) -> str:
    if not value:
        return "never"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _validate_timezone(name: str) -> None:
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError:
        typer.echo(f"Error: invalid timezone: {name}", err=True)
        raise typer.Exit(1)


@app.command("list")
def list_routines() -> None:
    """List all routines."""
    config = get_config()
    if not config:
        typer.echo("No routines configured.")
        return

    for routine in config.values():
        routine_id = routine.get("id", "")
        enabled_marker = "on" if routine.get("enabled") else "off"
        cadence = routine.get("cadence", "")
        last_run_display = _format_last_run(routine.get("last_run"))
        name = routine.get("name", "")
        typer.echo(
            f"{routine_id[:8]}  {enabled_marker}  {cadence:<20}  {last_run_display:<20}  {name}"
        )


@app.command()
def create(
    name: str = typer.Option(..., help="Routine name"),
    instruction: str = typer.Option(..., help="Natural-language instruction"),
    cadence: str = typer.Option(..., help="Cron expression (5-field)"),
    tz: str = typer.Option("UTC", "--timezone", help="IANA timezone"),
    facets: str = typer.Option("", help="Comma-separated facet names"),
    template: str = typer.Option("", help="Template name (stored only)"),
) -> None:
    """Create a routine."""
    try:
        cron_matches(cadence, datetime.now())
    except ValueError as exc:
        typer.echo(f"Error: invalid cadence: {exc}", err=True)
        raise typer.Exit(1)
    _validate_timezone(tz)

    routine_id = str(uuid.uuid4())
    routine = {
        "id": routine_id,
        "name": name,
        "instruction": instruction,
        "cadence": cadence,
        "timezone": tz,
        "facets": [f.strip() for f in facets.split(",") if f.strip()],
        "enabled": True,
        "created": datetime.now(dt_tz.utc).isoformat(),
        "last_run": None,
        "template": template or None,
        "notify": False,
    }

    config = get_config()
    config[routine_id] = routine
    save_config(config)
    typer.echo(f'Created routine {routine_id[:8]} "{name}"')


@app.command()
def edit(
    routine_id: str = typer.Argument(help="Routine ID (or prefix)"),
    name: str | None = typer.Option(None, help="New name"),
    instruction: str | None = typer.Option(None, help="New instruction"),
    cadence: str | None = typer.Option(None, help="New cron expression"),
    tz: str | None = typer.Option(None, "--timezone", help="New timezone"),
    enabled: bool | None = typer.Option(None, help="Enable or disable"),
    facets: str | None = typer.Option(None, help="Comma-separated facet names"),
    template: str | None = typer.Option(None, help="Template name"),
) -> None:
    """Edit a routine."""
    config = get_config()
    full_id = _resolve_id(config, routine_id)
    routine = config[full_id]

    if cadence is not None:
        try:
            cron_matches(cadence, datetime.now())
        except ValueError as exc:
            typer.echo(f"Error: invalid cadence: {exc}", err=True)
            raise typer.Exit(1)
        routine["cadence"] = cadence
    if name is not None:
        routine["name"] = name
    if instruction is not None:
        routine["instruction"] = instruction
    if tz is not None:
        _validate_timezone(tz)
        routine["timezone"] = tz
    if enabled is not None:
        routine["enabled"] = enabled
    if facets is not None:
        routine["facets"] = [f.strip() for f in facets.split(",") if f.strip()]
    if template is not None:
        routine["template"] = template or None

    config[full_id] = routine
    save_config(config)
    typer.echo(f'Updated routine {full_id[:8]} "{routine.get("name", "")}"')


@app.command()
def delete(routine_id: str = typer.Argument(help="Routine ID (or prefix)")) -> None:
    """Delete a routine."""
    config = get_config()
    full_id = _resolve_id(config, routine_id)
    routine = config.pop(full_id)
    save_config(config)
    typer.echo(f'Deleted routine {full_id[:8]} "{routine.get("name", "")}"')


@app.command()
def run(routine_id: str = typer.Argument(help="Routine ID (or prefix)")) -> None:
    """Run a routine immediately."""
    config = get_config()
    full_id = _resolve_id(config, routine_id)
    routine = config[full_id]
    typer.echo(f'Running routine "{routine.get("name", "")}"...')
    _run_routine(routine)
    typer.echo("Done.")


@app.command()
def output(routine_id: str = typer.Argument(help="Routine ID (or prefix)")) -> None:
    """Print the most recent routine output."""
    config = get_config()
    full_id = _resolve_id(config, routine_id)
    output_dir = Path(get_journal()) / "routines" / full_id
    if not output_dir.exists():
        typer.echo("No output yet.")
        return
    outputs = sorted(output_dir.glob("*.md"), reverse=True)
    if not outputs:
        typer.echo("No output yet.")
        return
    sys.stdout.write(outputs[0].read_text(encoding="utf-8"))
