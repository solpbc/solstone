# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for managing user-defined routines.

Mounted by ``think.call`` as ``sol call routines ...``.
"""

import sys
import uuid
from datetime import datetime
from datetime import timezone as dt_tz
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import frontmatter
import typer

from think.routines import _run_routine, cron_matches, get_config, save_config
from think.utils import get_journal

app = typer.Typer(help="Manage custom routines.")


def _resolve_id(config: dict[str, dict], prefix: str) -> str:
    matches = sorted(
        routine_id for routine_id in config if routine_id.startswith(prefix)
    )
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


def _templates_dir() -> Path:
    """Resolve the routines templates directory."""
    return Path(__file__).resolve().parents[2] / "routines" / "templates"


def _load_template(name: str) -> tuple[dict, str]:
    """Load a template by name. Returns (metadata, instruction_body)."""
    path = _templates_dir() / f"{name}.md"
    if not path.is_file():
        typer.echo(f"Error: template '{name}' not found.", err=True)
        raise typer.Exit(1)
    post = frontmatter.load(path)
    return dict(post.metadata), post.content.strip()


def _format_cadence(cadence: object) -> str:
    """Format a cadence value for display."""
    if isinstance(cadence, dict):
        offset = cadence.get("offset_minutes", 0)
        return f"event:calendar:{offset}m"
    return str(cadence)


def _validate_routine_cadence(cadence: object) -> None:
    """Validate a cadence value accepted by routine config."""
    if isinstance(cadence, str):
        try:
            cron_matches(cadence, datetime.now())
        except ValueError as exc:
            typer.echo(f"Error: invalid cadence: {exc}", err=True)
            raise typer.Exit(1)
        return

    if isinstance(cadence, dict):
        required_keys = {"type", "trigger", "offset_minutes"}
        missing = required_keys - set(cadence)
        if missing:
            typer.echo(
                f"Error: invalid cadence: missing keys: {', '.join(sorted(missing))}",
                err=True,
            )
            raise typer.Exit(1)
        if cadence.get("type") != "event":
            typer.echo("Error: invalid cadence: type must be 'event'", err=True)
            raise typer.Exit(1)
        if cadence.get("trigger") != "calendar":
            typer.echo("Error: invalid cadence: trigger must be 'calendar'", err=True)
            raise typer.Exit(1)
        if not isinstance(cadence.get("offset_minutes"), int):
            typer.echo(
                "Error: invalid cadence: offset_minutes must be an integer",
                err=True,
            )
            raise typer.Exit(1)
        return

    typer.echo("Error: invalid cadence: unsupported cadence type", err=True)
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
        cadence_display = _format_cadence(routine.get("cadence", ""))
        last_run_display = _format_last_run(routine.get("last_run"))
        name = routine.get("name", "")
        typer.echo(
            f"{routine_id[:8]}  {enabled_marker}  {cadence_display:<20}  {last_run_display:<20}  {name}"
        )


@app.command()
def templates() -> None:
    """List available routine templates."""
    tpl_dir = _templates_dir()
    if not tpl_dir.is_dir():
        typer.echo("No templates directory found.")
        return
    found = False
    for path in sorted(tpl_dir.glob("*.md")):
        post = frontmatter.load(path)
        desc = post.metadata.get("description", "")
        typer.echo(f"{path.stem:<25}  {desc}")
        found = True
    if not found:
        typer.echo("No templates found.")


@app.command()
def create(
    name: str = typer.Option(None, help="Routine name"),
    instruction: str = typer.Option(None, help="Natural-language instruction"),
    cadence: str = typer.Option(None, help="Cron expression (5-field)"),
    tz: str = typer.Option("", "--timezone", help="IANA timezone"),
    facets: str = typer.Option("", help="Comma-separated facet names"),
    template: str = typer.Option("", help="Template name"),
) -> None:
    """Create a routine."""
    metadata: dict = {}
    template_body = ""
    if template:
        metadata, template_body = _load_template(template)
        name = name or metadata.get("name", template)
        instruction = instruction or template_body
        if cadence is None:
            cadence = metadata.get("default_cadence")
        if not tz:
            tz = str(metadata.get("default_timezone", "UTC"))
        if not facets:
            default_facets = metadata.get("default_facets", [])
            if isinstance(default_facets, list):
                facets = ",".join(str(facet) for facet in default_facets)

    if name is None:
        typer.echo("Error: routine name is required.", err=True)
        raise typer.Exit(1)
    if instruction is None:
        typer.echo("Error: instruction is required.", err=True)
        raise typer.Exit(1)
    if cadence is None:
        typer.echo("Error: cadence is required.", err=True)
        raise typer.Exit(1)

    _validate_routine_cadence(cadence)
    if not tz:
        tz = "UTC"
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
