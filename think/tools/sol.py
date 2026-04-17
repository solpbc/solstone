# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for sol/ identity directory.

Provides read and write access to ``{journal}/sol/self.md``,
``{journal}/sol/partner.md``, ``{journal}/sol/agency.md``, and
``{journal}/sol/pulse.md``, and ``{journal}/sol/awareness.md`` — sol's
identity and initiative files. Also provides read access to the morning
briefing at
``{journal}/YYYYMMDD/agents/morning_briefing.md``.

Mounted by ``think.call`` as ``sol call identity ...``.
"""

import sys
from pathlib import Path

import typer

from think.awareness import (
    _log_identity_change,
    ensure_sol_directory,
    update_identity_section,
    update_self_md_section,
)
from think.entities.core import atomic_write
from think.utils import day_dirs, day_path, get_journal, require_solstone

app = typer.Typer(
    help="Sol identity directory — self.md, partner.md, agency.md, pulse.md, awareness.md, and morning briefing.",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _hydrate() -> str:
    """Return the combined identity hydration document."""
    sol_dir = Path(get_journal()) / "sol"
    chunks = []
    for stem in ("self", "partner", "agency", "awareness"):
        path = sol_dir / f"{stem}.md"
        content = (
            path.read_text(encoding="utf-8").strip()
            if path.exists()
            else "(not present)"
        )
        chunks.append(f"# {stem}\n\n{content}\n")
    return "\n".join(chunks)


@app.callback(invoke_without_command=True)
def _require_up(ctx: typer.Context) -> None:
    require_solstone()
    if ctx.invoked_subcommand is None:
        print(_hydrate(), end="")


def _sol_dir():
    """Return the sol/ directory path, creating it if needed."""
    return ensure_sol_directory()


def _resolve_content(value: str | None) -> str:
    """Return *value* if provided, else read stdin. Exit 1 if empty."""
    if value is not None:
        content = value
    else:
        content = sys.stdin.read()
    if not content.strip():
        typer.echo("Error: no content provided.", err=True)
        raise typer.Exit(1)
    return content


@app.command("self")
def self_cmd(
    write: bool = typer.Option(
        False, "--write", "-w", help="Overwrite self.md (content via --value or stdin)."
    ),
    update_section: str | None = typer.Option(
        None,
        "--update-section",
        help="Update a specific ## section of self.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write sol/self.md."""
    sol_dir = _sol_dir()
    self_path = sol_dir / "self.md"

    if update_section:
        content = _resolve_content(value)
        if update_self_md_section(update_section, content.strip()):
            typer.echo(f"Updated ## {update_section} in self.md.")
        else:
            typer.echo(f"Error: section '## {update_section}' not found.", err=True)
            raise typer.Exit(1)
        return

    if write:
        content = _resolve_content(value)
        old_content = (
            self_path.read_text(encoding="utf-8") if self_path.exists() else ""
        )
        atomic_write(self_path, content)
        _log_identity_change(
            "self.md", old_content, content, section=None, source="cli"
        )
        typer.echo("self.md updated.")
        return

    # Read mode
    if not self_path.exists():
        typer.echo("self.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(self_path.read_text(encoding="utf-8"))


@app.command("partner")
def partner_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite partner.md (content via --value or stdin).",
    ),
    update_section: str | None = typer.Option(
        None,
        "--update-section",
        help="Update a specific ## section of partner.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write sol/partner.md."""
    sol_dir = _sol_dir()
    partner_path = sol_dir / "partner.md"

    if update_section:
        content = _resolve_content(value)
        if update_identity_section("partner.md", update_section, content.strip()):
            typer.echo(f"Updated ## {update_section} in partner.md.")
        else:
            typer.echo(f"Error: section '## {update_section}' not found.", err=True)
            raise typer.Exit(1)
        return

    if write:
        content = _resolve_content(value)
        old_content = (
            partner_path.read_text(encoding="utf-8") if partner_path.exists() else ""
        )
        atomic_write(partner_path, content)
        _log_identity_change(
            "partner.md", old_content, content, section=None, source="cli"
        )
        typer.echo("partner.md updated.")
        return

    # Read mode
    if not partner_path.exists():
        typer.echo("partner.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(partner_path.read_text(encoding="utf-8"))


@app.command("agency")
def agency_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite agency.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write sol/agency.md."""
    sol_dir = _sol_dir()
    agency_path = sol_dir / "agency.md"

    if write:
        content = _resolve_content(value)
        old_content = (
            agency_path.read_text(encoding="utf-8") if agency_path.exists() else ""
        )
        atomic_write(agency_path, content)
        _log_identity_change(
            "agency.md",
            old_content,
            content,
            section=None,
            source="cli",
        )
        typer.echo("agency.md updated.")
        return

    # Read mode
    if not agency_path.exists():
        typer.echo("agency.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(agency_path.read_text(encoding="utf-8"))


@app.command("pulse")
def pulse_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite pulse.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write sol/pulse.md."""
    sol_dir = _sol_dir()
    pulse_path = sol_dir / "pulse.md"

    if write:
        content = _resolve_content(value)
        old_content = (
            pulse_path.read_text(encoding="utf-8") if pulse_path.exists() else ""
        )
        atomic_write(pulse_path, content)
        _log_identity_change(
            "pulse.md", old_content, content, section=None, source="cli"
        )
        typer.echo("pulse.md updated.")
        return

    # Read mode
    if not pulse_path.exists():
        typer.echo("pulse.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(pulse_path.read_text(encoding="utf-8"))


@app.command("awareness")
def awareness_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite awareness.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write sol/awareness.md."""
    sol_dir = _sol_dir()
    awareness_path = sol_dir / "awareness.md"

    if write:
        content = _resolve_content(value)
        old_content = (
            awareness_path.read_text(encoding="utf-8")
            if awareness_path.exists()
            else ""
        )
        atomic_write(awareness_path, content)
        _log_identity_change(
            "awareness.md", old_content, content, section=None, source="cli"
        )
        typer.echo("awareness.md updated.")
        return

    # Read mode
    if not awareness_path.exists():
        typer.echo("awareness.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(awareness_path.read_text(encoding="utf-8"))


@app.command("briefing")
def briefing_cmd(
    day: str | None = typer.Option(None, "--day", "-d", help="Specific day YYYYMMDD."),
) -> None:
    """Read the morning briefing from YYYYMMDD/agents/morning_briefing.md."""
    if day:
        path = day_path(day, create=False) / "agents" / "morning_briefing.md"
        if not path.exists():
            typer.echo("No briefing found.", err=True)
            raise typer.Exit(1)
        typer.echo(path.read_text(encoding="utf-8"))
        return

    # No day specified — find most recent
    for day in sorted(day_dirs().keys(), reverse=True):
        agents_dir = day_path(day, create=False) / "agents"
        briefing = agents_dir / "morning_briefing.md"
        if briefing.exists() and briefing.stat().st_size > 0:
            typer.echo(briefing.read_text(encoding="utf-8"))
            return

    typer.echo("No briefing found.", err=True)
    raise typer.Exit(1)
