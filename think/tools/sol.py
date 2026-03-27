# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for sol/ identity directory.

Provides read and write access to ``{journal}/sol/self.md``,
``{journal}/sol/agency.md``, ``{journal}/sol/pulse.md``, and
``{journal}/sol/briefing/{day}.md`` — sol's
identity and initiative files.

Mounted by ``think.call`` as ``sol call sol ...``.
"""

import sys

import typer

from think.awareness import ensure_sol_directory, update_self_md_section

app = typer.Typer(
    help="Sol identity directory — self.md, agency.md, pulse.md, and briefing."
)


def _sol_dir():
    """Return the sol/ directory path, creating it if needed."""
    return ensure_sol_directory()


@app.command("self")
def self_cmd(
    write: bool = typer.Option(
        False, "--write", "-w", help="Write self.md from stdin."
    ),
    update_section: str | None = typer.Option(
        None,
        "--update-section",
        help="Update a specific ## section of self.md from stdin (e.g. 'who I'm here for').",
    ),
) -> None:
    """Read or write sol/self.md."""
    sol_dir = _sol_dir()
    self_path = sol_dir / "self.md"

    if update_section:
        content = sys.stdin.read()
        if not content.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)
        if update_self_md_section(update_section, content.strip()):
            typer.echo(f"Updated ## {update_section} in self.md.")
        else:
            typer.echo(f"Error: section '## {update_section}' not found.", err=True)
            raise typer.Exit(1)
        return

    if write:
        content = sys.stdin.read()
        if not content.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)
        self_path.write_text(content, encoding="utf-8")
        typer.echo("self.md updated.")
        return

    # Read mode
    if not self_path.exists():
        typer.echo("self.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(self_path.read_text(encoding="utf-8"))


@app.command("agency")
def agency_cmd(
    write: bool = typer.Option(
        False, "--write", "-w", help="Write agency.md from stdin."
    ),
) -> None:
    """Read or write sol/agency.md."""
    sol_dir = _sol_dir()
    agency_path = sol_dir / "agency.md"

    if write:
        content = sys.stdin.read()
        if not content.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)
        agency_path.write_text(content, encoding="utf-8")
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
        False, "--write", "-w", help="Write pulse.md from stdin."
    ),
) -> None:
    """Read or write sol/pulse.md."""
    sol_dir = _sol_dir()
    pulse_path = sol_dir / "pulse.md"

    if write:
        content = sys.stdin.read()
        if not content.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)
        pulse_path.write_text(content, encoding="utf-8")
        typer.echo("pulse.md updated.")
        return

    # Read mode
    if not pulse_path.exists():
        typer.echo("pulse.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(pulse_path.read_text(encoding="utf-8"))


@app.command("briefing")
def briefing_cmd(
    write: bool = typer.Option(
        False, "--write", "-w", help="Write briefing from stdin."
    ),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Specific day YYYYMMDD."
    ),
) -> None:
    """Read or write sol/briefing/{day}.md."""
    import os

    sol_dir = _sol_dir()
    briefing_dir = sol_dir / "briefing"

    if write:
        target_day = day or os.environ.get("SOL_DAY")
        if not target_day:
            typer.echo("Error: --day required (or set SOL_DAY).", err=True)
            raise typer.Exit(1)
        content = sys.stdin.read()
        if not content.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)
        briefing_dir.mkdir(parents=True, exist_ok=True)
        (briefing_dir / f"{target_day}.md").write_text(content, encoding="utf-8")
        typer.echo(f"Briefing for {target_day} saved.")
        return

    # Read mode
    if day:
        path = briefing_dir / f"{day}.md"
        if not path.exists():
            typer.echo("No briefing found.", err=True)
            raise typer.Exit(1)
        typer.echo(path.read_text(encoding="utf-8"))
        return

    # No day specified — find most recent
    if not briefing_dir.exists():
        typer.echo("No briefing found.", err=True)
        raise typer.Exit(1)
    files = sorted(briefing_dir.glob("*.md"), reverse=True)
    if not files:
        typer.echo("No briefing found.", err=True)
        raise typer.Exit(1)
    typer.echo(files[0].read_text(encoding="utf-8"))
