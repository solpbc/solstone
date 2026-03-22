# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for sol/ identity directory.

Provides read and write access to ``{journal}/sol/self.md`` and
``{journal}/sol/agency.md`` — sol's identity and initiative files.

Mounted by ``think.call`` as ``sol call sol ...``.
"""

import sys

import typer

from think.awareness import ensure_sol_directory, update_self_md_section

app = typer.Typer(help="Sol identity directory — self.md and agency.md.")


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
