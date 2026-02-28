# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for chat browser navigation.

Auto-discovered by ``think.call`` and mounted as ``sol call chat ...``.
"""

import typer

from think.callosum import callosum_send

app = typer.Typer(help="Chat tools.")


@app.command("navigate")
def navigate(
    path: str | None = typer.Argument(None, help="URL path to navigate to."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Facet to switch to."),
) -> None:
    """Navigate the browser to a path and/or switch facet."""
    if not path and not facet:
        typer.echo("Error: provide a path and/or --facet", err=True)
        raise typer.Exit(1)

    fields: dict = {}
    if path is not None:
        fields["path"] = path
    if facet is not None:
        fields["facet"] = facet

    callosum_send("navigate", "request", **fields)

    parts = []
    if path:
        parts.append(path)
    if facet:
        parts.append(f"[{facet}]")
    typer.echo(f"Navigate: {' '.join(parts)}")
