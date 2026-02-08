# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for todo management.

Provides human-friendly CLI access to todo operations, paralleling the
MCP tools in ``apps/todos/tools.py`` but optimized for terminal use.

Auto-discovered by ``think.call`` and mounted as ``sol call todos ...``.
"""

import typer

from apps.todos import todo

app = typer.Typer(help="Todo checklist management.")


def _print_day_facet(day: str, facet: str) -> bool:
    """Print todos for a single day+facet. Returns True if any items exist."""
    checklist = todo.TodoChecklist.load(day, facet)
    if not checklist.items:
        return False
    typer.echo(checklist.display())
    return True


@app.command("list")
def list_todos(
    day: str = typer.Argument(help="Journal day in YYYYMMDD format."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name. Omit to show all facets."
    ),
    to: str | None = typer.Option(
        None, "--to", help="End day for range query (YYYYMMDD, inclusive)."
    ),
) -> None:
    """Show the todo checklist for a day (or date range)."""
    from think.utils import get_journal

    get_journal()

    if to is not None and to < day:
        typer.echo(f"Error: --to ({to}) must not be before day ({day})", err=True)
        raise typer.Exit(1)

    # Range query
    if to is not None and to != day:
        # Use all facets for range — get_facets_with_todos only checks the start day
        from think.facets import get_facets

        facets = [facet] if facet else sorted(get_facets())

        for f in facets:
            days_with_todos = todo.get_todo_days_in_range(f, day, to)
            if not days_with_todos:
                continue
            if len(facets) > 1:
                typer.echo(f"## {f}")
            for day_str in days_with_todos:
                checklist = todo.TodoChecklist.load(day_str, f)
                if checklist.items:
                    typer.echo(f"### {day_str}")
                    typer.echo(checklist.display())
                    typer.echo()
        return

    # Single day
    facets = [facet] if facet else todo.get_facets_with_todos(day)

    if not facets:
        typer.echo(f"No todos found for {day}.")
        return

    if len(facets) == 1:
        if not _print_day_facet(day, facets[0]):
            typer.echo(f"No todos found for {day}.")
        return

    for f in facets:
        typer.echo(f"## {f}")
        _print_day_facet(day, f)
        typer.echo()
