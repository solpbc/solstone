# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for todo management.

Auto-discovered by ``think.call`` and mounted as ``sol call todos ...``.
"""

import typer

from apps.todos import todo
from think.facets import log_call_action

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
    day: str | None = typer.Argument(
        None, help="Journal day in YYYYMMDD format (or set SOL_DAY)."
    ),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name. Omit to show all facets."
    ),
    to: str | None = typer.Option(
        None, "--to", help="End day for range query (YYYYMMDD, inclusive)."
    ),
) -> None:
    """Show the todo checklist for a day (or date range)."""
    from think.utils import get_journal, get_sol_facet, resolve_sol_day

    get_journal()
    day = resolve_sol_day(day)
    if facet is None:
        facet = get_sol_facet()

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


@app.command("add")
def add_todo(
    text: str = typer.Argument(help="Todo item text."),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Journal day in YYYYMMDD format (or set SOL_DAY)."
    ),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """Add a new todo item."""
    from datetime import datetime

    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    try:
        datetime.strptime(day, "%Y%m%d")
    except ValueError:
        typer.echo(f"Error: invalid day format '{day}'", err=True)
        raise typer.Exit(1)

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.append_entry(text)
        log_call_action(
            facet=facet,
            action="todo_add",
            params={"line_number": item.index, "text": item.text},
            day=day,
        )
        typer.echo(checklist.display())
    except todo.TodoEmptyTextError:
        typer.echo("Error: todo text cannot be empty", err=True)
        raise typer.Exit(1)


@app.command("done")
def done_todo(
    line_number: int = typer.Argument(help="1-based line number of the todo."),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Journal day in YYYYMMDD format (or set SOL_DAY)."
    ),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """Mark a todo item as done."""
    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.mark_done(line_number)
        log_call_action(
            facet=facet,
            action="todo_done",
            params={"line_number": line_number, "text": item.text},
            day=day,
        )
        typer.echo(checklist.display())
    except FileNotFoundError:
        typer.echo(f"Error: no todos found for facet '{facet}' on {day}", err=True)
        raise typer.Exit(1)
    except IndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command("cancel")
def cancel_todo(
    line_number: int = typer.Argument(help="1-based line number of the todo."),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Journal day in YYYYMMDD format (or set SOL_DAY)."
    ),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """Cancel a todo item."""
    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.cancel_entry(line_number)
        log_call_action(
            facet=facet,
            action="todo_cancel",
            params={"line_number": line_number, "text": item.text},
            day=day,
        )
        typer.echo(checklist.display())
    except FileNotFoundError:
        typer.echo(f"Error: no todos found for facet '{facet}' on {day}", err=True)
        raise typer.Exit(1)
    except IndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command("upcoming")
def upcoming_todos(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of todos."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name. Omit to show all facets."
    ),
) -> None:
    """Show upcoming todos across future days."""
    from think.utils import get_journal, get_sol_facet

    get_journal()
    if facet is None:
        facet = get_sol_facet()

    result = todo.upcoming(limit=limit, facet=facet)
    typer.echo(result)
