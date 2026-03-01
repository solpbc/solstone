# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for calendar event management.

Auto-discovered by ``think.call`` and mounted as ``sol call calendar ...``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from apps.calendar import event
from think.facets import log_call_action

app = typer.Typer(help="Calendar event management.")


def _print_day_facet(day: str, facet: str) -> bool:
    """Print calendar events for a single day+facet. Returns True if any exist."""
    event_day = event.EventDay.load(day, facet)
    if not event_day.items:
        return False
    typer.echo(event_day.display())
    return True


@app.command("create")
def create_event(
    title: str = typer.Argument(help="Event title."),
    start: str = typer.Option(..., "--start", "-s", help="Start time in HH:MM format."),
    day: str | None = typer.Option(
        None,
        "--day",
        "-d",
        help="Journal day YYYYMMDD (or set SOL_DAY).",
    ),
    facet: str | None = typer.Option(
        None,
        "--facet",
        "-f",
        help="Facet name (or set SOL_FACET).",
    ),
    end: str | None = typer.Option(
        None, "--end", "-e", help="End time in HH:MM format."
    ),
    summary: str | None = typer.Option(None, "--summary", help="Event summary."),
    participants: str | None = typer.Option(
        None,
        "--participants",
        "-p",
        help="Comma-separated participant names.",
    ),
) -> None:
    """Create a new calendar event."""
    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    try:
        datetime.strptime(day, "%Y%m%d")
    except ValueError:
        typer.echo(f"Error: invalid day format '{day}'", err=True)
        raise typer.Exit(1)

    parsed_participants = None
    if participants is not None:
        parsed_participants = [p.strip() for p in participants.split(",") if p.strip()]

    try:

        def _create(day_events: event.EventDay) -> event.EventDay:
            day_events.append_event(
                title=title,
                start=start,
                end=end,
                summary=summary,
                participants=parsed_participants,
            )
            return day_events

        day_events = event.EventDay.locked_modify(day, facet, _create)
        item = day_events.items[-1]
        log_call_action(
            facet=facet,
            action="calendar_create",
            params={
                "line_number": item.index,
                "title": item.title,
                "start": item.start,
                "end": item.end,
                "summary": item.summary,
                "participants": item.participants,
            },
            day=day,
        )
        typer.echo(day_events.display())
    except event.CalendarEventEmptyTitleError:
        typer.echo("Error: event title cannot be empty", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command("list")
def list_events(
    day: str | None = typer.Argument(
        None, help="Journal day YYYYMMDD (or set SOL_DAY)."
    ),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name. Omit to show all facets."
    ),
) -> None:
    """List events for a day."""
    from think.utils import get_journal, get_sol_facet, resolve_sol_day

    journal = get_journal()
    day = resolve_sol_day(day)
    if facet is None:
        facet = get_sol_facet()

    if facet:
        if not _print_day_facet(day, facet):
            typer.echo(f"No events found for {day}.")
        return

    facets_dir = Path(journal) / "facets"
    if not facets_dir.is_dir():
        typer.echo(f"No events found for {day}.")
        return

    facets: list[str] = []
    for facet_dir in sorted(facets_dir.iterdir()):
        if not facet_dir.is_dir():
            continue
        event_path = facet_dir / "calendar" / f"{day}.jsonl"
        if event_path.is_file():
            facets.append(facet_dir.name)

    if not facets:
        typer.echo(f"No events found for {day}.")
        return

    if len(facets) == 1:
        if not _print_day_facet(day, facets[0]):
            typer.echo(f"No events found for {day}.")
        return

    for f in facets:
        typer.echo(f"## {f}")
        _print_day_facet(day, f)
        typer.echo()


@app.command("update")
def update_event(
    line_number: int = typer.Argument(help="1-based line number of the event."),
    day: str | None = typer.Option(
        None,
        "--day",
        "-d",
        help="Journal day YYYYMMDD (or set SOL_DAY).",
    ),
    facet: str | None = typer.Option(
        None,
        "--facet",
        "-f",
        help="Facet name (or set SOL_FACET).",
    ),
    title: str | None = typer.Option(None, "--title", help="New title."),
    start: str | None = typer.Option(
        None, "--start", "-s", help="New start time HH:MM."
    ),
    end: str | None = typer.Option(None, "--end", "-e", help="New end time HH:MM."),
    summary: str | None = typer.Option(None, "--summary", help="New summary."),
    participants: str | None = typer.Option(
        None,
        "--participants",
        "-p",
        help="New comma-separated participants.",
    ),
) -> None:
    """Update fields on an existing calendar event."""
    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    parsed_participants = None
    if participants is not None:
        parsed_participants = [p.strip() for p in participants.split(",") if p.strip()]

    updates = {
        "title": title,
        "start": start,
        "end": end,
        "summary": summary,
        "participants": parsed_participants if participants is not None else None,
    }

    try:

        def _update(
            day_events: event.EventDay,
        ) -> tuple[event.EventDay, event.CalendarEvent]:
            item = day_events.update_event(line_number, **updates)
            return day_events, item

        day_events, item = event.EventDay.locked_modify(day, facet, _update)
        log_call_action(
            facet=facet,
            action="calendar_update",
            params={
                "line_number": line_number,
                "title": item.title,
                "start": item.start,
                "end": item.end,
                "summary": item.summary,
                "participants": item.participants,
            },
            day=day,
        )
        typer.echo(day_events.display())
    except FileNotFoundError:
        typer.echo(f"Error: no events found for facet '{facet}' on {day}", err=True)
        raise typer.Exit(1)
    except IndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    except event.CalendarEventEmptyTitleError:
        typer.echo("Error: event title cannot be empty", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


@app.command("cancel")
def cancel_event(
    line_number: int = typer.Argument(help="1-based line number of the event."),
    day: str | None = typer.Option(
        None,
        "--day",
        "-d",
        help="Journal day YYYYMMDD (or set SOL_DAY).",
    ),
    facet: str | None = typer.Option(
        None,
        "--facet",
        "-f",
        help="Facet name (or set SOL_FACET).",
    ),
) -> None:
    """Cancel a calendar event."""
    from think.utils import get_journal, resolve_sol_day, resolve_sol_facet

    get_journal()
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)

    try:

        def _cancel(
            day_events: event.EventDay,
        ) -> tuple[event.EventDay, event.CalendarEvent]:
            item = day_events.cancel_event(line_number)
            return day_events, item

        day_events, item = event.EventDay.locked_modify(day, facet, _cancel)
        log_call_action(
            facet=facet,
            action="calendar_cancel",
            params={"line_number": line_number, "title": item.title},
            day=day,
        )
        typer.echo(day_events.display())
    except FileNotFoundError:
        typer.echo(f"Error: no events found for facet '{facet}' on {day}", err=True)
        raise typer.Exit(1)
    except IndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
