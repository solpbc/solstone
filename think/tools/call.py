# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for journal search and browsing.

Provides human-friendly CLI access to journal operations, paralleling the
tool functions in ``think/tools/search.py`` and ``think/tools/facets.py`` but
optimized for terminal use.

Mounted by ``think.call`` as ``sol call journal ...``.
"""

import typer

from think.facets import facet_summary, get_facet_news
from think.indexer.journal import get_events as get_events_impl
from think.indexer.journal import search_journal as search_journal_impl

app = typer.Typer(help="Journal search and browsing.")


@app.command()
def search(
    query: str = typer.Argument("", help="Search query (FTS5 syntax)."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
    offset: int = typer.Option(0, "--offset", help="Skip N results."),
    day: str | None = typer.Option(None, "--day", "-d", help="Filter by day YYYYMMDD."),
    day_from: str | None = typer.Option(
        None, "--day-from", help="Date range start YYYYMMDD."
    ),
    day_to: str | None = typer.Option(
        None, "--day-to", help="Date range end YYYYMMDD."
    ),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Filter by topic."),
) -> None:
    """Search the journal index."""
    total, results = search_journal_impl(
        query,
        limit,
        offset,
        day=day,
        day_from=day_from,
        day_to=day_to,
        facet=facet,
        topic=topic,
    )
    typer.echo(f"{total} results")
    for r in results:
        meta = r["metadata"]
        typer.echo(f"\n--- {meta['day']} | {meta['facet']} | {meta['topic']} ---")
        typer.echo(r["text"].strip())


@app.command()
def events(
    day: str = typer.Argument(help="Day in YYYYMMDD format."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
) -> None:
    """List events for a day."""
    items = get_events_impl(day, facet)
    if not items:
        typer.echo("No events found.")
        return
    for ev in items:
        time_range = ""
        if ev.get("start"):
            time_range = ev["start"]
            if ev.get("end"):
                time_range += f"-{ev['end']}"
            time_range = f" ({time_range})"
        typer.echo(f"- {ev.get('title', 'Untitled')}{time_range}")
        if ev.get("summary"):
            typer.echo(f"  {ev['summary']}")


@app.command()
def facet(
    name: str = typer.Argument(help="Facet name."),
) -> None:
    """Show facet summary."""
    try:
        summary = facet_summary(name)
    except FileNotFoundError:
        typer.echo(f"Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(summary)


@app.command()
def news(
    name: str = typer.Argument(help="Facet name."),
    day: str | None = typer.Option(None, "--day", "-d", help="Specific day YYYYMMDD."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max days to show."),
    cursor: str | None = typer.Option(None, "--cursor", help="Pagination cursor."),
) -> None:
    """Read facet news."""
    result = get_facet_news(name, cursor=cursor, limit=limit, day=day)
    days = result.get("days", [])
    if not days:
        typer.echo("No news found.")
        return
    for entry in days:
        typer.echo(entry.get("raw_content", ""))
