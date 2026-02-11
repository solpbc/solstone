# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for journal search and browsing.

Provides human-friendly CLI access to journal operations, paralleling the
tool functions in ``think/tools/search.py`` and ``think/tools/facets.py`` but
optimized for terminal use.

Mounted by ``think.call`` as ``sol call journal ...``.
"""

import re
import sys
from pathlib import Path

import typer

from think.facets import facet_summary, get_enabled_facets, get_facet_news
from think.indexer.journal import get_events as get_events_impl
from think.indexer.journal import search_counts as search_counts_impl
from think.indexer.journal import search_journal as search_journal_impl
from think.utils import get_journal, iter_segments, truncated_echo

app = typer.Typer(help="Journal search and browsing.")

SEGMENT_RE = re.compile(r"\d{6}_\d+")


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
    kwargs = {}
    if day is not None:
        kwargs["day"] = day
    if day_from is not None:
        kwargs["day_from"] = day_from
    if day_to is not None:
        kwargs["day_to"] = day_to
    if facet is not None:
        kwargs["facet"] = facet
    if topic is not None:
        kwargs["topic"] = topic

    total, results = search_journal_impl(query, limit, offset, **kwargs)

    # Counts summary
    counts = search_counts_impl(query, **kwargs)
    typer.echo(f"{total} results")

    facet_counts = counts.get("facets", {})
    if facet_counts:
        parts = [f"{f}:{c}" for f, c in facet_counts.most_common(10)]
        typer.echo(f"Facets: {', '.join(parts)}")

    topic_counts = counts.get("topics", {})
    if topic_counts:
        parts = [f"{t}:{c}" for t, c in topic_counts.most_common(10)]
        typer.echo(f"Topics: {', '.join(parts)}")

    day_counts = counts.get("days", {})
    if day_counts:
        top_days = sorted(day_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
        parts = [f"{d}:{c}" for d, c in top_days]
        typer.echo(f"Top days: {', '.join(parts)}")

    # Results
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
        facet_tag = f" [{ev.get('facet', '')}]" if ev.get("facet") else ""
        typer.echo(f"- {ev.get('title', 'Untitled')}{time_range}{facet_tag}")
        if ev.get("summary"):
            typer.echo(f"  {ev['summary']}")
        participants = ev.get("participants", [])
        if participants:
            typer.echo(f"  Participants: {', '.join(participants)}")
        if ev.get("details"):
            typer.echo(f"  Details: {ev['details']}")


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
def facets() -> None:
    """List enabled (non-muted) facets."""
    all_facets = get_enabled_facets()
    if not all_facets:
        typer.echo("No facets found.")
        return
    for name, info in sorted(all_facets.items()):
        title = info.get("title", name)
        desc = info.get("description", "")
        line = f"- {title} ({name})"
        if desc:
            line += f": {desc}"
        typer.echo(line)


@app.command()
def news(
    name: str = typer.Argument(help="Facet name."),
    day: str | None = typer.Option(None, "--day", "-d", help="Specific day YYYYMMDD."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max days to show."),
    cursor: str | None = typer.Option(None, "--cursor", help="Pagination cursor."),
    write: bool = typer.Option(False, "--write", "-w", help="Write news from stdin."),
) -> None:
    """Read or write facet news."""
    if write:
        if not day:
            typer.echo("Error: --day is required when writing news.", err=True)
            raise typer.Exit(1)

        # Read markdown from stdin
        markdown = sys.stdin.read()
        if not markdown.strip():
            typer.echo("Error: no content provided on stdin.", err=True)
            raise typer.Exit(1)

        journal_path = Path(get_journal())
        facet_path = journal_path / "facets" / name
        if not facet_path.exists():
            typer.echo(f"Error: facet '{name}' not found.", err=True)
            raise typer.Exit(1)

        news_dir = facet_path / "news"
        news_dir.mkdir(exist_ok=True)
        news_file = news_dir / f"{day}.md"
        news_file.write_text(markdown, encoding="utf-8")
        typer.echo(f"News for {day} saved to {name}.")
        return

    result = get_facet_news(name, cursor=cursor, limit=limit, day=day)
    days = result.get("days", [])
    if not days:
        typer.echo("No news found.")
        return
    for entry in days:
        typer.echo(entry.get("raw_content", ""))


@app.command()
def topics(
    day: str = typer.Argument(help="Day in YYYYMMDD format."),
    segment: str | None = typer.Option(
        None, "--segment", "-s", help="Segment key (HHMMSS_LEN)."
    ),
) -> None:
    """List available agent outputs for a day."""
    journal = get_journal()
    day_path = Path(journal) / day

    if not day_path.is_dir():
        typer.echo(f"No data for {day}.")
        return

    if segment:
        # List outputs in a specific segment directory
        seg_path = day_path / segment / "agents"
        if not seg_path.is_dir():
            typer.echo(f"Segment {segment} not found for {day}.")
            return
        _list_outputs(seg_path, f"Segment {segment}")
        return

    # List daily agent outputs
    agents_path = day_path / "agents"
    if agents_path.is_dir():
        _list_outputs(agents_path, "Daily agents")

    # List segments and their outputs (across all streams)
    seg_list = iter_segments(day)
    if seg_list:
        typer.echo(f"\nSegments: {len(seg_list)}")
        for stream_name, seg_key, seg_path_obj in seg_list:
            agents_dir = seg_path_obj / "agents"
            outputs = _get_output_names(agents_dir)
            label = f"  {stream_name}/{seg_key}" if stream_name else f"  {seg_key}"
            if outputs:
                typer.echo(f"{label}: {', '.join(outputs)}")
            else:
                typer.echo(f"{label}: (no outputs)")


def _get_output_names(directory: Path) -> list[str]:
    """Get sorted list of output file basenames in a directory."""
    names = []
    if not directory.is_dir():
        return names

    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix in (".md", ".json", ".jsonl"):
            names.append(f.name)
        elif f.is_dir():
            for nested in sorted(f.iterdir()):
                if nested.is_file() and nested.suffix in (".md", ".json", ".jsonl"):
                    names.append(f"{f.name}/{nested.name}")
    return names


def _list_outputs(directory: Path, label: str) -> None:
    """Print output files in a directory."""
    outputs = _get_output_names(directory)
    if not outputs:
        typer.echo(f"{label}: (none)")
        return
    typer.echo(f"{label}:")
    for name in outputs:
        size = (directory / name).stat().st_size
        typer.echo(f"  {name} ({size:,} bytes)")


@app.command()
def read(
    day: str = typer.Argument(help="Day in YYYYMMDD format."),
    topic: str = typer.Argument(help="Topic name (e.g., flow, meetings, activity)."),
    segment: str | None = typer.Option(
        None, "--segment", "-s", help="Segment key (HHMMSS_LEN)."
    ),
    max_bytes: int = typer.Option(
        16384, "--max", help="Max output bytes (0 = unlimited)."
    ),
) -> None:
    """Read full content of an agent output."""
    journal = get_journal()
    day_path = Path(journal) / day

    if not day_path.is_dir():
        typer.echo(f"No data for {day}.", err=True)
        raise typer.Exit(1)

    if segment:
        base_dir = day_path / segment / "agents"
    else:
        base_dir = day_path / "agents"

    if not base_dir.is_dir():
        location = f"segment {segment}" if segment else "agents"
        typer.echo(f"No {location} directory for {day}.", err=True)
        raise typer.Exit(1)

    # Try common extensions
    for ext in (".md", ".json", ".jsonl"):
        candidate = base_dir / f"{topic}{ext}"
        if candidate.is_file():
            truncated_echo(candidate.read_text(encoding="utf-8"), max_bytes)
            return

    # List what is available
    available = _get_output_names(base_dir)
    if available:
        typer.echo(
            f"Topic '{topic}' not found. Available: {', '.join(available)}", err=True
        )
    else:
        typer.echo(f"Topic '{topic}' not found and no outputs exist.", err=True)
    raise typer.Exit(1)
