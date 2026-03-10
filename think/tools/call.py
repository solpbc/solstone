# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for journal search and browsing.

Provides human-friendly CLI access to journal operations, paralleling the
tool functions in ``think/tools/search.py`` and ``think/tools/facets.py`` but
optimized for terminal use.

Mounted by ``think.call`` as ``sol call journal ...``.
"""

import json
import sys
from pathlib import Path

import typer

from think.facets import (
    create_facet,
    delete_facet,
    facet_summary,
    get_enabled_facets,
    get_facet_news,
    get_facets,
    log_call_action,
    rename_facet,
    set_facet_muted,
    update_facet,
)
from think.importers.utils import (
    build_import_info,
    get_import_details,
    list_import_timestamps,
)
from think.indexer.journal import get_events as get_events_impl
from think.indexer.journal import search_counts as search_counts_impl
from think.indexer.journal import search_journal as search_journal_impl
from think.utils import (
    get_journal,
    iter_segments,
    resolve_sol_day,
    resolve_sol_facet,
    resolve_sol_segment,
    truncated_echo,
)

app = typer.Typer(help="Journal search and browsing.")
facet_app = typer.Typer(help="Facet management.")
app.add_typer(facet_app, name="facet")


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
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter by agent."),
    stream: str | None = typer.Option(
        None, "--stream", help="Filter by stream (e.g. import.ics, archon)."
    ),
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
    if agent is not None:
        kwargs["agent"] = agent
    if stream is not None:
        kwargs["stream"] = stream

    total, results = search_journal_impl(query, limit, offset, **kwargs)

    # Counts summary
    counts = search_counts_impl(query, **kwargs)
    typer.echo(f"{total} results")

    facet_counts = counts.get("facets", {})
    if facet_counts:
        parts = [f"{f}:{c}" for f, c in facet_counts.most_common(10)]
        typer.echo(f"Facets: {', '.join(parts)}")

    agent_counts = counts.get("agents", {})
    if agent_counts:
        parts = [f"{a}:{c}" for a, c in agent_counts.most_common(10)]
        typer.echo(f"Agents: {', '.join(parts)}")

    day_counts = counts.get("days", {})
    if day_counts:
        top_days = sorted(day_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
        parts = [f"{d}:{c}" for d, c in top_days]
        typer.echo(f"Top days: {', '.join(parts)}")

    # Results
    for r in results:
        meta = r["metadata"]
        stream_tag = f" | {meta['stream']}" if meta.get("stream") else ""
        typer.echo(
            f"\n--- {meta['day']} | {meta['facet']} | {meta['agent']}{stream_tag} ---"
        )
        typer.echo(r["text"].strip())


@app.command()
def events(
    day: str | None = typer.Argument(
        default=None, help="Day YYYYMMDD (default: SOL_DAY env)."
    ),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
) -> None:
    """List events for a day."""
    day = resolve_sol_day(day)
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


@facet_app.command()
def show(
    name: str | None = typer.Argument(
        default=None, help="Facet name (default: SOL_FACET env)."
    ),
) -> None:
    """Show facet summary."""
    name = resolve_sol_facet(name)
    try:
        summary = facet_summary(name)
    except FileNotFoundError:
        typer.echo(f"Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(summary)


@app.command()
def facets(
    all_: bool = typer.Option(False, "--all", help="Include muted facets."),
) -> None:
    """List facets."""
    if all_:
        all_facets = get_facets()
    else:
        all_facets = get_enabled_facets()
    if not all_facets:
        typer.echo("No facets found.")
        return
    for name, info in sorted(all_facets.items()):
        title = info.get("title", name)
        emoji = info.get("emoji", "")
        desc = info.get("description", "")
        muted = info.get("muted", False)

        parts = []
        if emoji:
            parts.append(f"{emoji} {title} ({name})")
        else:
            parts.append(f"{title} ({name})")
        if desc:
            parts.append(f": {desc}")
        if muted:
            parts.append(" [muted]")
        typer.echo(f"- {''.join(parts)}")


@facet_app.command()
def create(
    title: str = typer.Argument(help="Display title for the new facet."),
    emoji: str = typer.Option("📦", "--emoji", help="Icon emoji."),
    color: str = typer.Option("#667eea", "--color", help="Hex color."),
    description: str = typer.Option("", "--description", help="Facet description."),
) -> None:
    """Create a new facet."""
    try:
        slug = create_facet(title, emoji=emoji, color=color, description=description)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Created facet '{slug}'.")


@facet_app.command()
def update(
    name: str = typer.Argument(help="Facet name to update."),
    title: str | None = typer.Option(None, "--title", help="New display title."),
    description: str | None = typer.Option(
        None, "--description", help="New description."
    ),
    emoji: str | None = typer.Option(None, "--emoji", help="New icon emoji."),
    color: str | None = typer.Option(None, "--color", help="New hex color."),
) -> None:
    """Update facet configuration."""
    kwargs = {}
    if title is not None:
        kwargs["title"] = title
    if description is not None:
        kwargs["description"] = description
    if emoji is not None:
        kwargs["emoji"] = emoji
    if color is not None:
        kwargs["color"] = color

    if not kwargs:
        typer.echo(
            "Error: No fields to update. Use --title, --description, --emoji, or --color.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        changed = update_facet(name, **kwargs)
    except FileNotFoundError:
        typer.echo(f"Error: Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if changed:
        fields = ", ".join(changed.keys())
        typer.echo(f"Updated {fields} for facet '{name}'.")
    else:
        typer.echo(f"No changes for facet '{name}'.")


@facet_app.command()
def rename(
    name: str = typer.Argument(help="Current facet name."),
    new_name: str = typer.Argument(help="New facet name."),
) -> None:
    """Rename a facet."""
    try:
        rename_facet(name, new_name)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    log_call_action(
        facet=new_name,
        action="facet_rename",
        params={"old_name": name, "new_name": new_name},
    )


@facet_app.command()
def mute(name: str = typer.Argument(help="Facet name to mute.")) -> None:
    """Mute a facet (hide from default listings)."""
    try:
        set_facet_muted(name, True)
    except FileNotFoundError:
        typer.echo(f"Error: Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Facet '{name}' muted.")


@facet_app.command()
def unmute(name: str = typer.Argument(help="Facet name to unmute.")) -> None:
    """Unmute a facet (show in default listings)."""
    try:
        set_facet_muted(name, False)
    except FileNotFoundError:
        typer.echo(f"Error: Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Facet '{name}' unmuted.")


@facet_app.command()
def delete(
    name: str = typer.Argument(help="Facet name to delete."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete a facet and all its data."""
    if not yes:
        typer.echo(
            f"This will permanently delete facet '{name}' and all its data "
            "(entities, todos, events, logs, news).\n"
            "Use --yes to confirm."
        )
        raise typer.Exit(1)

    try:
        delete_facet(name)
    except FileNotFoundError:
        typer.echo(f"Error: Facet '{name}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Deleted facet '{name}'.")


@app.command()
def news(
    name: str | None = typer.Argument(
        default=None, help="Facet name (default: SOL_FACET env)."
    ),
    day: str | None = typer.Option(None, "--day", "-d", help="Specific day YYYYMMDD."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max days to show."),
    cursor: str | None = typer.Option(None, "--cursor", help="Pagination cursor."),
    write: bool = typer.Option(False, "--write", "-w", help="Write news from stdin."),
) -> None:
    """Read or write facet news."""
    name = resolve_sol_facet(name)
    if write:
        day = resolve_sol_day(day)
    elif day is None:
        from think.utils import get_sol_day

        day = get_sol_day()
    if write:
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
def agents(
    day: str | None = typer.Argument(
        default=None, help="Day YYYYMMDD (default: SOL_DAY env)."
    ),
    segment: str | None = typer.Option(
        None,
        "--segment",
        "-s",
        help="Segment key (HHMMSS_LEN, default: SOL_SEGMENT env).",
    ),
) -> None:
    """List available agent outputs for a day."""
    day = resolve_sol_day(day)
    segment = resolve_sol_segment(segment)
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
    agent: str = typer.Argument(help="Agent name (e.g., flow, meetings, activity)."),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Day YYYYMMDD (default: SOL_DAY env)."
    ),
    segment: str | None = typer.Option(
        None,
        "--segment",
        "-s",
        help="Segment key (HHMMSS_LEN, default: SOL_SEGMENT env).",
    ),
    max_bytes: int = typer.Option(
        16384, "--max", help="Max output bytes (0 = unlimited)."
    ),
) -> None:
    """Read full content of an agent output."""
    day = resolve_sol_day(day)
    segment = resolve_sol_segment(segment)
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
        candidate = base_dir / f"{agent}{ext}"
        if candidate.is_file():
            truncated_echo(candidate.read_text(encoding="utf-8"), max_bytes)
            return

    # List what is available
    available = _get_output_names(base_dir)
    if available:
        typer.echo(
            f"Agent '{agent}' not found. Available: {', '.join(available)}", err=True
        )
    else:
        typer.echo(f"Agent '{agent}' not found and no outputs exist.", err=True)
    raise typer.Exit(1)


# ============================================================================
# Import Commands
# ============================================================================


def _derive_status(info: dict) -> str:
    """Derive import status from info dict fields."""
    if info.get("error"):
        return "failed"
    if info.get("processed"):
        return "success"
    if info.get("task_id"):
        return "running"
    return "pending"


def _get_source_type(journal_root: Path, timestamp: str, info: dict) -> str:
    """Get source type from manifest.json or infer from mime_type."""
    manifest_path = journal_root / "imports" / timestamp / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_type"):
                return manifest["source_type"]
        except Exception:
            pass

    # Infer from mime_type
    mime = info.get("mime_type", "")
    if "calendar" in mime or "ics" in mime:
        return "ics"
    if "zip" in mime:
        return "archive"
    if "audio" in mime:
        return "audio"
    if "text" in mime:
        return "text"
    return "unknown"


def _get_entry_count(journal_root: Path, timestamp: str, info: dict) -> int | None:
    """Get entry count from manifest.json or total_files_created."""
    manifest_path = journal_root / "imports" / timestamp / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("entry_count") is not None:
                return manifest["entry_count"]
        except Exception:
            pass
    count = info.get("total_files_created")
    if count is not None and count > 0:
        return count
    return None


def _match_import_id(timestamps: list[str], prefix: str) -> str | None:
    """Match a partial import ID prefix to a full timestamp."""
    matches = [t for t in timestamps if t.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Check for exact match first
        if prefix in matches:
            return prefix
        typer.echo(
            f"Ambiguous prefix '{prefix}' matches {len(matches)} imports. "
            "Be more specific.",
            err=True,
        )
        raise typer.Exit(1)
    return None


@app.command(name="imports")
def imports_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Filter by source type."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List recent imports with metadata."""
    journal_root = Path(get_journal())
    timestamps = list_import_timestamps(journal_root)

    if not timestamps:
        typer.echo("No imports found.")
        return

    # Reverse chronological
    timestamps.sort(reverse=True)

    # Build info for each import
    rows = []
    for ts in timestamps:
        info = build_import_info(journal_root, ts)
        source_type = _get_source_type(journal_root, ts, info)

        if source and source_type != source:
            continue

        status = _derive_status(info)
        filename = info.get("original_filename", "unknown")
        entry_count = _get_entry_count(journal_root, ts, info)

        rows.append(
            {
                "timestamp": ts,
                "status": status,
                "source_type": source_type,
                "filename": filename,
                "entry_count": entry_count,
                "error": info.get("error"),
            }
        )

        if len(rows) >= limit:
            break

    if not rows:
        typer.echo("No imports found.")
        return

    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return

    for row in rows:
        parts = [f"{row['timestamp']} [{row['status']}]"]
        parts.append(f"{row['source_type']:8s}")
        parts.append(row["filename"])
        if row["status"] == "failed" and row.get("error"):
            parts.append(f"— error: {row['error']}")
        elif row.get("entry_count") is not None:
            parts.append(f"({row['entry_count']} entries)")
        typer.echo(" ".join(parts))


@app.command(name="import")
def import_detail(
    id: str = typer.Argument(help="Import ID or prefix (e.g. 20260309_143000)."),
) -> None:
    """Show full metadata for a single import."""
    journal_root = Path(get_journal())
    timestamps = list_import_timestamps(journal_root)

    if not timestamps:
        typer.echo("No imports found.", err=True)
        raise typer.Exit(1)

    # Match partial prefix
    matched = _match_import_id(timestamps, id)
    if matched is None:
        typer.echo(f"Import '{id}' not found.", err=True)
        raise typer.Exit(1)

    info = build_import_info(journal_root, matched)
    info["status"] = _derive_status(info)
    info["source_type"] = _get_source_type(journal_root, matched, info)

    # Merge manifest and detail data
    try:
        details = get_import_details(journal_root, matched)
        if details.get("import_json"):
            info["import_metadata"] = details["import_json"]
        if details.get("imported_json"):
            info["imported_results"] = details["imported_json"]
        if details.get("segments_json"):
            info["segments"] = details["segments_json"]
    except FileNotFoundError:
        pass

    typer.echo(json.dumps(info, indent=2, default=str))
