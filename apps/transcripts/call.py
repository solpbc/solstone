# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for transcript browsing.

Provides human-friendly CLI access to transcript operations, paralleling the
transcript helper functions in ``think.cluster`` but optimized for terminal use.

Auto-discovered by ``think.call`` and mounted as ``sol call transcripts ...``.
"""

import typer

from think.cluster import (
    cluster,
    cluster_period,
    cluster_range,
    cluster_scan,
    cluster_segments,
)
from think.utils import day_dirs, truncated_echo

app = typer.Typer(help="Transcript browsing.")


@app.command("scan")
def scan(day: str = typer.Argument(help="Day (YYYYMMDD).")) -> None:
    """List transcript coverage ranges for a day."""
    audio_ranges, screen_ranges = cluster_scan(day)

    typer.echo("Audio:")
    if audio_ranges:
        for start, end in audio_ranges:
            typer.echo(f"  {start} - {end}")
    else:
        typer.echo("  (none)")

    typer.echo("Screen:")
    if screen_ranges:
        for start, end in screen_ranges:
            typer.echo(f"  {start} - {end}")
    else:
        typer.echo("  (none)")


@app.command("segments")
def segments(day: str = typer.Argument(help="Day (YYYYMMDD).")) -> None:
    """List recording segments for a day."""
    segment_list = cluster_segments(day)
    if not segment_list:
        typer.echo("No segments.")
        return

    for segment in segment_list:
        key = segment.get("key", "")
        start = segment.get("start", "")
        end = segment.get("end", "")
        types = ", ".join(segment.get("types", []))
        typer.echo(f"{key}  {start} - {end}  [{types}]")


@app.command("read")
def read(
    day: str = typer.Argument(help="Day (YYYYMMDD)."),
    start: str | None = typer.Option(None, "--start", help="Start time (HHMMSS)."),
    length: int | None = typer.Option(None, "--length", help="Length in minutes."),
    segment: str | None = typer.Option(
        None, "--segment", help="Segment key (HHMMSS_LEN)."
    ),
    full: bool = typer.Option(
        False, "--full", help="Include audio, screen, and agents."
    ),
    raw: bool = typer.Option(False, "--raw", help="Include audio and screen only."),
    audio: bool = typer.Option(False, "--audio", help="Include audio transcripts."),
    screen: bool = typer.Option(False, "--screen", help="Include screen transcripts."),
    agents: bool = typer.Option(False, "--agents", help="Include agent outputs."),
    max_bytes: int = typer.Option(
        16384, "--max", help="Max output bytes (0 = unlimited)."
    ),
) -> None:
    """Read transcript content for a day, segment, or time range."""
    if full and raw:
        typer.echo("Error: Cannot use --full and --raw together.", err=True)
        raise typer.Exit(1)

    if (full or raw) and (audio or screen or agents):
        typer.echo(
            "Error: Cannot mix --full/--raw with individual source flags.", err=True
        )
        raise typer.Exit(1)

    if full:
        sources: dict[str, bool] = {"audio": True, "screen": True, "agents": True}
    elif raw:
        sources = {"audio": True, "screen": True, "agents": False}
    elif audio or screen or agents:
        sources = {"audio": audio, "screen": screen, "agents": agents}
    else:
        sources = {"audio": True, "screen": False, "agents": True}

    if segment and (start or length is not None):
        typer.echo("Error: Cannot mix --segment with --start/--length.", err=True)
        raise typer.Exit(1)

    if (start is not None) != (length is not None):
        typer.echo("Error: --start and --length must be used together.", err=True)
        raise typer.Exit(1)

    if start is not None and length is not None:
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start, "%H%M%S")
        end_dt = start_dt + timedelta(minutes=length)
        markdown = cluster_range(day, start, end_dt.strftime("%H%M%S"), sources)
    elif segment is not None:
        markdown, _counts = cluster_period(day, segment, sources)
    else:
        markdown, _counts = cluster(day, sources)

    truncated_echo(markdown, max_bytes)


@app.command("stats")
def stats(month: str = typer.Argument(help="Month (YYYYMM).")) -> None:
    """Show daily transcript coverage counts for a month."""
    days = sorted(day for day in day_dirs().keys() if day.startswith(month))

    days_with_data = 0
    for day in days:
        audio_ranges, screen_ranges = cluster_scan(day)
        if audio_ranges or screen_ranges:
            days_with_data += 1
            typer.echo(f"{day}  audio:{len(audio_ranges)} screen:{len(screen_ranges)}")

    if not days_with_data:
        typer.echo(f"No data for {month}.")
        return

    typer.echo("")
    typer.echo(f"Total: {days_with_data} days with data")
