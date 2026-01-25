# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP resource handlers for transcripts."""

from datetime import datetime, timedelta

from fastmcp.resources import TextResource

from think.cluster import cluster_range
from think.mcp import mcp


def _get_transcript_resource(
    mode: str, day: str, time: str, length: str
) -> TextResource:
    """Shared handler for all transcript resource modes.

    Args:
        mode: Transcript mode - "full", "audio", "screen", or "summary"
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    try:
        # Parse the length as minutes and convert to end time
        length_minutes = int(length)

        # Validate maximum length to prevent context overload
        if length_minutes > 120:
            error_content = f"# Error\n\nRequested {length_minutes} minutes exceeds the maximum of 120 minutes per call to minimize context overload. Please request a shorter time range."
            return TextResource(
                uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
                name=f"Transcripts Error ({mode}): {day} {time} ({length}min)",
                description=f"Error: Requested length exceeds maximum",
                mime_type="text/markdown",
                text=error_content,
            )

        # Parse start time
        start_dt = datetime.strptime(f"{day}{time}", "%Y%m%d%H%M%S")
        # Calculate end time
        end_dt = start_dt + timedelta(minutes=length_minutes)
        end_time = end_dt.strftime("%H%M%S")

        # Configure cluster_range based on mode
        if mode == "full":
            markdown_content = cluster_range(
                day=day,
                start=time,
                end=end_time,
                audio=True,
                screen=True,
                insights=False,
            )
            description = f"Raw audio and screencast transcripts from {day} at {time} for {length} minutes"
        elif mode == "audio":
            markdown_content = cluster_range(
                day=day,
                start=time,
                end=end_time,
                audio=True,
                screen=False,
                insights=False,
            )
            description = (
                f"Raw audio transcripts from {day} at {time} for {length} minutes"
            )
        elif mode == "screen":
            markdown_content = cluster_range(
                day=day,
                start=time,
                end=end_time,
                audio=False,
                screen=True,
                insights=False,
            )
            description = (
                f"Raw screencast transcripts from {day} at {time} for {length} minutes"
            )
        elif mode == "summary":
            markdown_content = cluster_range(
                day=day,
                start=time,
                end=end_time,
                audio=False,
                screen=False,
                insights=True,
            )
            description = (
                f"AI-generated summaries from {day} at {time} for {length} minutes"
            )
        else:
            raise ValueError(f"Invalid transcript mode: {mode}")

        return TextResource(
            uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
            name=f"Transcripts ({mode}): {day} {time} ({length}min)",
            description=description,
            mime_type="text/markdown",
            text=markdown_content,
        )

    except Exception as e:
        error_content = f"# Error\n\nFailed to generate {mode} transcripts for {day} {time} ({length}min): {str(e)}"
        return TextResource(
            uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
            name=f"Transcripts Error ({mode}): {day} {time} ({length}min)",
            description=f"Error generating {mode} transcripts",
            mime_type="text/markdown",
            text=error_content,
        )


@mcp.resource("journal://transcripts/full/{day}/{time}/{length}")
def get_transcripts_full(day: str, time: str, length: str) -> TextResource:
    """Return formatted raw audio and screencast transcripts.

    Provides both spoken word transcripts and frame-level screen activity
    transcriptions. Use this when you need the complete raw capture data.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("full", day, time, length)


@mcp.resource("journal://transcripts/audio/{day}/{time}/{length}")
def get_transcripts_audio(day: str, time: str, length: str) -> TextResource:
    """Return formatted raw audio transcripts only.

    Provides spoken word transcripts from microphone and system audio capture.
    Use this when you only need what was said, without screen activity.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("audio", day, time, length)


@mcp.resource("journal://transcripts/screen/{day}/{time}/{length}")
def get_transcripts_screen(day: str, time: str, length: str) -> TextResource:
    """Return formatted raw screencast transcripts only.

    Provides frame-level screen activity transcriptions showing what appeared
    on screen. Use this when you only need visual activity, without audio.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("screen", day, time, length)


@mcp.resource("journal://transcripts/summary/{day}/{time}/{length}")
def get_transcripts_summary(day: str, time: str, length: str) -> TextResource:
    """Return AI-generated summaries and insights.

    Provides processed summaries of screen activity and other observations.
    Use this for high-level understanding rather than raw transcript data.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("summary", day, time, length)
