"""MCP resource handlers for transcripts."""

from datetime import datetime, timedelta

from fastmcp.resources import TextResource

from muse.mcp import mcp
from think.cluster import cluster_range


def _get_transcript_resource(
    mode: str, day: str, time: str, length: str
) -> TextResource:
    """Shared handler for all transcript resource modes.

    Args:
        mode: Transcript mode - "full", "audio", or "screen"
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
                day=day, start=time, end=end_time, audio=True, screen="raw"
            )
            description = f"Full transcripts (audio + raw screen) from {day} starting at {time} for {length} minutes"
        elif mode == "audio":
            markdown_content = cluster_range(
                day=day, start=time, end=end_time, audio=True, screen=None
            )
            description = f"Audio transcripts only from {day} starting at {time} for {length} minutes"
        elif mode == "screen":
            markdown_content = cluster_range(
                day=day, start=time, end=end_time, audio=False, screen="summary"
            )
            description = f"Screen summaries only from {day} starting at {time} for {length} minutes"
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
    """Return full audio and raw screen transcripts for a specific time range.

    This resource provides both audio transcripts and raw screen diffs for a given
    time range. The data is organized into 5-minute intervals and formatted
    as markdown. Each 5 minute segment could potentially be very large if there was a lot of activity.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("full", day, time, length)


@mcp.resource("journal://transcripts/audio/{day}/{time}/{length}")
def get_transcripts_audio(day: str, time: str, length: str) -> TextResource:
    """Return audio transcripts only for a specific time range.

    This resource provides audio transcripts without screen data for a given
    time range. Useful when you only need verbal/audio content.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("audio", day, time, length)


@mcp.resource("journal://transcripts/screen/{day}/{time}/{length}")
def get_transcripts_screen(day: str, time: str, length: str) -> TextResource:
    """Return screen summaries only for a specific time range.

    This resource provides processed screen summaries without audio data for a given
    time range. Useful when you only need visual activity summaries.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("screen", day, time, length)
