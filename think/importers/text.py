# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import logging
import os

from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.importers.shared import _write_import_jsonl

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)


def _read_transcript(path: str) -> str:
    """Return transcript text from a .txt/.md/.pdf file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf required for PDF support")
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("unsupported transcript format")


def _time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS time string to seconds from midnight."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def process_transcript(
    path: str,
    day_dir: str,
    base_dt: dt.datetime,
    *,
    import_id: str,
    stream: str,
    facet: str | None = None,
    setting: str | None = None,
    audio_duration: int | None = None,
) -> list[str]:
    """Process a transcript file and write imported JSONL segments.

    Args:
        path: Path to transcript file
        day_dir: Journal day directory
        base_dt: Base datetime for the import
        import_id: Import identifier
        stream: Stream name for directory layout (day/stream/segment/)
        facet: Optional facet name
        setting: Optional setting description
        audio_duration: Optional total audio duration in seconds (for last segment)

    Returns:
        List of created file paths.
    """
    created_files = []
    text = _read_transcript(path)
    stream_dir = os.path.join(day_dir, stream)

    # Get start time from base_dt for segmentation
    start_time = base_dt.strftime("%H:%M:%S")

    # Get segments with their absolute start times
    segments = detect_transcript_segment(text, start_time)

    for idx, (start_at, seg_text) in enumerate(segments):
        # Convert segment text to structured JSON with absolute timestamps
        json_data = detect_transcript_json(seg_text, start_at)
        if not json_data:
            continue

        # Parse absolute time for segment directory name
        time_part = start_at.replace(":", "")  # "12:05:30" -> "120530"

        # Compute segment duration from absolute times
        start_seconds = _time_to_seconds(start_at)
        if idx + 1 < len(segments):
            next_start_at, _ = segments[idx + 1]
            next_seconds = _time_to_seconds(next_start_at)
            duration = next_seconds - start_seconds
        else:
            # Last segment: use remaining audio duration or default +5s
            if audio_duration:
                # audio_duration is total length, start_seconds is time-of-day
                # Need to calculate offset from recording start
                recording_start_seconds = _time_to_seconds(start_time)
                segment_offset = start_seconds - recording_start_seconds
                duration = audio_duration - segment_offset
            else:
                duration = 5

        # Negative duration indicates corrupted/invalid timestamp data
        if duration < 0:
            raise ValueError(
                f"Invalid segment duration: {duration}s for segment at {time_part}. "
                "Timestamps may be out of order or audio_duration is incorrect."
            )

        # Ensure minimum duration of 1 second
        duration = max(1, duration)

        segment_name = f"{time_part}_{duration}"
        ts_dir = os.path.join(stream_dir, segment_name)
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "imported_audio.jsonl")

        _write_import_jsonl(
            json_path,
            json_data,
            import_id=import_id,
            raw_filename=os.path.basename(path),
            facet=facet,
            setting=setting,
        )
        logger.info(f"Added transcript segment to journal: {json_path}")
        created_files.append(json_path)

    return created_files
