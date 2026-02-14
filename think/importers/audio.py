# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import logging
import subprocess
from datetime import timedelta
from pathlib import Path

from observe.utils import find_available_segment

logger = logging.getLogger(__name__)


def slice_audio_segment(
    source_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> str:
    """Extract an audio segment from source file, preserving original format.

    Uses stream copy for lossless extraction when possible.

    Args:
        source_path: Path to source audio file
        output_path: Path for output segment file
        start_seconds: Start offset in seconds
        duration_seconds: Duration to extract in seconds

    Returns:
        Output path on success

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_seconds),
        "-i",
        source_path,
        "-t",
        str(duration_seconds),
        "-vn",  # No video
        "-c:a",
        "copy",  # Stream copy for lossless extraction
        "-y",  # Overwrite output
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback: re-encode if stream copy fails (some formats don't support it)
        logger.debug(f"Stream copy failed, re-encoding: {output_path}")
        cmd_reencode = [
            "ffmpeg",
            "-ss",
            str(start_seconds),
            "-i",
            source_path,
            "-t",
            str(duration_seconds),
            "-vn",
            "-y",
            output_path,
        ]
        subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)

    logger.info(f"Created audio segment: {output_path}")
    return output_path


def _get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return None


def prepare_audio_segments(
    media_path: str,
    day_dir: str,
    base_dt: dt.datetime,
    import_id: str,
    stream: str,
) -> list[tuple[str, Path, list[str]]]:
    """Slice audio into 5-minute segments for observe pipeline.

    Creates segment directories with audio slices, ready for transcription
    via observe.observing events.

    Args:
        media_path: Path to source audio file
        day_dir: Day directory path (YYYYMMDD)
        base_dt: Base datetime for timestamp calculation
        import_id: Import identifier
        stream: Stream name for directory layout (day/stream/segment/)

    Returns:
        List of (segment_key, segment_dir, files_list) tuples
        where files_list contains the audio filename(s) created
    """
    media = Path(media_path)
    source_ext = media.suffix.lower()
    stream_dir = Path(day_dir) / stream

    # Get audio duration to calculate number of segments
    duration = _get_audio_duration(media_path)
    if duration is None:
        raise RuntimeError(f"Could not determine duration of {media_path}")

    # Calculate number of 5-minute segments (ceiling division)
    segment_duration = 300  # 5 minutes
    num_segments = int((duration + segment_duration - 1) // segment_duration)
    if num_segments == 0:
        num_segments = 1  # At least one segment for very short audio

    segments: list[tuple[str, Path, list[str]]] = []

    for chunk_index in range(num_segments):
        # Calculate timestamp for this segment
        ts = base_dt + timedelta(minutes=chunk_index * 5)
        time_part = ts.strftime("%H%M%S")

        # Create segment key with 5-minute duration
        segment_key_candidate = f"{time_part}_{segment_duration}"

        # Check for collision and deconflict if needed
        available_key = find_available_segment(stream_dir, segment_key_candidate)
        if available_key is None:
            logger.warning(
                f"Could not find available segment key near {segment_key_candidate}"
            )
            continue

        if available_key != segment_key_candidate:
            logger.info(
                f"Segment collision: {segment_key_candidate} -> {available_key}"
            )

        # Create segment directory under stream
        segment_dir = stream_dir / available_key
        segment_dir.mkdir(parents=True, exist_ok=True)

        # Slice audio for this segment
        audio_filename = f"imported_audio{source_ext}"
        audio_path = segment_dir / audio_filename
        start_seconds = chunk_index * segment_duration

        # For the last segment, use remaining duration
        if chunk_index == num_segments - 1:
            chunk_duration = duration - start_seconds
        else:
            chunk_duration = segment_duration

        try:
            slice_audio_segment(
                media_path,
                str(audio_path),
                start_seconds,
                chunk_duration,
            )
            segments.append((available_key, segment_dir, [audio_filename]))
            logger.info(f"Created segment: {available_key} with {audio_filename}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to slice segment {available_key}: {e}")
            # Clean up empty directory
            if segment_dir.exists() and not any(segment_dir.iterdir()):
                segment_dir.rmdir()

    return segments
