# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Audio recording utilities for observe package."""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
import soundfile as sf

from observe.utils import SAMPLE_RATE

# Constants
BLOCK_SIZE = 1024


class AudioRecorder:
    """Records stereo audio from microphone and system audio."""

    def __init__(self):
        # Queue now holds stereo chunks (mic=left, sys=right)
        self.audio_queue = Queue()
        self._running = True
        self.recording_thread = None

    def detect(self):
        """Detect microphone and system audio devices."""
        from observe.detect import input_detect

        mic, loopback = input_detect()
        if mic is None or loopback is None:
            logging.error(f"Detection failed: mic {mic} sys {loopback}")
            return False
        logging.info(f"Detected microphone: {mic.name}")
        logging.info(f"Detected system audio: {loopback.name}")
        self.mic_device = mic
        self.sys_device = loopback
        return True

    def record_both(self):
        """Record from both mic and system audio in a loop."""
        while self._running:
            try:
                with (
                    self.mic_device.recorder(
                        samplerate=SAMPLE_RATE, channels=[-1], blocksize=BLOCK_SIZE
                    ) as mic_rec,
                    self.sys_device.recorder(
                        samplerate=SAMPLE_RATE, channels=[-1], blocksize=BLOCK_SIZE
                    ) as sys_rec,
                ):
                    block_count = 0
                    while self._running and block_count < 1000:
                        try:
                            mic_chunk = mic_rec.record(numframes=BLOCK_SIZE)
                            sys_chunk = sys_rec.record(numframes=BLOCK_SIZE)

                            # Basic validation
                            if mic_chunk is None or mic_chunk.size == 0:
                                logging.warning("Empty microphone buffer")
                                continue
                            if sys_chunk is None or sys_chunk.size == 0:
                                logging.warning("Empty system buffer")
                                continue

                            try:
                                # Try to create stereo chunk - this is where shape errors occur
                                stereo_chunk = np.column_stack((mic_chunk, sys_chunk))
                                self.audio_queue.put(stereo_chunk)
                                block_count += 1
                            except (TypeError, ValueError, AttributeError) as e:
                                # Audio device returned unexpected format - trigger clean shutdown
                                logging.error(
                                    f"Fatal audio format error - triggering clean shutdown: {e}\n"
                                    f"  mic_chunk type={type(mic_chunk)}, "
                                    f"shape={getattr(mic_chunk, 'shape', 'N/A')}, "
                                    f"dtype={getattr(mic_chunk, 'dtype', 'N/A')}\n"
                                    f"  sys_chunk type={type(sys_chunk)}, "
                                    f"shape={getattr(sys_chunk, 'shape', 'N/A')}, "
                                    f"dtype={getattr(sys_chunk, 'dtype', 'N/A')}"
                                )
                                # Stop recording thread
                                self._running = False
                                # Send SIGTERM to trigger graceful shutdown (same as Ctrl-C)
                                os.kill(os.getpid(), signal.SIGTERM)
                                return
                        except Exception as e:
                            logging.error(f"Error recording audio: {e}")
                            if not self._running:
                                break
                            time.sleep(0.5)
                del (
                    mic_rec,
                    sys_rec,
                )  # Explicitly delete to reset system audio device resources
                gc.collect()  # Force garbage collection after deleting recorders
            except Exception as e:
                logging.error(f"Error setting up recorders: {e}")
                if self._running:
                    time.sleep(1)  # Wait before retrying

    def get_buffers(self) -> np.ndarray:
        """Return concatenated stereo audio data from the queue."""
        stereo_buffer = np.array([], dtype=np.float32).reshape(0, 2)

        while not self.audio_queue.empty():
            stereo_chunk = self.audio_queue.get()

            if stereo_chunk is None or stereo_chunk.size == 0:
                logging.warning("Queue contained empty chunk")
                continue

            # Clean the data
            stereo_chunk = np.nan_to_num(
                stereo_chunk, nan=0.0, posinf=1e10, neginf=-1e10
            )
            stereo_buffer = np.vstack((stereo_buffer, stereo_chunk))

        if stereo_buffer.size == 0:
            logging.warning("No valid audio data retrieved from queue")

        return stereo_buffer

    def create_flac_bytes(self, stereo_data: np.ndarray) -> bytes:
        """Create FLAC bytes from stereo audio data."""
        if stereo_data is None or stereo_data.size == 0:
            logging.warning("Audio data is empty. Returning empty bytes.")
            return b""

        # Convert to int16
        audio_data = (np.clip(stereo_data, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        try:
            sf.write(buf, audio_data, SAMPLE_RATE, format="FLAC")
        except Exception as e:
            logging.error(
                f"Error creating FLAC: {e}. Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}"
            )
            return b""

        return buf.getvalue()

    def create_mono_flac_bytes(self, mono_data: np.ndarray) -> bytes:
        """Create FLAC bytes from mono audio data."""
        if mono_data is None or mono_data.size == 0:
            logging.warning("Mono audio data is empty. Returning empty bytes.")
            return b""

        # Convert to int16
        audio_data = (np.clip(mono_data, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        try:
            sf.write(buf, audio_data, SAMPLE_RATE, format="FLAC")
        except Exception as e:
            logging.error(
                f"Error creating mono FLAC: {e}. Audio shape: {audio_data.shape}"
            )
            return b""

        return buf.getvalue()

    def start_recording(self):
        """Start the recording thread."""
        self._running = True
        self.recording_thread = threading.Thread(target=self.record_both, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop the recording thread."""
        self._running = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)


def load_transcript(
    file_path: str | os.PathLike,
) -> tuple[dict, list[dict] | None, str]:
    """Load a transcript JSONL file with metadata, entries, and formatted text.

    The JSONL format has metadata as the first line (may be empty {})
    and transcript entries as subsequent lines. Handles both native
    transcripts (segment/audio.jsonl) and imported transcripts (segment/imported_audio.jsonl).

    Args:
        file_path: Path to the JSONL transcript file

    Returns:
        Tuple of (metadata, entries, formatted_text) where:
        - metadata: Dict from first line. Native transcripts may have empty {}
                   or contain "topics"/"setting". Imported transcripts contain
                   {"imported": {"id": "...", "facet": "...", ...}}.
                   On error, returns {"error": "message"}.
        - entries: List of entry dicts from subsequent lines, each with fields
                  like "start", "text", "source", etc. Returns None on error.
        - formatted_text: Human-readable formatted text with header and entries.
                         Format: "Start: 2024-06-15 10:05a Setting: work\n[00:00:15] (mic) Speaker 1: Hello"

    Examples:
        # Load a native transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000/audio.jsonl")
        if entries is None:
            print(f"Error: {metadata.get('error')}")
            return
        print(formatted_text)  # Human-readable output
        for entry in entries:
            print(f"{entry['start']}: {entry['text']}")

        # Load an imported transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000/imported_audio.jsonl")
        if entries is not None:
            import_id = metadata.get("imported", {}).get("id")
            facet = metadata.get("imported", {}).get("facet")
            print(f"Imported from {import_id} (facet: {facet})")

        # Check for topics/setting in native transcript
        metadata, entries, formatted_text = load_transcript(path)
        if entries is not None:
            topics = metadata.get("topics")
            setting = metadata.get("setting")
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return (
                {"error": f"File not found: {file_path}"},
                None,
                f"Error loading transcript: File not found: {file_path}",
            )

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return (
                {"error": "File is empty"},
                None,
                "Error loading transcript: File is empty",
            )

        lines = content.split("\n")

        # Parse metadata from first line
        try:
            metadata = json.loads(lines[0])
            if not isinstance(metadata, dict):
                return (
                    {"error": "First line must be a JSON object"},
                    None,
                    "Error loading transcript: First line must be a JSON object",
                )
        except json.JSONDecodeError as e:
            return (
                {"error": f"Invalid JSON in metadata line: {e}"},
                None,
                f"Error loading transcript: Invalid JSON in metadata line: {e}",
            )

        # Parse entries from remaining lines
        entries = []
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    return (
                        {"error": f"Line {i} is not a JSON object"},
                        None,
                        f"Error loading transcript: Line {i} is not a JSON object",
                    )
                entries.append(entry)
            except json.JSONDecodeError as e:
                return (
                    {"error": f"Invalid JSON at line {i}: {e}"},
                    None,
                    f"Error loading transcript: Invalid JSON at line {i}: {e}",
                )

        # Format the transcript as human-readable text
        formatted_text = _format_transcript_entries(path, metadata, entries)

        return metadata, entries, formatted_text

    except Exception as e:
        return (
            {"error": f"Failed to load transcript: {e}"},
            None,
            f"Error loading transcript: {e}",
        )


def format_audio(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format audio transcript entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (first line is metadata, rest are transcript entries)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting base timestamp)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original transcript entry)
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")

    # Separate metadata from transcript entries
    # Only first entry can be metadata (has no "start" key)
    metadata = {}
    transcript_entries = []
    skipped_count = 0
    for i, entry in enumerate(entries):
        if i == 0 and "start" not in entry:
            metadata = entry
        elif "start" in entry:
            transcript_entries.append(entry)
        else:
            skipped_count += 1

    # Build meta dict with optional error
    meta: dict[str, Any] = {}
    if skipped_count > 0:
        logger = logging.getLogger(__name__)
        error_msg = f"Skipped {skipped_count} entries missing 'start' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logger.info(error_msg)

    chunks: list[dict[str, Any]] = []

    # Parse day and time from path structure
    # Expected format: YYYYMMDD/HHMMSS_LEN/audio.jsonl
    day_str = None
    start_time = None
    base_timestamp = 0

    if file_path:
        file_path = Path(file_path)
        parts = file_path.parts

        # Try to find YYYYMMDD and HHMMSS_LEN in path
        for i, part in enumerate(reversed(parts)):
            if re.match(r"^\d{8}$", part):
                day_str = part
                # Check if previous part (parent dir) is HHMMSS_LEN segment
                if i > 0:
                    from think.utils import segment_parse

                    prev_part = list(reversed(parts))[i - 1]
                    start_time, _ = segment_parse(prev_part)
                break

    # Build header line
    header_parts = []

    # Add start time if we could parse it
    if day_str and start_time:
        try:
            day_date = datetime.strptime(day_str, "%Y%m%d").date()
            dt = datetime.combine(day_date, start_time)
            # Format as "2024-06-15 10:05a"
            time_formatted = dt.strftime("%Y-%m-%d %I:%M%p").lower()
            header_parts.append(f"Start: {time_formatted}")
            # Calculate base timestamp for entries (milliseconds)
            base_timestamp = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Add metadata fields (excluding special fields)
    skip_fields = {"error", "raw", "imported"}

    for key, value in metadata.items():
        if key in skip_fields:
            continue

        # Format the value
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)

        if value_str:
            header_parts.append(f"{key.capitalize()}: {value_str}")

    # Handle imported metadata specially
    if "imported" in metadata and isinstance(metadata["imported"], dict):
        imported = metadata["imported"]
        if "facet" in imported:
            header_parts.append(f"Facet: {imported['facet']}")
        if "id" in imported:
            header_parts.append(f"Import ID: {imported['id']}")

    # Build header from metadata parts
    if header_parts:
        meta["header"] = " ".join(header_parts)

    # Format transcript entries
    for entry in transcript_entries:
        entry_parts = []

        # Timestamp
        start = entry.get("start", "")
        entry_timestamp = base_timestamp
        if start:
            entry_parts.append(f"[{start}]")
            # Parse timestamp for chunk ordering (HH:MM:SS format, offset in ms)
            try:
                h, m, s = map(int, start.split(":"))
                entry_timestamp = base_timestamp + (h * 3600 + m * 60 + s) * 1000
            except (ValueError, AttributeError):
                pass

        # Source (mic/sys)
        source = entry.get("source", "")
        if source:
            entry_parts.append(f"({source})")

        # Speaker - handle both int and string formats (optional, for legacy transcripts)
        speaker = entry.get("speaker")
        if speaker is not None:
            if isinstance(speaker, int):
                entry_parts.append(f"Speaker {speaker}:")
            else:
                entry_parts.append(f"{speaker}:")
        else:
            entry_parts.append("")

        # Text
        text = entry.get("text", "")

        # Audio description (tone, delivery cues)
        description = entry.get("description", "")

        # Combine into markdown
        prefix = " ".join(entry_parts).strip()
        if prefix:
            markdown = f"{prefix} {text}" if text else prefix
        elif text:
            markdown = text
        else:
            continue  # Skip empty entries

        # Append description in italics if present
        if description:
            markdown = f"{markdown} *({description})*"

        chunks.append(
            {
                "timestamp": entry_timestamp,
                "markdown": markdown,
                "source": entry,
            }
        )

    # Indexer metadata - topic is always "audio" for audio transcripts
    meta["indexer"] = {"topic": "audio"}

    return chunks, meta


def _format_transcript_entries(path: Path, metadata: dict, entries: list[dict]) -> str:
    """Format transcript metadata and entries as human-readable text.

    This is a convenience wrapper around format_audio() that returns
    a single concatenated string.
    """
    # Reconstruct full entries list with metadata as first entry
    # (format_audio expects raw JSONL entries with metadata on first line)
    full_entries = [metadata] + entries
    context = {"file_path": path}
    chunks, meta = format_audio(full_entries, context)
    parts = []
    if meta.get("header"):
        parts.append(meta["header"])
    parts.extend(chunk["markdown"] for chunk in chunks)
    return "\n".join(parts)
