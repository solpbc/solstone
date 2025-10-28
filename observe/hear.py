"""Audio recording utilities for observe package."""

from __future__ import annotations

import gc
import io
import logging
import os
import signal
import threading
import time
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps

from observe.detect import input_detect

# Constants
SAMPLE_RATE = 16000
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


# Audio processing utilities


def merge_streams(
    sys_data: np.ndarray,
    mic_data: np.ndarray,
    sample_rate: int,
    window_ms: int = 50,
    threshold: float = 0.005,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mix system and microphone audio while avoiding feedback."""

    length = min(len(sys_data), len(mic_data))
    if length == 0:
        return np.array([], dtype=np.float32), []

    sys_data = sys_data[:length]
    mic_data = mic_data[:length]

    sys_rms_full = float(np.sqrt(np.mean(sys_data**2))) if len(sys_data) else 0.0
    if np.isclose(sys_rms_full, 0.0):
        mic_range = (0.0, length / sample_rate)
        return mic_data, [mic_range]
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    output = np.zeros(length, dtype=np.float32)
    mic_ranges: list[tuple[float, float]] = []
    in_range = False
    range_start = 0
    consecutive_mic_windows = 0

    for start in range(0, length, window_samples):
        end = min(length, start + window_samples)
        sys_win = sys_data[start:end]
        mic_win = mic_data[start:end]
        sys_rms = float(np.sqrt(np.mean(sys_win**2))) if len(sys_win) else 0.0
        mic_rms = float(np.sqrt(np.mean(mic_win**2))) if len(mic_win) else 0.0

        if sys_rms > threshold and mic_rms > threshold:
            output[start:end] = sys_win
            consecutive_mic_windows = 0
            if in_range:
                in_range = False
                mic_ranges.append((range_start / sample_rate, start / sample_rate))
        else:
            output[start:end] = sys_win + mic_win
            if sys_rms < threshold and mic_rms > threshold:
                consecutive_mic_windows += 1
                if consecutive_mic_windows >= 2 and not in_range:
                    in_range = True
                    range_start = start - window_samples
            else:
                consecutive_mic_windows = 0

    if in_range:
        mic_ranges.append((range_start / sample_rate, length / sample_rate))

    mic_ranges = [(s, e) for s, e in mic_ranges if e - s >= 1.0]

    return output, mic_ranges


def resample_audio(
    audio_data: np.ndarray,
    in_sr: int,
    out_sr: int,
    df_model,
    df_state,
    df_sr: int,
    denoise: bool = False,
) -> np.ndarray:
    """Resample audio and optionally denoise using DeepFilterNet."""

    import torch
    from df.enhance import enhance
    from df.io import resample as df_resample

    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

    if denoise:
        if in_sr != df_sr:
            audio_tensor = df_resample(audio_tensor, in_sr, df_sr, method="kaiser_best")
            in_sr = df_sr
        with torch.no_grad():
            audio_tensor = enhance(df_model, df_state, audio_tensor)[0]

    if in_sr != out_sr:
        audio_tensor = df_resample(
            audio_tensor.unsqueeze(0), in_sr, out_sr, method="kaiser_best"
        )

    return audio_tensor.squeeze().cpu().numpy()


def calculate_mic_overlap(
    seg_start: float, seg_end: float, mic_ranges: List[Tuple[float, float]]
) -> float:
    """Return percentage of a segment overlapping with mic ranges."""

    if not mic_ranges:
        return 0.0

    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return 0.0

    overlap_duration = 0.0
    for mic_start, mic_end in mic_ranges:
        overlap_start = max(seg_start, mic_start)
        overlap_end = min(seg_end, mic_end)
        if overlap_start < overlap_end:
            overlap_duration += overlap_end - overlap_start

    return overlap_duration / seg_duration


def detect_speech(
    vad_model,
    label: str,
    buffer_data: np.ndarray,
    mic_ranges: Optional[List[Tuple[float, float]]] = None,
    no_stash: bool = False,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Detect speech segments using silero VAD."""

    if buffer_data is None or len(buffer_data) == 0:
        logging.info(f"No audio data found in {label} buffer.")
        return [], np.array([], dtype=np.float32)
    try:
        vad_model.reset_states()
        speech_segments = get_speech_timestamps(
            buffer_data,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=70,
            min_silence_duration_ms=100,
            min_speech_duration_ms=200,
            threshold=0.25,
        )
        buffer_seconds = len(buffer_data) / SAMPLE_RATE
        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE
        unprocessed_data = np.array([], dtype=np.float32)
        for i, seg in enumerate(speech_segments):
            # Skip stash logic if no_stash is True
            if (
                not no_stash
                and i == len(speech_segments) - 1
                and total_duration - seg["end"] < 1
            ):
                start_idx = int(seg["start"] * SAMPLE_RATE)
                unprocessed_data = buffer_data[start_idx:]
                break
            start_idx = int(seg["start"] * SAMPLE_RATE)
            end_idx = int(seg["end"] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]

            mic_overlap = calculate_mic_overlap(
                seg["start"], seg["end"], mic_ranges or []
            )
            in_mic = mic_overlap > 0.8

            segments.append({"offset": seg["start"], "data": seg_data, "mic": in_mic})
        logging.info(
            "Detected %d speech segments in %s of %.1f seconds with %.1f seconds remainder",
            len(speech_segments),
            label,
            buffer_seconds,
            len(unprocessed_data) / SAMPLE_RATE,
        )
        return segments, unprocessed_data
    except Exception as e:
        logging.error(f"Error in detect_speech for {label}: {e}")
        raise


def load_transcript(
    file_path: str | os.PathLike,
) -> tuple[dict, list[dict] | None, str]:
    """Load a transcript JSONL file with metadata, entries, and formatted text.

    The JSONL format has metadata as the first line (may be empty {})
    and transcript entries as subsequent lines. Handles both native
    transcripts (*_audio.jsonl) and imported transcripts (*_imported_audio.jsonl).

    Args:
        file_path: Path to the JSONL transcript file

    Returns:
        Tuple of (metadata, entries, formatted_text) where:
        - metadata: Dict from first line. Native transcripts may have empty {}
                   or contain "topics"/"setting". Imported transcripts contain
                   {"imported": {"id": "...", "domain": "...", ...}}.
                   On error, returns {"error": "message"}.
        - entries: List of entry dicts from subsequent lines, each with fields
                  like "start", "text", "source", etc. Returns None on error.
        - formatted_text: Human-readable formatted text with header and entries.
                         Format: "Start: 2024-06-15 10:05a Setting: work\n[00:00:15] (mic) Speaker 1: Hello"

    Examples:
        # Load a native transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000_audio.jsonl")
        if entries is None:
            print(f"Error: {metadata.get('error')}")
            return
        print(formatted_text)  # Human-readable output
        for entry in entries:
            print(f"{entry['start']}: {entry['text']}")

        # Load an imported transcript
        metadata, entries, formatted_text = load_transcript("20250101/120000_imported_audio.jsonl")
        if entries is not None:
            import_id = metadata.get("imported", {}).get("id")
            domain = metadata.get("imported", {}).get("domain")
            print(f"Imported from {import_id} (domain: {domain})")

        # Check for topics/setting in native transcript
        metadata, entries, formatted_text = load_transcript(path)
        if entries is not None:
            topics = metadata.get("topics")
            setting = metadata.get("setting")
    """
    import json
    import re
    from datetime import datetime
    from pathlib import Path

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


def _format_transcript_entries(path: Path, metadata: dict, entries: list[dict]) -> str:
    """Format transcript metadata and entries as human-readable text.

    Internal helper for load_transcript().
    """
    import re
    from datetime import datetime

    # Parse day and time from filename
    # Expected format: YYYYMMDD/HHMMSS_audio.jsonl or YYYYMMDD/HHMMSS_imported_audio.jsonl
    parts = path.parts
    day_str = None
    time_str = None

    # Try to find YYYYMMDD in path
    for part in reversed(parts):
        if re.match(r"^\d{8}$", part):
            day_str = part
            break

    # Parse time from filename
    filename = path.name
    time_match = re.match(r"^(\d{6}).*_audio\.jsonl$", filename)
    if time_match:
        time_str = time_match.group(1)

    # Build header line
    header_parts = []

    # Add start time if we could parse it
    if day_str and time_str:
        try:
            dt = datetime.strptime(f"{day_str}{time_str}", "%Y%m%d%H%M%S")
            # Format as "2024-06-15 10:05a"
            time_formatted = dt.strftime("%Y-%m-%d %I:%M%p").lower()
            header_parts.append(f"Start: {time_formatted}")
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
        if "domain" in imported:
            header_parts.append(f"Domain: {imported['domain']}")
        if "id" in imported:
            header_parts.append(f"Import ID: {imported['id']}")

    # Build output
    output_lines = []
    if header_parts:
        output_lines.append(" ".join(header_parts))

    # Format entries
    for entry in entries:
        entry_parts = []

        # Timestamp
        start = entry.get("start", "")
        if start:
            entry_parts.append(f"[{start}]")

        # Source (mic/sys)
        source = entry.get("source", "")
        if source:
            entry_parts.append(f"({source})")

        # Speaker
        speaker = entry.get("speaker")
        if speaker is not None:
            entry_parts.append(f"Speaker {speaker}:")
        else:
            entry_parts.append("")

        # Text
        text = entry.get("text", "")

        # Combine and add to output
        prefix = " ".join(entry_parts).strip()
        if prefix:
            output_lines.append(f"{prefix} {text}" if text else prefix)
        elif text:
            output_lines.append(text)

    return "\n".join(output_lines)
