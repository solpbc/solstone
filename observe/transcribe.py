# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files using Gemini with speaker diarization."""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from google import genai

from observe.diarize import DiarizationError, diarize, save_speaker_embeddings
from observe.utils import (
    SAMPLE_RATE,
    get_output_dir,
    get_segment_key,
    prepare_audio_file,
)
from think.callosum import callosum_send
from think.entities import load_entity_names
from think.models import GEMINI_FLASH
from think.utils import PromptNotFoundError, get_journal, load_prompt, setup_cli

# Constants
MODEL = GEMINI_FLASH

USER_PROMPT = (
    "Process the provided audio clips and output your professional accurate "
    "transcription in the specified JSON format."
)


def validate_transcription(result: list) -> tuple[bool, str]:
    """Validate transcription result format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(result, list):
        return False, "Result is not a list"

    if len(result) == 0:
        # Empty result is OK
        return True, ""

    # Check if last item is metadata (has topics/setting, no "start" field)
    last_item = result[-1]
    if not isinstance(last_item, dict):
        return False, "Last item is not a dictionary"

    # Metadata is identified by having "topics" or "setting" and no "start"
    has_metadata = "start" not in last_item and (
        "topics" in last_item or "setting" in last_item
    )

    # Determine which items to validate as transcript entries
    items_to_check = result[:-1] if has_metadata else result

    # It's OK to have no transcript items (only metadata or empty)
    for idx, item in enumerate(items_to_check):
        if not isinstance(item, dict):
            return False, f"Item {idx} is not a dictionary"

        if "start" not in item:
            return False, f"Item {idx} missing 'start' field"

        # Validate timestamp format (HH:MM:SS)
        start = item["start"]
        if not isinstance(start, str):
            return False, f"Item {idx} 'start' is not a string"

        try:
            parts = start.split(":")
            if len(parts) != 3:
                return False, f"Item {idx} 'start' not in HH:MM:SS format"
            hours, minutes, seconds = parts
            int(hours)
            int(minutes)
            int(seconds)
        except (ValueError, AttributeError):
            return False, f"Item {idx} 'start' has invalid timestamp: {start}"

        # Validate text field exists and is string
        if "text" not in item:
            return False, f"Item {idx} missing 'text' field"

        if not isinstance(item.get("text"), str):
            return False, f"Item {idx} 'text' is not a string"

    return True, ""


def transcribe_turns(
    client,
    model: str,
    prompt_text: str,
    context_text: str,
    turns: List[Dict[str, object]],
) -> list:
    """Send audio turns to Gemini and return the parsed JSON result.

    Args:
        client: Gemini client
        model: Model name
        prompt_text: System prompt
        context_text: Entity and speaker context
        turns: List of turn dicts with "start", "speaker", "bytes"
    """
    from google.genai import types

    from think.models import gemini_generate

    contents = [context_text, USER_PROMPT]
    for turn in turns:
        contents.append(
            f"This clip starts at {turn['start']} and is spoken by '{turn['speaker']}':"
        )
        contents.append(
            types.Part.from_bytes(data=turn["bytes"], mime_type="audio/flac")
        )

    response_text = gemini_generate(
        contents=contents,
        model=model,
        temperature=0.3,
        max_output_tokens=8192 * 3,
        thinking_budget=8192,
        system_instruction=prompt_text,
        json_output=True,
        client=client,
    )
    result = json.loads(response_text)
    logging.info("Transcription result: %s", json.dumps(result, indent=2))
    return result


class Transcriber:
    def __init__(
        self,
        api_key: str,
        prompt_name: str = "transcribe",
    ):
        self.client = genai.Client(api_key=api_key)

        try:
            prompt_data = load_prompt(prompt_name, base_dir=Path(__file__).parent)
        except PromptNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        self.prompt_text = prompt_data.text

    def _prepare_audio(self, raw_path: Path) -> Path:
        """Prepare audio file for diarization, converting if needed.

        Returns path to a file suitable for diarization (mono or stereo FLAC/WAV).
        For m4a files, converts to temporary FLAC, mixing all audio streams.
        """
        return prepare_audio_file(raw_path, SAMPLE_RATE)

    def _process_audio(self, raw_path: Path) -> dict | None:
        """Process audio file with diarization and return turns for transcription.

        Returns:
            Dict with keys or None on error:
            - turns: List of dicts with "start" (HH:MM:SS), "speaker", "bytes"
            - embeddings: numpy array (num_turns, 256)
            - speakers: List of unique speaker labels
            - diarization: Dict with raw diarization data for metadata storage
              - turns: List of dicts with "start", "end" (floats), "speaker"
              - overlaps: List of dicts with "start", "end" (floats)
              - timings: Dict with timing info
        """
        # Prepare audio (convert m4a if needed)
        audio_path = self._prepare_audio(raw_path)
        is_temp = audio_path != raw_path

        try:
            # Run diarization
            diarization_turns, speaker_embeddings, timings, overlaps = diarize(
                audio_path
            )

            if not diarization_turns:
                logging.info(f"No speech detected in {raw_path}")
                return {
                    "turns": [],
                    "speaker_embeddings": {},
                    "speakers": [],
                    "diarization": {
                        "turns": [],
                        "overlaps": overlaps,
                        "timings": timings,
                    },
                }

            logging.info(
                f"Diarization complete: {len(diarization_turns)} turns, "
                f"{timings['total']:.1f}s total time"
            )

            # Load audio for extracting turn clips
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim == 2:
                # Mix to mono for clips
                data = data.mean(axis=1)

            # Extract date and time based on path structure
            # Files are always in segment directories: YYYYMMDD/HHMMSS_LEN/audio.flac
            segment = get_segment_key(raw_path)
            time_part = segment.split("_")[0] if segment else "000000"
            day_str = raw_path.parent.parent.name

            base_dt = datetime.datetime.strptime(
                f"{day_str}_{time_part}", "%Y%m%d_%H%M%S"
            )

            # Convert diarization turns to format for transcription
            processed: List[Dict[str, object]] = []
            speakers_seen: set[str] = set()

            for turn in diarization_turns:
                start_sec = turn["start"]
                end_sec = turn["end"]
                speaker = turn["speaker"]
                speakers_seen.add(speaker)

                # Calculate timestamp
                start_dt = base_dt + datetime.timedelta(seconds=start_sec)
                start_str = start_dt.strftime("%H:%M:%S")

                # Extract audio segment
                start_idx = int(start_sec * sr)
                end_idx = int(end_sec * sr)
                segment_data = data[start_idx:end_idx]

                # Convert to FLAC bytes
                audio_int16 = (np.clip(segment_data, -1.0, 1.0) * 32767).astype(
                    np.int16
                )
                buf = io.BytesIO()
                sf.write(buf, audio_int16, sr, format="FLAC")

                processed.append(
                    {
                        "start": start_str,
                        "speaker": speaker,
                        "bytes": buf.getvalue(),
                    }
                )

            speakers = sorted(speakers_seen)
            logging.info(
                f"Processed {raw_path}: {len(processed)} turns, "
                f"speakers: {', '.join(speakers)}"
            )

            return {
                "turns": processed,
                "speaker_embeddings": speaker_embeddings,
                "speakers": speakers,
                "diarization": {
                    "turns": diarization_turns,
                    "overlaps": overlaps,
                    "timings": timings,
                },
            }

        except DiarizationError as e:
            logging.error(f"Diarization failed for {raw_path}: {e}")
            raise SystemExit(1) from e
        except Exception as e:
            logging.error(f"Error processing {raw_path}: {e}", exc_info=True)
            return None
        finally:
            # Clean up temp file if created
            if is_temp and audio_path.exists():
                audio_path.unlink()

    def _get_json_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSONL path in segment directory.

        Files are always in segment directories:
        YYYYMMDD/HHMMSS_LEN/audio.flac -> YYYYMMDD/HHMMSS_LEN/audio.jsonl
        """
        return audio_path.with_suffix(".jsonl")

    def _get_embeddings_dir(self, audio_path: Path) -> Path:
        """Get directory for storing speaker embeddings."""
        return get_output_dir(audio_path)

    def _transcribe(
        self,
        raw_path: Path,
        turns: List[Dict[str, object]],
        speakers: list[str],
        diarization_data: dict | None = None,
    ) -> bool:
        """Transcribe turns using Gemini and save JSONL.

        Args:
            raw_path: Path to the raw audio file
            turns: List of audio turns to transcribe
            speakers: List of speaker labels to pass to Gemini
            diarization_data: Optional dict with raw diarization data to include
                in metadata (turns, overlaps, timings)
        """
        json_path = self._get_json_path(raw_path)

        try:
            # Build context with entities and speaker labels
            context_parts = []

            entity_names = load_entity_names(spoken=True)
            if entity_names:
                entities_str = ", ".join(entity_names)
                context_parts.append(f"Known entities: {entities_str}")

            if speakers:
                speakers_str = ", ".join(speakers)
                context_parts.append(f"Speaker labels to use: {speakers_str}")

            context_text = "\n".join(context_parts)

            # Try transcription with validation and retry logic
            for attempt in range(2):
                result = transcribe_turns(
                    self.client, MODEL, self.prompt_text, context_text, turns
                )

                # Validate the result
                is_valid, error_msg = validate_transcription(result)
                if is_valid:
                    break

                # Log validation failure
                if attempt == 0:
                    logging.info(f"Validation failed (retrying): {error_msg}")
                else:
                    logging.info(f"Validation failed on retry: {error_msg}")
                    return False

            # Extract metadata and transcript items
            metadata = {}
            transcript_items = result
            if result and isinstance(result[-1], dict):
                last_item = result[-1]
                if "start" not in last_item and (
                    "topics" in last_item or "setting" in last_item
                ):
                    metadata = last_item
                    transcript_items = result[:-1]

            # Add audio file reference to metadata
            # Files are in segment directories, stem is the suffix (e.g., "audio")
            metadata["raw"] = f"{raw_path.stem}{raw_path.suffix}"

            # Add remote origin if set (from sense.py for remote observer uploads)
            remote = os.getenv("REMOTE_NAME")
            if remote:
                metadata["remote"] = remote

            # Add diarization data if provided
            if diarization_data:
                metadata["diarization"] = {
                    "turns": diarization_data.get("turns", []),
                    "overlaps": diarization_data.get("overlaps", []),
                    "timings": diarization_data.get("timings", {}),
                    "speakers": speakers,
                }

            # Extract source from <source>_audio pattern
            # mic_audio -> "mic", sys_audio -> "sys", phone_audio -> "phone", etc.
            source = None
            suffix = raw_path.stem
            if suffix.endswith("_audio") and suffix != "audio":
                source = suffix[:-6]  # Remove "_audio" suffix

            # Add source field to transcript items for split files
            if source:
                for item in transcript_items:
                    item["source"] = source

            # Write JSONL format: metadata first, then transcript items
            jsonl_lines = [json.dumps(metadata)]
            jsonl_lines.extend(json.dumps(item) for item in transcript_items)
            json_path.write_text("\n".join(jsonl_lines) + "\n")
            logging.info(f"Transcribed {raw_path} -> {json_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
            return False

    def _handle_raw(self, raw_path: Path, redo: bool = False) -> None:
        """Process a raw audio file.

        Files are expected to be in segment directories (YYYYMMDD/HHMMSS_LEN/).

        Args:
            raw_path: Path to audio file in segment directory
            redo: If True, skip "already processed" check
        """
        start_time = time.time()

        # Derive segment from path (parent dir is segment dir)
        segment = get_segment_key(raw_path)

        # Skip if already processed (unless redo mode)
        json_path = self._get_json_path(raw_path)
        if not redo and json_path.exists():
            logging.info(f"Already processed: {raw_path}")
            return

        # Process audio with diarization
        result = self._process_audio(raw_path)
        if result is None:
            raise SystemExit(1)

        turns = result["turns"]
        speaker_embeddings = result["speaker_embeddings"]
        speakers = result["speakers"]
        diarization_data = result["diarization"]

        # Skip if no speech detected
        if len(turns) == 0:
            logging.info(f"No speech detected in {raw_path}, removing file")
            raw_path.unlink()
            return

        # Transcribe
        success = self._transcribe(raw_path, turns, speakers, diarization_data)
        if success:
            # Save speaker embeddings
            if speaker_embeddings:
                embeddings_dir = self._get_embeddings_dir(raw_path)
                save_speaker_embeddings(embeddings_dir, speaker_embeddings)

            # Emit completion event
            journal_path = Path(get_journal())
            duration_ms = int((time.time() - start_time) * 1000)

            try:
                rel_input = raw_path.relative_to(journal_path)
                rel_output = json_path.relative_to(journal_path)
            except ValueError:
                rel_input = raw_path
                rel_output = json_path

            # Extract day from audio path (grandparent is day dir)
            day = raw_path.parent.parent.name

            event_fields = {
                "input": str(rel_input),
                "output": str(rel_output),
                "duration_ms": duration_ms,
            }
            if day:
                event_fields["day"] = day
            if segment:
                event_fields["segment"] = segment
            remote = os.getenv("REMOTE_NAME")
            if remote:
                event_fields["remote"] = remote
            callosum_send("observe", "transcribed", **event_fields)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Gemini with speaker diarization"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file in segment directory (.flac or .m4a)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Reprocess file, overwriting existing outputs",
    )
    args = setup_cli(parser)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    faulthandler.enable()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    # Validate supported formats
    supported_formats = {".flac", ".m4a"}
    if audio_path.suffix.lower() not in supported_formats:
        parser.error(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Files must be in segment directories (YYYYMMDD/HHMMSS_LEN/)
    if get_segment_key(audio_path) is None:
        parser.error(
            f"Audio file must be in a segment directory (HHMMSS_LEN/), "
            f"but parent is: {audio_path.parent.name}"
        )

    logging.info(f"Processing audio: {audio_path}")

    transcriber = Transcriber(api_key)
    transcriber._handle_raw(audio_path, redo=args.redo)


if __name__ == "__main__":
    main()
