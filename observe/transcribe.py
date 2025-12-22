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
from observe.hear import SAMPLE_RATE
from think.callosum import callosum_send
from think.entities import load_entity_names
from think.models import GEMINI_FLASH
from think.utils import (
    PromptNotFoundError,
    load_prompt,
    setup_cli,
)

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
) -> dict:
    """Send audio turns to Gemini and return the parsed JSON result.

    Args:
        client: Gemini client
        model: Model name
        prompt_text: System prompt
        context_text: Entity and speaker context
        turns: List of turn dicts with "start", "end", "speaker", "bytes"
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
        journal_dir: Path,
        api_key: str,
        prompt_name: str = "transcribe",
    ):
        self.journal_dir = journal_dir
        self.client = genai.Client(api_key=api_key)

        try:
            prompt_data = load_prompt(prompt_name, base_dir=Path(__file__).parent)
        except PromptNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        self.prompt_path = prompt_data.path
        self.prompt_text = prompt_data.text

    def _move_to_segment(self, audio_path: Path) -> Path:
        """Move audio file to its segment and return new path."""
        from observe.utils import extract_descriptive_suffix
        from think.utils import segment_key

        segment = segment_key(audio_path.stem)
        if segment is None:
            raise ValueError(f"Invalid audio filename: {audio_path.stem}")
        suffix = extract_descriptive_suffix(audio_path.stem)
        segment_dir = audio_path.parent / segment
        try:
            segment_dir.mkdir(exist_ok=True)
            new_path = segment_dir / f"{suffix}.flac"
            audio_path.rename(new_path)
            logging.info("Moved %s to %s", audio_path, segment_dir)
            return new_path
        except Exception as exc:
            logging.error("Failed to move %s to segment: %s", audio_path, exc)
            return audio_path

    def _prepare_audio(self, raw_path: Path) -> Path:
        """Prepare audio file for diarization, converting if needed.

        Returns path to a file suitable for diarization (mono or stereo FLAC/WAV).
        For m4a files, converts to temporary FLAC.
        """
        import av

        if raw_path.suffix.lower() != ".m4a":
            return raw_path

        logging.info(f"Converting m4a to FLAC for diarization: {raw_path}")

        container = av.open(str(raw_path))
        audio_streams = list(container.streams.audio)

        if len(audio_streams) == 0:
            container.close()
            raise ValueError(f"No audio streams found in {raw_path}")

        # Decode to mono for diarization
        stream = audio_streams[0]
        resampler = av.audio.resampler.AudioResampler(
            format="flt", layout="mono", rate=SAMPLE_RATE
        )
        chunks = []
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()
                chunks.append(arr)

        container.close()

        if not chunks:
            raise ValueError(f"No audio data decoded from {raw_path}")

        combined = np.concatenate(chunks, axis=1).flatten()

        # Write to temporary FLAC in same directory
        temp_path = raw_path.with_suffix(".tmp.flac")
        audio_int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(temp_path, audio_int16, SAMPLE_RATE, format="FLAC")

        return temp_path

    def _process_audio(
        self, raw_path: Path
    ) -> tuple[List[Dict[str, object]], np.ndarray, list[str]] | None:
        """Process audio file with diarization and return turns for transcription.

        Returns:
            Tuple of (turns, embeddings, speakers) or None on error
            - turns: List of dicts with "start" (HH:MM:SS), "speaker", "bytes"
            - embeddings: numpy array (num_turns, 256)
            - speakers: List of unique speaker labels
        """
        try:
            # Prepare audio (convert m4a if needed)
            audio_path = self._prepare_audio(raw_path)
            is_temp = audio_path != raw_path

            try:
                # Run diarization
                diarization_turns, embeddings, timings = diarize(audio_path)
            finally:
                # Clean up temp file if created
                if is_temp and audio_path.exists():
                    audio_path.unlink()

            if not diarization_turns:
                logging.info(f"No speech detected in {raw_path}")
                return [], np.array([]), []

            logging.info(
                f"Diarization complete: {len(diarization_turns)} turns, "
                f"{timings['total']:.1f}s total time"
            )

            # Load audio for extracting turn clips
            data, sr = sf.read(raw_path, dtype="float32")
            if data.ndim == 2:
                # Mix to mono for clips
                data = data.mean(axis=1)

            # Extract timestamp from filename
            time_part = raw_path.stem.split("_")[0]
            today = datetime.date.today().strftime("%Y%m%d")
            base_dt = datetime.datetime.strptime(
                f"{today}_{time_part}", "%Y%m%d_%H%M%S"
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

            return processed, embeddings, speakers

        except DiarizationError as e:
            logging.error(f"Diarization failed for {raw_path}: {e}")
            raise SystemExit(1) from e
        except Exception as e:
            logging.error(f"Error processing {raw_path}: {e}", exc_info=True)
            return None

    def _get_json_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSONL path in timestamp directory.

        For split audio files (mic_audio.flac, sys_audio.flac), generates
        corresponding JSONL names (mic_audio.jsonl, sys_audio.jsonl).
        For regular stereo files (*_audio.flac), generates audio.jsonl.
        """
        from observe.utils import extract_descriptive_suffix
        from think.utils import segment_key

        segment = segment_key(audio_path.stem)
        if segment is None:
            raise ValueError(f"Invalid audio filename: {audio_path.stem}")
        segment_dir = audio_path.parent / segment
        segment_dir.mkdir(exist_ok=True)

        # Derive JSON filename from audio filename suffix
        suffix = extract_descriptive_suffix(audio_path.stem)
        # suffix is like "audio", "mic_audio", or "sys_audio"
        return segment_dir / f"{suffix}.jsonl"

    def _get_embeddings_dir(self, audio_path: Path) -> Path:
        """Get directory for storing speaker embeddings."""
        from observe.utils import extract_descriptive_suffix
        from think.utils import segment_key

        segment = segment_key(audio_path.stem)
        if segment is None:
            raise ValueError(f"Invalid audio filename: {audio_path.stem}")
        suffix = extract_descriptive_suffix(audio_path.stem)
        segment_dir = audio_path.parent / segment

        return segment_dir / suffix

    def _transcribe(
        self,
        raw_path: Path,
        turns: List[Dict[str, object]],
        speakers: list[str],
    ) -> bool:
        """Transcribe turns using Gemini and save JSONL.

        Args:
            raw_path: Path to the raw audio file
            turns: List of audio turns to transcribe
            speakers: List of speaker labels to pass to Gemini
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
            result = None
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
            from observe.utils import extract_descriptive_suffix

            suffix = extract_descriptive_suffix(raw_path.stem)
            metadata["raw"] = f"{suffix}.flac"

            # Determine source from filename for split audio files
            # mic_audio -> "mic", sys_audio -> "sys", audio -> None
            source = None
            if suffix.startswith("mic_"):
                source = "mic"
            elif suffix.startswith("sys_"):
                source = "sys"

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

    def _handle_raw(self, raw_path: Path) -> None:
        """Process a raw audio file."""
        start_time = time.time()

        # Skip if already processed
        json_path = self._get_json_path(raw_path)
        if json_path.exists():
            logging.info(f"Already processed, moving to timestamp dir: {raw_path}")
            self._move_to_segment(raw_path)
            return

        # Process audio with diarization
        result = self._process_audio(raw_path)
        if result is None:
            raise SystemExit(1)

        turns, embeddings, speakers = result

        # Skip if no speech detected
        if len(turns) == 0:
            logging.info(f"No speech detected in {raw_path}, removing file")
            raw_path.unlink()
            return

        # Transcribe
        success = self._transcribe(raw_path, turns, speakers)
        if success:
            moved_path = self._move_to_segment(raw_path)

            # Save speaker embeddings
            if embeddings.size > 0:
                embeddings_dir = self._get_embeddings_dir(raw_path)
                # Need to reconstruct turn list with speaker info for embedding save
                turn_info = [{"speaker": t["speaker"]} for t in turns]
                save_speaker_embeddings(embeddings_dir, turn_info, embeddings)

            # Emit completion event
            journal_path = Path(os.getenv("JOURNAL_PATH", ""))
            duration_ms = int((time.time() - start_time) * 1000)

            try:
                rel_input = moved_path.relative_to(journal_path)
                rel_output = json_path.relative_to(journal_path)
            except ValueError:
                rel_input = moved_path
                rel_output = json_path

            callosum_send(
                "observe",
                "transcribed",
                input=str(rel_input),
                output=str(rel_output),
                duration_ms=duration_ms,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Gemini with speaker diarization"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file to process (.flac or .m4a)",
    )
    args = setup_cli(parser)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    faulthandler.enable()

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")

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

    logging.info(f"Processing audio: {audio_path}")

    transcriber = Transcriber(journal, api_key)
    transcriber._handle_raw(audio_path)


if __name__ == "__main__":
    main()
