"""Transcribe audio files from observe using Gemini with VAD segmentation."""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import av
import numpy as np
import soundfile as sf
from google import genai
from silero_vad import load_silero_vad

from observe.hear import (
    SAMPLE_RATE,
    calculate_mic_overlap,
    detect_speech,
    merge_streams,
)
from think.crumbs import CrumbBuilder
from think.entities import load_entity_names
from think.models import GEMINI_FLASH
from think.utils import (
    PromptNotFoundError,
    load_prompt,
    setup_cli,
)

# Constants
MODEL = GEMINI_FLASH
MIN_SPEECH_SECONDS = 1.0

USER_PROMPT = (
    "Process the provided audio clips and output your professional accurate "
    "transcription in the specified JSON format, each clip may contain one or more speakers."
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


def transcribe_segments(
    client,
    model: str,
    prompt_text: str,
    entities_text: str,
    segments: List[Dict[str, object]],
) -> dict:
    """Send audio segments to Gemini and return the parsed JSON result."""
    from google.genai import types

    from think.models import gemini_generate

    contents = [entities_text, USER_PROMPT]
    for seg in segments:
        contents.append(
            f"This clip starts at {seg['start']} and the source is '{seg['source']}':"
        )
        contents.append(
            types.Part.from_bytes(data=seg["bytes"], mime_type="audio/flac")
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

        self.vad_model = load_silero_vad()

    def _move_to_heard(self, audio_path: Path) -> None:
        """Move processed audio_path into the day 'heard' directory."""
        heard_dir = audio_path.parent / "heard"
        try:
            heard_dir.mkdir(exist_ok=True)
            audio_path.rename(heard_dir / audio_path.name)
            logging.info("Moved %s to %s", audio_path, heard_dir)
        except Exception as exc:
            logging.error("Failed to move %s to heard: %s", audio_path, exc)

    def _process_audio(
        self, raw_path: Path, split: bool = False
    ) -> List[Dict[str, object]] | Dict[str, List[Dict[str, object]]] | None:
        """Process audio file and return segments for transcription.

        Args:
            raw_path: Path to the raw audio file
            split: If True, return dict with 'mic' and 'sys' keys for split processing

        Returns:
            If split=False: List of segments for merged processing
            If split=True: Dict with 'mic' and 'sys' keys containing their segments
            None on error
        """
        try:
            # Handle different audio formats
            if raw_path.suffix.lower() == ".m4a":
                logging.info(f"Converting m4a to numpy for processing: {raw_path}")

                # Open container and check for audio tracks
                container = av.open(str(raw_path))
                audio_streams = list(container.streams.audio)

                if len(audio_streams) == 0:
                    logging.error(f"No audio streams found in {raw_path}")
                    return None

                logging.info(
                    f"Found {len(audio_streams)} audio stream(s) in {raw_path}"
                )
                for s in audio_streams:
                    logging.info(
                        f"  Stream {s.index}: {s.codec_context.name}, "
                        f"{s.codec_context.channels} ch, {s.codec_context.sample_rate} Hz"
                    )

                # Decode function for a single stream to mono
                def decode_stream_to_mono(container, stream, target_rate=SAMPLE_RATE):
                    """Decode a stream to mono float32 at target sample rate."""
                    resampler = av.audio.resampler.AudioResampler(
                        format="flt", layout="mono", rate=target_rate
                    )
                    chunks = []
                    for frame in container.decode(stream):
                        for out_frame in resampler.resample(frame):
                            # to_ndarray() returns shape (channels, samples)
                            arr = out_frame.to_ndarray()
                            chunks.append(arr)

                    if not chunks:
                        return np.zeros(0, dtype=np.float32)

                    # Concatenate along samples axis and flatten to 1D
                    combined = np.concatenate(chunks, axis=1)
                    return combined.flatten()  # Ensure 1D mono

                def decode_stream_stereo(container, stream, target_rate=SAMPLE_RATE):
                    """Decode a stream to stereo float32 at target sample rate."""
                    resampler = av.audio.resampler.AudioResampler(
                        format="flt", layout="stereo", rate=target_rate
                    )
                    chunks = []
                    for frame in container.decode(stream):
                        for out_frame in resampler.resample(frame):
                            # to_ndarray() returns shape (channels, samples)
                            arr = out_frame.to_ndarray()
                            chunks.append(arr)

                    if not chunks:
                        return np.zeros((2, 0), dtype=np.float32)

                    # Concatenate along samples axis: shape (2, samples)
                    combined = np.concatenate(chunks, axis=1)
                    # Transpose to (samples, 2) to match soundfile format
                    return combined.T

                # Handle based on number of tracks and channels
                if len(audio_streams) == 1:
                    stream = audio_streams[0]
                    channels = stream.codec_context.channels

                    if channels >= 2:
                        # Single track with stereo/multichannel - process like normal FLAC
                        logging.info(
                            f"Single track with {channels} channels, processing as stereo"
                        )
                        container.seek(0)
                        data = decode_stream_stereo(container, stream)
                        sr = SAMPLE_RATE
                    else:
                        # Single track, mono
                        logging.info("Single track mono, processing as mono")
                        container.seek(0)
                        data = decode_stream_to_mono(container, stream)
                        sr = SAMPLE_RATE

                elif len(audio_streams) >= 2:
                    # Multiple tracks - track 0 = system, track 1 = mic
                    logging.info(
                        "Multiple tracks detected, treating as system (track 0) "
                        "and mic (track 1)"
                    )

                    # Decode track 0 (system)
                    container.seek(0)
                    sys_data = decode_stream_to_mono(container, audio_streams[0])

                    # Decode track 1 (mic)
                    container.seek(0)
                    mic_data = decode_stream_to_mono(container, audio_streams[1])

                    # Ensure same length (pad shorter one with zeros)
                    max_len = max(len(sys_data), len(mic_data))
                    if len(sys_data) < max_len:
                        sys_data = np.pad(sys_data, (0, max_len - len(sys_data)))
                    if len(mic_data) < max_len:
                        mic_data = np.pad(mic_data, (0, max_len - len(mic_data)))

                    # Stack into stereo array: shape (samples, 2)
                    # data[:, 0] = mic, data[:, 1] = system
                    data = np.column_stack([mic_data, sys_data])
                    sr = SAMPLE_RATE

                container.close()
            else:
                # Direct read for FLAC and other formats supported by soundfile
                data, sr = sf.read(raw_path, dtype="float32")

            mic_ranges: List[tuple[float, float]] = []

            # Handle split processing for dual-channel audio
            if split and data.ndim == 2:
                logging.info(
                    f"Split mode: processing mic and system channels independently for {raw_path}"
                )
                mic_data = data[:, 0]
                sys_data = data[:, 1]

                # Run VAD on each stream separately
                mic_segments, _ = detect_speech(
                    self.vad_model, "mic", mic_data, no_stash=True
                )
                sys_segments, _ = detect_speech(
                    self.vad_model, "sys", sys_data, no_stash=True
                )

                # Extract timestamp from filename (represents START of recording window)
                time_part = raw_path.stem.split("_")[0]
                today = datetime.date.today().strftime("%Y%m%d")
                base_dt = datetime.datetime.strptime(
                    f"{today}_{time_part}", "%Y%m%d_%H%M%S"
                )

                # Convert segments to format for Gemini
                result = {}
                for source, segments_list in [
                    ("mic", mic_segments),
                    ("sys", sys_segments),
                ]:
                    processed: List[Dict[str, object]] = []
                    for seg in segments_list:
                        start_dt = base_dt + datetime.timedelta(seconds=seg["offset"])
                        start_str = start_dt.strftime("%H:%M:%S")
                        audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(
                            np.int16
                        )
                        buf = io.BytesIO()
                        sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
                        processed.append(
                            {
                                "start": start_str,
                                "source": source,
                                "bytes": buf.getvalue(),
                            }
                        )
                    result[source] = processed

                    # Output segment map in verbose mode
                    segment_map = " ".join(
                        f"{seg['start']}:{seg['source']}" for seg in processed
                    )
                    logging.info(
                        f"Processed {raw_path} ({source}): {len(processed)} segments"
                    )
                    logging.info(f"Segment map ({source}): {segment_map}")

                return result

            # Standard merged processing
            if data.ndim == 1:
                merged = data
                logging.info(
                    f"Single channel audio detected in {raw_path}, no mic data."
                )
            else:
                logging.info(
                    f"Dual channel audio detected in {raw_path}, merging channels."
                )
                mic_data = data[:, 0]
                sys_data = data[:, 1]
                merged, mic_ranges = merge_streams(sys_data, mic_data, sr)

            # VAD segmentation - process complete file (no_stash=True)
            segments, _ = detect_speech(
                self.vad_model, "mix", merged, mic_ranges, no_stash=True
            )

            # Extract timestamp from filename (represents START of recording window)
            time_part = raw_path.stem.split("_")[0]
            today = datetime.date.today().strftime("%Y%m%d")
            base_dt = datetime.datetime.strptime(
                f"{today}_{time_part}", "%Y%m%d_%H%M%S"
            )

            # Convert segments to format for Gemini
            processed: List[Dict[str, object]] = []
            for seg in segments:
                start_dt = base_dt + datetime.timedelta(seconds=seg["offset"])
                start_str = start_dt.strftime("%H:%M:%S")
                audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
                source = "mic" if seg.get("mic") else "sys"
                processed.append(
                    {
                        "start": start_str,
                        "source": source,
                        "bytes": buf.getvalue(),
                    }
                )

            # Output segment map in verbose mode
            segment_map = " ".join(
                f"{seg['start']}:{seg['source']}" for seg in processed
            )
            logging.info(f"Processed {raw_path}: {len(processed)} segments")
            logging.info(f"Segment map: {segment_map}")

            return processed
        except Exception as e:
            logging.error(f"Error processing {raw_path}: {e}", exc_info=True)
            return None

    def _get_json_path(self, audio_path: Path, stream: str | None = None) -> Path:
        """Generate the corresponding JSONL path for an audio file.

        Args:
            audio_path: Path to the audio file
            stream: Optional stream identifier ('mic' or 'sys') for split processing
        """
        # Support both .flac and .m4a extensions
        if audio_path.name.endswith("_raw.flac"):
            json_name = audio_path.name.replace("_raw.flac", "_audio.jsonl")
        elif audio_path.name.endswith("_audio.flac"):
            json_name = audio_path.name.replace("_audio.flac", "_audio.jsonl")
        elif audio_path.name.endswith("_raw.m4a"):
            json_name = audio_path.name.replace("_raw.m4a", "_audio.jsonl")
        elif audio_path.name.endswith("_audio.m4a"):
            json_name = audio_path.name.replace("_audio.m4a", "_audio.jsonl")
        else:
            json_name = audio_path.stem + "_audio.jsonl"

        # Add stream prefix if provided (e.g., "123456_audio.jsonl" -> "123456_mic_audio.jsonl")
        if stream:
            json_name = json_name.replace("_audio.jsonl", f"_{stream}_audio.jsonl")

        return audio_path.with_name(json_name)

    def _transcribe(
        self,
        raw_path: Path,
        segments: List[Dict[str, object]],
        stream: str | None = None,
    ) -> bool:
        """Transcribe segments using Gemini and save JSONL.

        Args:
            raw_path: Path to the raw audio file
            segments: List of audio segments to transcribe
            stream: Optional stream identifier ('mic' or 'sys') for split processing
        """
        json_path = self._get_json_path(raw_path, stream=stream)

        try:
            entity_names = load_entity_names(spoken=True)
            if entity_names:
                entities_str = ", ".join(entity_names)
                entities_text = f"Known entities: {entities_str}"
            else:
                entities_text = ""

            # Try transcription with validation and retry logic
            result = None
            for attempt in range(2):
                result = transcribe_segments(
                    self.client, MODEL, self.prompt_text, entities_text, segments
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
            # Path is relative to the JSONL file, pointing to the heard/ subdirectory
            metadata["raw"] = f"heard/{raw_path.name}"

            # Write JSONL format: metadata first, then transcript items
            jsonl_lines = [json.dumps(metadata)]
            jsonl_lines.extend(json.dumps(item) for item in transcript_items)
            json_path.write_text("\n".join(jsonl_lines) + "\n")
            logging.info(f"Transcribed {raw_path} -> {json_path}")

            crumb_builder = CrumbBuilder().add_file(str(self.prompt_path))
            crumb_builder = crumb_builder.add_file(raw_path).add_model(MODEL)
            crumb_path = crumb_builder.commit(str(json_path))
            logging.info(f"Crumb saved to {crumb_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
            return False

    def _handle_raw(self, raw_path: Path, split: bool = False) -> None:
        """Process a raw audio file.

        Args:
            raw_path: Path to the raw audio file
            split: If True, process mic and system channels independently
        """
        if split:
            # Split processing mode
            mic_json_path = self._get_json_path(raw_path, stream="mic")
            sys_json_path = self._get_json_path(raw_path, stream="sys")

            # Check if already processed
            if mic_json_path.exists() and sys_json_path.exists():
                logging.info(f"Already processed (split), moving to heard: {raw_path}")
                self._move_to_heard(raw_path)
                return

            # Process audio in split mode
            segments_dict = self._process_audio(raw_path, split=True)
            if segments_dict is None:
                raise SystemExit(1)

            # Process each stream
            success = True
            any_segments = False
            for stream, segments in segments_dict.items():
                if len(segments) == 0:
                    logging.info(
                        f"No speech segments detected in {stream} for {raw_path}"
                    )
                    continue

                any_segments = True
                if not self._transcribe(raw_path, segments, stream=stream):
                    success = False

            # If no streams had any speech, delete the file
            if not any_segments:
                logging.info(
                    f"No speech segments detected in any stream for {raw_path}, removing file"
                )
                raw_path.unlink()
                return

            if success:
                self._move_to_heard(raw_path)
        else:
            # Standard merged processing
            # Skip if already processed
            json_path = self._get_json_path(raw_path)
            if json_path.exists():
                logging.info(f"Already processed, moving to heard: {raw_path}")
                self._move_to_heard(raw_path)
                return

            # Process audio
            segments = self._process_audio(raw_path, split=False)
            if segments is None:
                raise SystemExit(1)

            # Skip if no speech detected
            if len(segments) == 0:
                logging.info(
                    f"No speech segments detected in {raw_path}, removing file"
                )
                raw_path.unlink()
                return

            # Transcribe
            success = self._transcribe(raw_path, segments)
            if success:
                self._move_to_heard(raw_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Gemini")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file to process (.flac or .m4a)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Process mic and system channels independently into separate JSONL files",
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
    if args.split:
        logging.info(
            "Split mode enabled: processing mic and system channels independently"
        )

    transcriber = Transcriber(journal, api_key)
    transcriber._handle_raw(audio_path, split=args.split)


if __name__ == "__main__":
    main()
