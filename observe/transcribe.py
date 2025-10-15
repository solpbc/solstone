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
from think.models import GEMINI_FLASH
from think.utils import (
    PromptNotFoundError,
    load_entity_names,
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
        temperature=0.1,
        max_output_tokens=8192 * 2,
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
            prompt_data = load_prompt(
                prompt_name, base_dir=Path(__file__).parent
            )
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

    def _process_audio(self, raw_path: Path) -> List[Dict[str, object]] | None:
        """Process audio file and return segments for transcription."""
        try:
            data, sr = sf.read(raw_path, dtype="float32")

            mic_ranges: List[tuple[float, float]] = []
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

            # Extract timestamp from filename
            time_part = raw_path.stem.split("_")[0]
            today = datetime.date.today().strftime("%Y%m%d")
            end_dt = datetime.datetime.strptime(f"{today}_{time_part}", "%Y%m%d_%H%M%S")
            file_duration = len(data) / sr
            base_dt = end_dt - datetime.timedelta(seconds=file_duration)

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

    def _get_json_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSON path for an audio file."""
        if audio_path.name.endswith("_raw.flac"):
            json_name = audio_path.name.replace("_raw.flac", "_audio.json")
        elif audio_path.name.endswith("_audio.flac"):
            json_name = audio_path.name.replace("_audio.flac", "_audio.json")
        else:
            json_name = audio_path.stem + "_audio.json"
        return audio_path.with_name(json_name)

    def _transcribe(self, raw_path: Path, segments: List[Dict[str, object]]) -> bool:
        """Transcribe segments using Gemini and save JSON."""
        json_path = self._get_json_path(raw_path)

        try:
            entity_names = load_entity_names(spoken=True)
            if entity_names:
                entities_str = ", ".join(entity_names)
                entities_text = f"Known entities: {entities_str}"
            else:
                entities_text = ""

            result = transcribe_segments(
                self.client, MODEL, self.prompt_text, entities_text, segments
            )
            json_path.write_text(json.dumps({"text": result}, indent=2))
            logging.info(f"Transcribed {raw_path} -> {json_path}")

            crumb_builder = (
                CrumbBuilder()
                .add_file(str(self.prompt_path))
                .add_file(self.journal_dir / "entities.md")
            )
            crumb_builder = crumb_builder.add_file(raw_path).add_model(MODEL)
            crumb_path = crumb_builder.commit(str(json_path))
            logging.info(f"Crumb saved to {crumb_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
            return False

    def _handle_raw(self, raw_path: Path) -> None:
        """Process a raw audio file."""
        # Skip if already processed
        json_path = self._get_json_path(raw_path)
        if json_path.exists():
            logging.info(f"Already processed, moving to heard: {raw_path}")
            self._move_to_heard(raw_path)
            return

        # Process audio
        segments = self._process_audio(raw_path)
        if segments is None:
            return

        # Transcribe
        success = self._transcribe(raw_path, segments)
        if success:
            self._move_to_heard(raw_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file to process",
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

    logging.info(f"Processing audio: {audio_path}")

    transcriber = Transcriber(journal, api_key)
    transcriber._handle_raw(audio_path)


if __name__ == "__main__":
    main()
