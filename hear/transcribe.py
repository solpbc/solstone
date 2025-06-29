import argparse
import datetime
import faulthandler
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from deepfilternet import DFModel, DFStreamer
from dotenv import load_dotenv
from google import genai
from google.genai import types
from silero_vad import get_speech_timestamps, load_silero_vad
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from think.crumbs import CrumbBuilder

# Constants
MODEL = "gemini-2.5-flash"
SAMPLE_RATE = 16000
MIN_SPEECH_SECONDS = 1.0

# DeepFilterNet denoiser setup
denoise_model = DFModel.from_pretrained("dns64")
streamer = DFStreamer(denoise_model)


def denoise(audio_16k: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply DeepFilterNet and return denoised PCM at the original rate."""
    audio_48k = librosa.resample(audio_16k, orig_sr=sr, target_sr=48000)
    clean_48k = streamer.process(audio_48k)
    return librosa.resample(clean_48k, orig_sr=48000, target_sr=sr)


def merge_streams(
    sys_data: np.ndarray,
    mic_data: np.ndarray,
    sample_rate: int,
    window_ms: int = 50,
    threshold: float = 0.005,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mix system and microphone audio while avoiding feedback.

    Returns a tuple with the merged audio and a list of time ranges (in seconds)
    where the microphone contained significant audio based on ``threshold``.
    """

    length = min(len(sys_data), len(mic_data))
    if length == 0:
        return np.array([], dtype=np.float32), []

    sys_data = sys_data[:length]
    mic_data = mic_data[:length]

    # If the system channel contains only silence, skip merging and
    # treat the entire segment as microphone audio.
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
                    range_start = start - window_samples  # Start from previous window
            else:
                consecutive_mic_windows = 0

    if in_range:
        mic_ranges.append((range_start / sample_rate, length / sample_rate))

    # Filter out mic ranges shorter than 1 second
    mic_ranges = [(start, end) for start, end in mic_ranges if end - start >= 1.0]

    return output, mic_ranges


class Transcriber:
    def __init__(
        self,
        journal_dir: Path,
        api_key: str,
        prompt_path: Path,
    ):
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.client = genai.Client(api_key=api_key)
        self.prompt_path = prompt_path
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_path = journal_dir / "entities.md"
        self.model = load_silero_vad()
        self.merged_stash = np.array([], dtype=np.float32)
        self.processed: set[str] = set()
        self.observer: Optional[Observer] = None
        self.executor = ThreadPoolExecutor()
        self.attempts: dict[str, int] = {}

    def _calculate_mic_overlap(
        self, seg_start: float, seg_end: float, mic_ranges: List[Tuple[float, float]]
    ) -> float:
        """Calculate what percentage of a segment overlaps with mic ranges."""
        if not mic_ranges:
            return 0.0

        seg_duration = seg_end - seg_start
        if seg_duration <= 0:
            return 0.0

        overlap_duration = 0.0
        for mic_start, mic_end in mic_ranges:
            # Calculate overlap between segment and this mic range
            overlap_start = max(seg_start, mic_start)
            overlap_end = min(seg_end, mic_end)
            if overlap_start < overlap_end:
                overlap_duration += overlap_end - overlap_start

        return overlap_duration / seg_duration

    def detect_speech(
        self,
        label: str,
        buffer_data: np.ndarray,
        mic_ranges: Optional[List[Tuple[float, float]]] = None,
    ):
        if buffer_data is None or len(buffer_data) == 0:
            logging.info(f"No audio data found in {label} buffer.")
            return [], np.array([], dtype=np.float32)
        try:
            speech_segments = get_speech_timestamps(
                buffer_data,
                self.model,
                sampling_rate=SAMPLE_RATE,
                return_seconds=True,
                speech_pad_ms=70,
                min_silence_duration_ms=100,
                min_speech_duration_ms=200,
                threshold=0.3,
            )
            buffer_seconds = len(buffer_data) / SAMPLE_RATE
            logging.info(
                "Detected %d speech segments in %s of %.1f seconds",
                len(speech_segments),
                label,
                buffer_seconds,
            )
            segments = []
            total_duration = len(buffer_data) / SAMPLE_RATE
            unprocessed_data = np.array([], dtype=np.float32)
            for i, seg in enumerate(speech_segments):
                if i == len(speech_segments) - 1 and total_duration - seg["end"] < 1:
                    start_idx = int(seg["start"] * SAMPLE_RATE)
                    unprocessed_data = buffer_data[start_idx:]
                    break
                start_idx = int(seg["start"] * SAMPLE_RATE)
                end_idx = int(seg["end"] * SAMPLE_RATE)
                seg_data = buffer_data[start_idx:end_idx]

                # Calculate mic overlap percentage and tag if > 80%
                mic_overlap = self._calculate_mic_overlap(
                    seg["start"], seg["end"], mic_ranges or []
                )
                in_mic = mic_overlap > 0.8

                segments.append({"offset": seg["start"], "data": seg_data, "mic": in_mic})
            return segments, unprocessed_data
        except Exception as e:
            logging.error(f"Error in detect_speech for {label}: {e}")
            # Re-raise the exception instead of returning empty results
            # This prevents file deletion when there's a processing error
            raise

    def _process_raw(self, raw_path: Path) -> List[Dict[str, object]] | None:
        try:
            data, sr = sf.read(raw_path, dtype="float32")
            streamer.reset_state()
            if sr != SAMPLE_RATE:
                logging.warning(f"Unexpected sample rate {sr} in {raw_path}")

            if data.ndim == 1:
                data = denoise(data, sr)
            else:
                data[:, 0] = denoise(data[:, 0], sr)
                data[:, 1] = denoise(data[:, 1], sr)

            mic_ranges: List[Tuple[float, float]] = []
            if data.ndim == 1:
                merged = data
            else:
                mic_new = data[:, 0]
                sys_new = data[:, 1]
                merged, mic_ranges = merge_streams(sys_new, mic_new, SAMPLE_RATE)

            offset_seconds = len(self.merged_stash) / SAMPLE_RATE
            adjusted_ranges = [(s + offset_seconds, e + offset_seconds) for s, e in mic_ranges]

            merged = np.concatenate((self.merged_stash, merged))
            segments, self.merged_stash = self.detect_speech("mix", merged, adjusted_ranges)

            if not segments:
                logging.info(f"No speech segments detected, removing {raw_path}")
                raw_path.unlink(missing_ok=True)
                return None

            total_seconds = sum(len(seg["data"]) / SAMPLE_RATE for seg in segments)
            if total_seconds < MIN_SPEECH_SECONDS:
                logging.info(
                    "Total speech duration %.2fs < %.2fs, removing %s",
                    total_seconds,
                    MIN_SPEECH_SECONDS,
                    raw_path,
                )
                raw_path.unlink(missing_ok=True)
                return None

            day = raw_path.parent.name
            time_part = raw_path.stem.replace("_raw", "")
            end_dt = datetime.datetime.strptime(f"{day}_{time_part}", "%Y%m%d_%H%M%S")
            file_duration = len(data) / SAMPLE_RATE
            base_dt = end_dt - datetime.timedelta(seconds=file_duration + offset_seconds)

            processed: List[Dict[str, object]] = []
            for seg in segments:
                start_dt = base_dt + datetime.timedelta(seconds=seg["offset"])
                start_str = start_dt.strftime("%H:%M:%S")
                audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
                processed.append(
                    {
                        "start": start_str,
                        "source": "mic" if seg.get("mic") else "sys",
                        "bytes": buf.getvalue(),
                    }
                )

            logging.info(f"Processed {raw_path}: {len(processed)} segments")
            return processed
        except Exception as e:
            logging.error(f"Error processing {raw_path}: {e}")
            return None

    def _transcribe_segments(self, raw_path: Path, segments: List[Dict[str, object]]) -> bool:
        json_path = raw_path.with_name(raw_path.name.replace("_raw.flac", "_audio.json"))
        attempts = 0
        while attempts < 2:
            try:
                result = self._transcribe(segments)
                json_path.write_text(json.dumps({"text": result}, indent=2))
                logging.info(f"Transcribed {raw_path} -> {json_path}")

                crumb_builder = (
                    CrumbBuilder().add_file(self.prompt_path).add_file(self.entities_path)
                )
                crumb_builder = crumb_builder.add_file(raw_path).add_model(MODEL)
                crumb_path = crumb_builder.commit(str(json_path))
                logging.info(f"Crumb saved to {crumb_path}")
                return True
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {raw_path} due to error: {e}")
                else:
                    logging.error(f"Failed to transcribe {raw_path}: {e}")
                    return False
        return False

    def _handle_raw(self, raw_path: Path) -> None:
        segments = self._process_raw(raw_path)
        if segments is None:
            return
        json_path = raw_path.with_name(raw_path.name.replace("_raw.flac", "_audio.json"))
        if json_path.exists() or json_path.name in self.processed:
            self.processed.add(json_path.name)
            return

        if self._transcribe_segments(raw_path, segments):
            self.processed.add(json_path.name)

    def _transcribe(self, segments: List[Dict[str, object]]) -> dict:
        user_prompt = (
            "Process the provided audio clips now and output your professional "
            "accurate transcription in the specified JSON format, each clip may contain one or more speakers."
        )
        entities_text = self.entities_path.read_text().strip()
        contents = [entities_text, user_prompt]

        for seg in segments:
            contents.append(
                f"This clip starts at {seg['start']} and the source is '{seg['source']}'."
            )
            contents.append(types.Part.from_bytes(data=seg["bytes"], mime_type="audio/flac"))

        response = self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192 * 2,
                response_mime_type="application/json",
                system_instruction=self.prompt_text,
            ),
        )
        result = json.loads(response.text)
        logging.info(f"Transcription result: {json.dumps(result, indent=2)}")
        return result

    def repair_day(self, date_str: str):
        """Repair incomplete processing for a specific day."""
        day_dir = self.journal_dir / date_str
        if not day_dir.exists():
            logging.error(f"Day directory {day_dir} does not exist")
            return

        logging.info(f"Repairing day {date_str} in {day_dir}")

        repair_files = []
        repair_files.extend(day_dir.glob("*_raw.flac"))
        repair_files.extend(day_dir.glob("*_audio.flac"))
        repair_files.extend(day_dir.glob("*_audio.ogg"))

        missing = []
        for audio_path in repair_files:
            # Generate JSON path: change extension to .json and replace _raw with _audio
            json_name = audio_path.stem.replace("_raw", "_audio") + ".json"
            json_path = audio_path.with_name(json_name)
            if not json_path.exists():
                missing.append(audio_path)

        logging.info(f"Found {len(missing)} audio files missing transcripts")

        for audio_path in missing:
            logging.info(f"Processing audio file: {audio_path}")
            self._handle_raw(audio_path)

    def start(self):
        handler = PatternMatchingEventHandler(patterns=["*_raw.flac"], ignore_directories=True)

        def on_created(event):
            raw_path = Path(event.src_path)
            logging.info(f"New raw audio file detected: {raw_path}")
            self.executor.submit(self._handle_raw, raw_path)

        handler.on_created = on_created

        self.observer = None
        current_day: Optional[str] = None
        try:
            while True:
                today_str = datetime.datetime.now().strftime("%Y%m%d")
                day_dir = self.journal_dir / today_str
                if day_dir.exists() and (current_day != today_str):
                    if self.observer:
                        self.observer.stop()
                        self.observer.join()
                    self.observer = Observer()
                    self.observer.schedule(handler, str(day_dir), recursive=True)
                    self.observer.start()
                    self.watch_dir = day_dir
                    self.processed.clear()
                    current_day = today_str
                    logging.info(f"Watching {day_dir}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            self.executor.shutdown(wait=True)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument(
        "journal", type=Path, help="Journal directory containing daily audio folders"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("transcribe.txt"),
        help="Path to the system prompt text",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--repair",
        type=str,
        help="Repair mode: process incomplete files for specified day (YYYYMMDD format)",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    ent_path = args.journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    transcriber = Transcriber(args.journal, api_key, args.prompt)

    if args.repair:
        # Validate date format
        try:
            datetime.datetime.strptime(args.repair, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.repair}. Use YYYYMMDD format.")
        transcriber.repair_day(args.repair)
    else:
        transcriber.start()


if __name__ == "__main__":
    main()
