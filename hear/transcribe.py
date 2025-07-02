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
from dotenv import load_dotenv
from google import genai
from silero_vad import load_silero_vad
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from hear.audio_utils import SAMPLE_RATE, detect_speech, merge_streams, resample_audio
from hear.gemini import transcribe_segments
from think.crumbs import CrumbBuilder

# Constants
MODEL = "gemini-2.5-flash"
MIN_SPEECH_SECONDS = 1.0


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
        from df.enhance import init_df

        self._df_model, self._df_state, *_ = init_df(post_filter=True)
        self._df_sr = self._df_state.sr()

    def _trash_file(self, raw_path: Path) -> None:
        """Move the given file to a trash directory inside its day folder."""
        trash_dir = raw_path.parent / "trash"
        try:
            trash_dir.mkdir(exist_ok=True)
            raw_path.rename(trash_dir / raw_path.name)
        except Exception as e:
            logging.error(f"Failed to move {raw_path} to trash: {e}")

    def _process_raw(self, raw_path: Path) -> List[Dict[str, object]] | None:
        try:
            data, sr = sf.read(raw_path, dtype="float32")

            mic_ranges: List[Tuple[float, float]] = []
            if data.ndim == 1:
                merged = resample_audio(
                    data, sr, SAMPLE_RATE, self._df_model, self._df_state, self._df_sr
                )
                logging.info(f"Single channel audio detected in {raw_path} {sr}, no mic data.")
            else:
                logging.info(
                    f"Dual channel audio detected in {raw_path} {sr}, denoising mic channel."
                )
                mic_new = resample_audio(
                    data[:, 0],
                    sr,
                    SAMPLE_RATE,
                    self._df_model,
                    self._df_state,
                    self._df_sr,
                    denoise=True,
                )
                sys_new = resample_audio(
                    data[:, 1], sr, SAMPLE_RATE, self._df_model, self._df_state, self._df_sr
                )

                # when testing, save denoised mic audio for validation
                sf.write("./mic_denoise.flac", mic_new, SAMPLE_RATE)

                merged, mic_ranges = merge_streams(sys_new, mic_new, sr)

            offset_seconds = len(self.merged_stash) / SAMPLE_RATE
            adjusted_ranges = [(s + offset_seconds, e + offset_seconds) for s, e in mic_ranges]

            merged = np.concatenate((self.merged_stash, merged))
            segments, self.merged_stash = detect_speech(self.model, "mix", merged, adjusted_ranges)

            if not segments:
                # Don't trash if there's unfinished speech in the stash
                if len(self.merged_stash) > 0:
                    logging.info(
                        f"No complete segments but {len(self.merged_stash)/SAMPLE_RATE:.1f}s in stash, keeping {raw_path}"
                    )
                    return None
                logging.info(f"No speech segments detected, moving {raw_path} to trash")
                self._trash_file(raw_path)
                return None

            total_seconds = sum(len(seg["data"]) / SAMPLE_RATE for seg in segments)
            if total_seconds < MIN_SPEECH_SECONDS:
                # Don't trash if there's unfinished speech in the stash
                if len(self.merged_stash) > 0:
                    logging.info(
                        "Total speech duration %.2fs < %.2fs but %.1fs in stash, keeping %s",
                        total_seconds,
                        MIN_SPEECH_SECONDS,
                        len(self.merged_stash) / SAMPLE_RATE,
                        raw_path,
                    )
                    return None
                logging.info(
                    "Total speech duration %.2fs < %.2fs, moving %s to trash",
                    total_seconds,
                    MIN_SPEECH_SECONDS,
                    raw_path,
                )
                self._trash_file(raw_path)
                return None

            day = raw_path.parent.name
            # Handle both _raw and _audio suffixes
            time_part = raw_path.stem.replace("_raw", "").replace("_audio", "")
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
            logging.error(f"Error processing {raw_path}: {e}", exc_info=True)
            return None

    def _get_json_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSON path for an audio file."""
        if audio_path.name.endswith("_raw.flac"):
            json_name = audio_path.name.replace("_raw.flac", "_audio.json")
        elif audio_path.name.endswith("_audio.flac"):
            json_name = audio_path.name.replace("_audio.flac", "_audio.json")
        elif audio_path.name.endswith("_audio.ogg"):
            json_name = audio_path.name.replace("_audio.ogg", "_audio.json")
        else:
            # Fallback for other extensions
            json_name = audio_path.stem + "_audio.json"

        return audio_path.with_name(json_name)

    def _transcribe_segments(self, raw_path: Path, segments: List[Dict[str, object]]) -> bool:
        json_path = self._get_json_path(raw_path)
        attempts = 0
        while attempts < 2:
            try:
                entities_text = self.entities_path.read_text().strip()
                result = transcribe_segments(
                    self.client, MODEL, self.prompt_text, entities_text, segments
                )
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

        json_path = self._get_json_path(raw_path)
        if json_path.exists() or json_path.name in self.processed:
            self.processed.add(json_path.name)
            return

        if self._transcribe_segments(raw_path, segments):
            self.processed.add(json_path.name)

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
            json_path = self._get_json_path(audio_path)
            if not json_path.exists():
                missing.append(audio_path)

        # Sort by timestamp (HHMMSS) to process files in chronological order
        def extract_timestamp(path: Path) -> str:
            # Extract HHMMSS from filename like "HHMMSS_raw.flac" or "HHMMSS_audio.flac"
            stem = path.stem.replace("_raw", "").replace("_audio", "")
            return stem

        missing.sort(key=extract_timestamp)

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
        try:
            datetime.datetime.strptime(args.repair, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.repair}. Use YYYYMMDD format.")
        transcriber.repair_day(args.repair)
    else:
        transcriber.start()


if __name__ == "__main__":
    main()
