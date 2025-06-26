import argparse
import datetime
import faulthandler
import io
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from noisereduce import reduce_noise
from pyaec import Aec
from scipy.fft import irfft, rfft
from silero_vad import get_speech_timestamps, load_silero_vad
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from think.crumbs import CrumbBuilder

# Constants
MODEL = "gemini-2.5-pro"
SAMPLE_RATE = 16000
SYS_DELAY_MS = 100
MIN_SPEECH_SECONDS = 1.0


class Transcriber:
    def __init__(
        self,
        journal_dir: Path,
        api_key: str,
        prompt_path: Path,
        entities_path: Path,
        voice_sample: Optional[Path] = None,
    ):
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.client = genai.Client(api_key=api_key)
        self.prompt_path = prompt_path
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_path = entities_path
        self.voice_sample_path = voice_sample or journal_dir / "voice_sample.flac"
        self.voice_sample_bytes: Optional[bytes] = None
        if self.voice_sample_path.is_file():
            try:
                self.voice_sample_bytes = self.voice_sample_path.read_bytes()
                logging.info(f"Loaded voice sample from {self.voice_sample_path}")
            except Exception as e:  # pragma: no cover - best effort
                logging.warning(f"Failed to load voice sample: {e}")
        self.model = load_silero_vad()
        self.min_freq = 300
        self.max_freq = 3400
        self.boost_factor = 2
        self.frame_size = int(0.02 * SAMPLE_RATE)
        self.filter_length = int(SAMPLE_RATE * 0.2)
        self.aec = Aec(self.frame_size, self.filter_length, SAMPLE_RATE, True)
        self.mic_stash = np.array([], dtype=np.float32)
        self.sys_stash = np.array([], dtype=np.float32)
        self.processed: set[str] = set()
        self.observer: Optional[Observer] = None
        self.executor = ThreadPoolExecutor()
        self.attempts: dict[str, int] = {}

    def enhance_voice(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=1e10, neginf=-1e10)
        audio = np.where(audio == 0, 1e-10, audio)
        audio = reduce_noise(y=audio, sr=SAMPLE_RATE)
        X = rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1 / SAMPLE_RATE)
        vocal_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        X[vocal_mask] *= self.boost_factor
        enhanced = irfft(X)
        return enhanced.astype(np.float32)

    def detect_speech(self, label: str, buffer_data: np.ndarray):
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
                f"Detected {len(speech_segments)} speech segments in {label} of {buffer_seconds:.1f} seconds."
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
                segments.append({"offset": seg["start"], "data": seg_data})
            return segments, unprocessed_data
        except Exception as e:
            logging.error(f"Error in detect_speech for {label}: {e}")
            return [], np.array([], dtype=np.float32)

    @staticmethod
    def calculate_rms(audio_buffer: np.ndarray) -> float:
        if len(audio_buffer) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_buffer))))

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_rms: float) -> np.ndarray:
        if len(audio) == 0 or target_rms == 0:
            return audio
        current_rms = Transcriber.calculate_rms(audio)
        if current_rms == 0:
            return audio
        gain_factor = target_rms / current_rms
        return audio * gain_factor

    def is_system_muted(self, audio_buffer: np.ndarray | None = None) -> bool:
        try:
            result = subprocess.run(
                ["pactl", "get-sink-mute", "@DEFAULT_SINK@"],
                capture_output=True,
                text=True,
                check=True,
            )
            pactl_muted = "Mute: yes" in result.stdout
        except subprocess.SubprocessError as e:
            logging.error(f"Error checking system mute status: {e}")
            pactl_muted = False

        silent = False
        if audio_buffer is not None and len(audio_buffer) > 0:
            rms = self.calculate_rms(audio_buffer)
            silent = rms < 0.0001
            if silent:
                logging.info(f"System audio silent (RMS: {rms:.6f})")

        return pactl_muted or silent

    def apply_echo_cancellation(self, mic_buffer: np.ndarray, sys_buffer: np.ndarray) -> np.ndarray:
        min_length = min(len(mic_buffer), len(sys_buffer))
        if min_length == 0:
            return mic_buffer
        mic_int16 = (mic_buffer * 32767).astype(np.int16)
        sys_int16 = (sys_buffer * 32767).astype(np.int16)
        processed_chunks = []
        for i in range(0, min_length, self.frame_size):
            mic_frame = mic_int16[i : i + self.frame_size]
            sys_frame = sys_int16[i : i + self.frame_size]
            if min(len(mic_frame), len(sys_frame)) < self.frame_size:
                processed_chunks.append(mic_frame)
            else:
                processed_frame = self.aec.cancel_echo(mic_frame, sys_frame)
                processed_chunks.append(processed_frame)
        processed_mic = np.concatenate(processed_chunks).astype(np.float32) / 32767.0
        processed_mic = self.enhance_voice(processed_mic)
        return processed_mic

    def create_flac_bytes(self, left_data: np.ndarray) -> bytes:
        if left_data is None or left_data.size == 0:
            return b""
        left_int16 = (np.clip(left_data, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, left_int16, SAMPLE_RATE, format="FLAC")
        return buf.getvalue()

    def process_segments_and_save(self, segments, output_path: Path) -> float:
        if not segments:
            return 0.0
        total_seconds = sum([len(seg["data"]) / SAMPLE_RATE for seg in segments])
        if total_seconds < MIN_SPEECH_SECONDS:
            return 0.0
        combined_data_list = []
        for seg in segments:
            data = seg.get("data")
            if isinstance(data, np.ndarray) and data.size > 0:
                combined_data_list.append(data)
        if not combined_data_list:
            return 0.0
        combined_audio = np.concatenate(combined_data_list)
        flac_bytes = self.create_flac_bytes(combined_audio)
        output_path.write_bytes(flac_bytes)
        logging.info(f"Saved processed audio to {output_path}")
        return total_seconds

    def process_buffer(self, buffer: np.ndarray, new_data: np.ndarray, label: str):
        merged = np.concatenate((buffer, new_data)) if buffer.size > 0 else new_data
        return self.detect_speech(label, merged)

    def process_raw_file(self, raw_path: Path) -> Path | None:
        data, sr = sf.read(raw_path, dtype="float32")
        if sr != SAMPLE_RATE:
            logging.warning(f"Unexpected sample rate {sr} in {raw_path}")
        if data.ndim == 1:
            mic_new = data
            sys_new = np.array([], dtype=np.float32)
        else:
            mic_new = data[:, 0]
            sys_new = data[:, 1]

        sys_segments, self.sys_stash = self.process_buffer(self.sys_stash, sys_new, "sys")
        system_muted = self.is_system_muted(sys_new)
        if system_muted:
            mic_segments, self.mic_stash = self.process_buffer(self.mic_stash, mic_new, "mic")
            segments_all = mic_segments + sys_segments
        else:
            mic_processed = self.apply_echo_cancellation(mic_new, sys_new)
            mic_segments, self.mic_stash = self.process_buffer(self.mic_stash, mic_processed, "mic")
            segments_all = mic_segments + sys_segments
        if segments_all:
            segments_all.sort(key=lambda seg: seg["offset"])
        audio_path = raw_path.with_name(raw_path.name.replace("_raw.flac", "_audio.flac"))
        seconds = self.process_segments_and_save(segments_all, audio_path)
        if seconds > 0:
            return audio_path
        raw_path.unlink(missing_ok=True)
        return None

    def _process_once(self, audio_path: Path, json_path: Path) -> None:
        result = self.transcribe_file(audio_path)
        json_path.write_text(json.dumps({"text": result}, indent=2))
        logging.info(f"Transcribed {audio_path} -> {json_path}")
        crumb_builder = CrumbBuilder().add_file(self.prompt_path).add_file(self.entities_path)
        if self.voice_sample_bytes:
            crumb_builder = crumb_builder.add_file(self.voice_sample_path)
        crumb_builder = crumb_builder.add_file(audio_path).add_model(MODEL)
        crumb_path = crumb_builder.commit(str(json_path))
        logging.info(f"Crumb saved to {crumb_path}")

    def _process(self, raw_path: Path) -> None:
        audio_path = self.process_raw_file(raw_path)
        if not audio_path:
            return
        json_path = audio_path.with_suffix(".json")
        if json_path.exists() or json_path.name in self.processed:
            self.processed.add(json_path.name)
            return
        attempts = 0
        while attempts < 2:
            try:
                self._process_once(audio_path, json_path)
                break
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {audio_path} due to error: {e}")
                else:
                    logging.error(f"Failed to transcribe {audio_path}: {e}")
        self.processed.add(json_path.name)

    def transcribe_file(self, flac_path: Path) -> dict:
        logging.info(f"Processing {flac_path}")
        flac_bytes = flac_path.read_bytes()
        user_prompt = "Process the provided audio now and output your professional accurate transcription in the specified JSON format."
        entities_text = self.entities_path.read_text().strip()
        contents = [entities_text]
        if self.voice_sample_bytes:
            contents.append(
                "Here's a voice sample for Jeremie in case you hear him, but you must be very confident that the voice you hear matches him (there's often many other speakers, and some sound like him), then only if you are sure, you can tag his sections with his name as the speaker."
            )
            contents.append(
                types.Part.from_bytes(data=self.voice_sample_bytes, mime_type="audio/flac")
            )
        contents.append(user_prompt)
        contents.append(types.Part.from_bytes(data=flac_bytes, mime_type="audio/flac"))

        response = self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192 * 2,
                response_mime_type="application/json",
                system_instruction=self.prompt_text,
            ),
        )
        result = json.loads(response.text)
        logging.info(f"Transcription result: {json.dumps(result, indent=2)}")
        return result

    def start(self):
        handler = PatternMatchingEventHandler(patterns=["*_raw.flac"], ignore_directories=True)

        def on_created(event):
            raw_path = Path(event.src_path)
            logging.info(f"New raw audio file detected: {raw_path}")
            self.executor.submit(self._process, raw_path)

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
    parser.add_argument(
        "-e",
        "--entities",
        type=Path,
        help="Path to master entities file (defaults to <journal>/entities.md)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    ent_path = args.entities or args.journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    transcriber = Transcriber(args.journal, api_key, args.prompt, ent_path)
    transcriber.start()


if __name__ == "__main__":
    main()
