import argparse
import datetime
import faulthandler
import io
import logging
import os
import sys
import threading
import time
from queue import Queue

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from hear.input_detect import input_detect

# Constants
SAMPLE_RATE = 48000
CHANNELS = 1
BLOCK_SIZE = 1024
SYS_DELAY_MS = 100  # System audio delay in milliseconds to align with microphone


class AudioRecorder:
    def __init__(
        self,
        journal=None,
        debug=False,
        timer_interval=60,
    ):
        self.save_dir = journal or os.getcwd()
        # Queue holds tuples of (mic_chunk, sys_chunk) to keep the two streams
        # aligned as they were recorded.
        self.audio_queue = Queue()
        self._running = True
        self.debug = debug
        self.timer_interval = timer_interval

    def detect(self):
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
        try:
            with (
                self.mic_device.recorder(
                    samplerate=SAMPLE_RATE, channels=[-1], blocksize=BLOCK_SIZE
                ) as mic_rec,
                self.sys_device.recorder(
                    samplerate=SAMPLE_RATE, channels=[-1], blocksize=BLOCK_SIZE
                ) as sys_rec,
            ):
                while self._running:
                    try:
                        mic_chunk = mic_rec.record(numframes=BLOCK_SIZE)
                        sys_chunk = sys_rec.record(numframes=BLOCK_SIZE)

                        if mic_chunk is None or mic_chunk.size == 0:
                            logging.warning("Captured empty microphone buffer")
                        if sys_chunk is None or sys_chunk.size == 0:
                            logging.warning("Captured empty system buffer")

                        # Store both chunks together so they remain aligned
                        self.audio_queue.put((mic_chunk, sys_chunk))
                    except Exception as e:
                        logging.error(f"Error recording audio: {e}")
                        if not self._running:
                            break
                        time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error setting up recorders: {e}")
            if self._running:
                logging.error(f"Recording thread crashed: {e}")

    def get_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return concatenated system and microphone data from the queue."""
        mic_buffer = np.array([], dtype=np.float32)
        sys_buffer = np.array([], dtype=np.float32)

        while not self.audio_queue.empty():
            mic_chunk, sys_chunk = self.audio_queue.get()

            if mic_chunk is None or mic_chunk.size == 0:
                logging.warning("Queue contained empty microphone chunk")
            else:
                if mic_chunk.ndim > 1:
                    mic_chunk = np.mean(mic_chunk, axis=1).flatten()
                else:
                    mic_chunk = mic_chunk.flatten()
                mic_buffer = np.concatenate((mic_buffer, mic_chunk))

            if sys_chunk is None or sys_chunk.size == 0:
                logging.warning("Queue contained empty system chunk")
            else:
                if sys_chunk.ndim > 1:
                    sys_chunk = np.mean(sys_chunk, axis=1).flatten()
                else:
                    sys_chunk = sys_chunk.flatten()
                sys_buffer = np.concatenate((sys_buffer, sys_chunk))

        if mic_buffer.size > 0:
            mic_buffer = np.nan_to_num(mic_buffer, nan=0.0, posinf=1e10, neginf=-1e10)
        if sys_buffer.size > 0:
            sys_buffer = np.nan_to_num(sys_buffer, nan=0.0, posinf=1e10, neginf=-1e10)

        if mic_buffer.size == 0 and sys_buffer.size == 0:
            logging.warning("No valid audio data retrieved from queue")

        return sys_buffer, mic_buffer

    def create_flac_bytes(self, left_data: np.ndarray, right_data: np.ndarray = None) -> bytes:
        """Create FLAC bytes from audio data. If right_data is provided, creates stereo output with right channel delay alignment."""
        if (left_data is None or left_data.size == 0) and (
            right_data is None or right_data.size == 0
        ):
            logging.warning("Audio data is empty. Returning empty bytes.")
            return b""

        if right_data is None or right_data.size == 0:
            audio_data = (np.clip(left_data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            # Stereo output with system delay alignment
            delay_samples = int(SYS_DELAY_MS * SAMPLE_RATE / 1000)
            max_length = (
                max(len(left_data), len(right_data) + delay_samples)
                if right_data.size > 0
                else len(left_data)
            )

            left_channel = np.zeros(max_length, dtype=np.float32)
            right_channel = np.zeros(max_length, dtype=np.float32)

            left_channel[: len(left_data)] = left_data
            right_channel[delay_samples : delay_samples + len(right_data)] = right_data

            left_int16 = (np.clip(left_channel, -1.0, 1.0) * 32767).astype(np.int16)
            right_int16 = (np.clip(right_channel, -1.0, 1.0) * 32767).astype(np.int16)

            # Interleave channels for stereo output
            audio_data = np.column_stack((left_int16, right_int16))

        buf = io.BytesIO()
        try:
            sf.write(buf, audio_data, SAMPLE_RATE, format="FLAC")
        except Exception as e:
            logging.error(
                f"Error creating FLAC: {e}. Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}"
            )
            return b""

        return buf.getvalue()

    def speech_processing_timer(self):
        while self._running:
            time.sleep(self.timer_interval)
            sys_data, mic_data = self.get_buffers()
            if sys_data.size == 0 and mic_data.size == 0:
                logging.warning("Timer fired with no audio data")
                continue
            raw_bytes = self.create_flac_bytes(mic_data, sys_data)
            self.save_flac(raw_bytes, suffix="raw")

    def save_flac(self, flac_bytes, suffix="_audio"):
        """Save the audio to a dated directory."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        date_part, time_part = timestamp.split("_", 1)
        day_dir = os.path.join(self.save_dir, date_part)
        os.makedirs(day_dir, exist_ok=True)

        flac_filepath = os.path.join(day_dir, f"{time_part}_{suffix}.flac")
        with open(flac_filepath, "wb") as f:
            f.write(flac_bytes)
        logging.info(f"Saved audio to {flac_filepath}")

    def start(self):
        threads = [
            threading.Thread(
                target=self.record_both,
                daemon=True,
            ),
            threading.Thread(target=self.speech_processing_timer, daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._running = False
            logging.info("\nRecording stopped (Ctrl+C pressed)")
        except Exception as e:
            self._running = False
            logging.error(f"Error during recording: {e}")


def main():
    # 1. Load environment
    load_dotenv()

    # 2. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Record audio and save FLAC files.")
    parser.add_argument(
        "journal", nargs="?", default=None, help="Journal directory to store audio recordings"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug mode (save audio buffers)."
    )
    parser.add_argument(
        "-t", "--timer_interval", type=int, default=60, help="Timer interval in seconds."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level)

    # Create save directory if needed
    if args.journal and not os.path.exists(args.journal):
        os.makedirs(args.journal)

    # Enable faulthandler to help diagnose crashes
    faulthandler.enable()

    # 4. Create the recorder
    recorder = AudioRecorder(
        journal=args.journal,
        debug=args.debug,
        timer_interval=args.timer_interval,
    )

    # 5. Detect devices or exit
    if not recorder.detect():
        sys.exit("No suitable audio devices found.")

    # 6. Start recording
    recorder.start()


if __name__ == "__main__":
    main()
