import argparse
import asyncio
import datetime
import faulthandler
import gc
import io
import logging
import os
import sys
import threading
import time
from queue import Queue
from typing import Optional, Set

import numpy as np
import soundfile as sf
import websockets

from hear.input_detect import input_detect
from think.utils import setup_cli

# Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


class AudioRecorder:
    def __init__(
        self,
        journal=None,
        debug=False,
        timer_interval=60,
        websocket_port=None,
    ):
        self.save_dir = journal or os.getcwd()
        # Queue now holds stereo chunks (mic=left, sys=right)
        self.audio_queue = Queue()
        self._running = True
        self.debug = debug
        self.timer_interval = timer_interval
        self.websocket_port = websocket_port
        self.ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self.ws_clients: Set[websockets.WebSocketServerProtocol] = set()

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

                            if mic_chunk is None or mic_chunk.size != BLOCK_SIZE:
                                logging.warning("Bad microphone buffer")
                                continue
                            if sys_chunk is None or sys_chunk.size != BLOCK_SIZE:
                                logging.warning("Bad system buffer")
                                continue

                            stereo_chunk = np.column_stack((mic_chunk, sys_chunk))
                            self.audio_queue.put(stereo_chunk)
                            block_count += 1
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

    async def _ws_handler(self, websocket):
        self.ws_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.ws_clients.remove(websocket)

    def websocket_server(self):
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)

        async def start_server():
            server = await websockets.serve(
                self._ws_handler, "0.0.0.0", self.websocket_port
            )
            await server.wait_closed()

        self.ws_loop.run_until_complete(start_server())

    def broadcast_audio(self, stereo_data: np.ndarray):
        if not (self.ws_loop and self.ws_clients and stereo_data.size > 0):
            return
        audio_bytes = stereo_data.astype(np.float32).tobytes()
        for ws in list(self.ws_clients):
            asyncio.run_coroutine_threadsafe(ws.send(audio_bytes), self.ws_loop)

    def speech_processing_timer(self):
        accumulated_data = np.array([], dtype=np.float32).reshape(0, 2)
        last_save_time = time.time()

        while self._running:
            time.sleep(1)
            stereo_data = self.get_buffers()

            if stereo_data.size > 0:
                accumulated_data = np.vstack((accumulated_data, stereo_data))
            self.broadcast_audio(stereo_data)

            current_time = time.time()
            if current_time - last_save_time >= self.timer_interval:
                if accumulated_data.size == 0:
                    logging.warning(
                        "Timer interval elapsed with no accumulated audio data"
                    )
                else:
                    raw_bytes = self.create_flac_bytes(accumulated_data)
                    self.save_flac(raw_bytes, suffix="raw")
                    accumulated_data = np.array([], dtype=np.float32).reshape(0, 2)
                last_save_time = current_time

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
            threading.Thread(target=self.record_both, daemon=True),
            threading.Thread(target=self.speech_processing_timer, daemon=True),
        ]
        if self.websocket_port:
            threads.append(threading.Thread(target=self.websocket_server, daemon=True))
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
    parser = argparse.ArgumentParser(description="Record audio and save FLAC files.")
    parser.add_argument(
        "-t",
        "--timer_interval",
        type=int,
        default=60,
        help="Timer interval in seconds.",
    )
    parser.add_argument(
        "--ws-port", type=int, default=None, help="Start websocket server on this port"
    )
    args = setup_cli(parser)

    journal = os.getenv("JOURNAL_PATH")
    if not os.path.exists(journal):
        os.makedirs(journal)

    # Enable faulthandler to help diagnose crashes
    faulthandler.enable()

    # 4. Create the recorder
    recorder = AudioRecorder(
        journal=journal,
        debug=args.debug,
        timer_interval=args.timer_interval,
        websocket_port=args.ws_port,
    )

    # 5. Detect devices or exit
    if not recorder.detect():
        sys.exit("No suitable audio devices found.")

    # 6. Start recording
    recorder.start()


if __name__ == "__main__":
    main()
