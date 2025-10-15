"""Audio recording utilities for observe package."""

import gc
import io
import logging
import os
import signal
import threading
import time
from queue import Queue

import numpy as np
import soundfile as sf

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
