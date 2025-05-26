import os
import sys
import io
import faulthandler
import numpy as np
import soundcard as sc
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
from dotenv import load_dotenv
from google import genai
from google.genai import types
import threading
import datetime
import json
from noisereduce import reduce_noise
import time
from queue import Queue
from pyaec import Aec
from scipy.fft import rfft, irfft
import subprocess
import argparse
import logging
from audio_detect import audio_detect
from typing import List, Tuple

# Constants
CHUNK_DURATION = 5
SAMPLE_RATE = 16000
CHANNELS = 1

class AudioRecorder:
    def __init__(self, save_dir=None, debug=False, timer_interval=60, api_key=None):
        self.save_dir = save_dir or os.getcwd()
        self.model = load_silero_vad()
        self.client = genai.Client(api_key=api_key)
        self.mic_queue = Queue()
        self.sys_queue = Queue()
        self._running = True
        self.debug = debug
        
        # Voice enhancement parameters
        self.min_freq = 300
        self.max_freq = 3400
        self.boost_factor = 2
        
        # PyAEC parameters
        self.frame_size = int(0.02 * SAMPLE_RATE)
        self.filter_length = int(SAMPLE_RATE * 0.2)
        self.aec = Aec(self.frame_size, self.filter_length, SAMPLE_RATE, True)
        self.timer_interval = timer_interval

    def enhance_voice(self, audio):
        if len(audio) == 0:
            return audio

        # clean edges
        audio = np.nan_to_num(audio, nan=0.0, posinf=1e10, neginf=-1e10)
        audio = np.where(audio == 0, 1e-10, audio)

        # Apply noise reduction
        audio = reduce_noise(y=audio, sr=SAMPLE_RATE)

        # Compute the FFT
        X = rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1/SAMPLE_RATE)
        
        # Define vocal range mask
        vocal_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        # Apply boost to the vocal range frequencies
        X[vocal_mask] *= self.boost_factor
        
        # Reconstruct the time-domain signal via inverse FFT
        enhanced = irfft(X)
        
        # Return the enhanced audio as float32 in the same range as input
        return enhanced.astype(np.float32)

    def detect(self):
        mic, loopback = audio_detect()
        if mic is None or loopback is None:
            logging.error(f"Detection failed: mic {mic} sys {loopback}")
            return False
        logging.info(f"Detected microphone: {mic.name}")
        logging.info(f"Detected system audio: {loopback.name}")
        self.mic_device = mic
        self.sys_device = loopback
        return True

    def record_device(self, device, queue, label):
        try:
            with device.recorder(samplerate=SAMPLE_RATE, channels=[-1]) as recorder:
                while self._running:
                    try:
                        recording = recorder.record(numframes=None)
                        if recording is not None and recording.size > 0:
                            queue.put(recording)
                    except Exception as e:
                        logging.error(f"Error recording from {label}: {e}")
                        if not self._running:
                            break
                        time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error setting up recorder for {label}: {e}")
            if self._running:
                logging.error(f"Recording thread for {label} crashed: {e}")

    def detect_speech(self, label, buffer_data):
        if buffer_data is None or len(buffer_data) == 0:
            logging.info(f"No audio data found in {label} buffer.")
            return [], np.array([], dtype=np.float32)

        speech_segments = get_speech_timestamps(
            buffer_data, self.model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=70,
            min_silence_duration_ms=100,
            min_speech_duration_ms=200,
            threshold=0.3
        )
        buffer_seconds = len(buffer_data) / SAMPLE_RATE
        logging.info(f"Detected {len(speech_segments)} speech segments in {label} of {buffer_seconds:.1f} seconds.")
        if self.debug:
            debug_filename = f"test_{label}.ogg"
            debug_data = self.create_ogg_bytes([{"data": buffer_data}])
            with open(debug_filename, "wb") as f:
                f.write(debug_data)
            logging.debug(f"Saved debug file: {debug_filename}")

        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE
        unprocessed_data = np.array([], dtype=np.float32)

        for i, seg in enumerate(speech_segments):
            if i == len(speech_segments) - 1 and total_duration - seg['end'] < 1:
                start_idx = int(seg['start'] * SAMPLE_RATE)
                unprocessed_data = buffer_data[start_idx:]
                logging.debug(f"Unprocessed segment at end of {label} buffer of length {len(unprocessed_data)/SAMPLE_RATE:.1f} seconds.")
                break
            start_idx = int(seg['start'] * SAMPLE_RATE)
            end_idx = int(seg['end'] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]
            segments.append({
                "offset": seg['start'],
                "data": seg_data
            })
        return segments, unprocessed_data

    def transcribe(self, ogg_bytes):
        size_mb = len(ogg_bytes) / (1024 * 1024)
        logging.info(f"Transcribing chunk: {size_mb:.2f}MB")
        
        with open("gemini_mic.txt", "r") as f:
            prompt_text = f.read().strip()
            
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",#"gemini-2.0-flash",
                contents=[
                    "Process the provided audio now and output your professional accurate transcription in the specified JSON format.",
                    types.Part.from_bytes(data=ogg_bytes, mime_type="audio/ogg")
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    system_instruction=prompt_text
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    def calculate_rms(self, audio_buffer):
        if len(audio_buffer) == 0:
            return 0
        return np.sqrt(np.mean(np.square(audio_buffer)))
    
    def normalize_audio(self, audio, target_rms):
        if len(audio) == 0 or target_rms == 0:
            return audio
            
        current_rms = self.calculate_rms(audio)
        if current_rms == 0:
            return audio
            
        gain_factor = target_rms / current_rms
        return audio * gain_factor

    def is_system_muted(self, audio_buffer=None):
        # Check if system is muted via pulseaudio
        try:
            result = subprocess.run(
                ["pactl", "get-sink-mute", "@DEFAULT_SINK@"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            pactl_muted = "Mute: yes" in result.stdout
        except subprocess.SubprocessError as e:
            logging.error(f"Error checking system mute status: {e}")
            pactl_muted = False
        
        # Check if audio buffer is silent
        silent = False
        if audio_buffer is not None and len(audio_buffer) > 0:
            rms = self.calculate_rms(audio_buffer)
            silent = rms < 0.0001
            if silent:
                logging.info(f"System audio silent (RMS: {rms:.6f})")
        
        return pactl_muted or silent

    def get_buffer(self, queue):
        buffer = np.array([], dtype=np.float32)
        while not queue.empty():
            chunk = queue.get()
            if chunk.ndim > 1:
                chunk_mono = np.mean(chunk, axis=1).flatten()
            else:
                chunk_mono = chunk.flatten()
            buffer = np.concatenate((buffer, chunk_mono))
        return buffer

    def apply_echo_cancellation(self, mic_buffer, sys_buffer):
        min_length = min(len(mic_buffer), len(sys_buffer))
        if min_length == 0:
            logging.info("Missing audio data in one or both buffers.")
            return mic_buffer

        logging.info(f"Echo cancelling mic seconds {len(mic_buffer)/SAMPLE_RATE:.4f} sys seconds {len(sys_buffer)/SAMPLE_RATE:.4f}")        

        # Convert float32 arrays to int16 (required by PyAEC)
        mic_int16 = (mic_buffer * 32767).astype(np.int16)
        sys_int16 = (sys_buffer * 32767).astype(np.int16)
        min_length = min(len(mic_int16), len(sys_int16))

        # Process in chunks of frame_size
        processed_chunks = []

        # Process complete frames
        for i in range(0, min_length, self.frame_size):
            # Get the frame (could be partial at the end)
            mic_frame = mic_int16[i:i+self.frame_size]
            sys_frame = sys_int16[i:i+self.frame_size]
            
            # If we have a partial frame at the end
            if min(len(mic_frame), len(sys_frame)) < self.frame_size:
                # Just append the raw partial frame
                processed_chunks.append(mic_frame)
            else:
                # Process with echo canceller
                processed_frame = self.aec.cancel_echo(mic_frame, sys_frame)
                processed_chunks.append(processed_frame)

        processed_mic = np.concatenate(processed_chunks).astype(np.float32) / 32767.0
        
        # Only enhance/normalize if we're doing echo cancellation
        processed_mic = self.enhance_voice(processed_mic)
        sys_rms = self.calculate_rms(sys_buffer)
        mic_rms = self.calculate_rms(processed_mic)
        logging.info(f"System RMS: {sys_rms:.6f} Microphone RMS: {mic_rms:.6f}")
        # if sys_rms > mic_rms and mic_rms > 0.01:
        #     processed_mic = self.normalize_audio(processed_mic, sys_rms)
        
        return processed_mic

    def create_ogg_bytes(self, segments: list) -> bytes:
        # Concatenate audio data from segments
        combined = np.concatenate([seg["data"] for seg in segments])
        # Convert to int16 format
        chunk_int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
        # Write to an in-memory OGG buffer
        buf = io.BytesIO()
        audio_data = chunk_int16.reshape(-1, CHANNELS)
        sf.write(buf, audio_data, SAMPLE_RATE, format='OGG', subtype='VORBIS')
        return buf.getvalue()

    def process_segments_and_transcribe(self, segments, suffix=None):
        if not segments:
            return
        total_seconds = sum([len(seg["data"]) / SAMPLE_RATE for seg in segments])
        if total_seconds < 3:
            logging.info(f"Skipping processing of {total_seconds:.1f} seconds of audio.")
            return
        ogg_bytes = self.create_ogg_bytes(segments)
        transcription = self.transcribe(ogg_bytes)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            timestamp += suffix
        self.save_results(timestamp, ogg_bytes, transcription)

    def process_buffer(self, buffer, new_data, label):
        merged = np.concatenate((buffer, new_data)) if buffer.size > 0 else new_data
        segments, unprocessed = self.detect_speech(label, merged)
        return segments, unprocessed

    def speech_processing_timer(self):
        mic_stash = np.array([], dtype=np.float32)
        sys_stash = np.array([], dtype=np.float32)
        while self._running:
            time.sleep(self.timer_interval)

            sys_new = self.get_buffer(self.sys_queue)
            sys_segments, sys_stash = self.process_buffer(sys_stash, sys_new, "sys")
            mic_new = self.get_buffer(self.mic_queue)

            system_muted = self.is_system_muted(sys_new)
            logging.info(f"System audio mute status: {'Muted' if system_muted else 'Not muted'}")

            if system_muted:
                mic_segments, mic_stash = self.process_buffer(mic_stash, mic_new, "mic")
                self.process_segments_and_transcribe(mic_segments, suffix="_mic")
                self.process_segments_and_transcribe(sys_segments, suffix="_sys")
            else:
                mic_processed = self.apply_echo_cancellation(mic_new, sys_new)
                mic_segments, mic_stash = self.process_buffer(mic_stash, mic_processed, "mic")
                segments_all = mic_segments + sys_segments
                if segments_all:
                    segments_all.sort(key=lambda seg: seg["offset"])  # weave together in time
                    self.process_segments_and_transcribe(segments_all)
            logging.info(f"Found {len(mic_segments)} microphone and {len(sys_segments)} system segments.")

    def save_results(self, timestamp, ogg_bytes, response_text):
        ogg_filepath = os.path.join(self.save_dir, f"audio_{timestamp}.ogg")
        json_filepath = os.path.join(self.save_dir, f"audio_{timestamp}.json")
        
        with open(ogg_filepath, "wb") as f:
            f.write(ogg_bytes)
        with open(json_filepath, "w") as f:
            json.dump({"text": response_text}, f)
        logging.info(f"Saved to {ogg_filepath}: {response_text}")

    def start(self):
        threads = [
            threading.Thread(target=self.record_device, args=(self.mic_device, self.mic_queue, "mic"), daemon=True),
            threading.Thread(target=self.record_device, args=(self.sys_device, self.sys_queue, "sys"), daemon=True),
            threading.Thread(target=self.speech_processing_timer, daemon=True)
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
    parser = argparse.ArgumentParser(description="Record audio and transcribe using Gemini API.")
    parser.add_argument("save_dir", nargs="?", default=None, help="Directory to save audio and transcriptions.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (save audio buffers).")
    parser.add_argument("-t", "--timer_interval", type=int, default=60, help="Timer interval in seconds.")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    # 3. Validate environment variables (API_KEY)
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        sys.exit("Error: GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
    
    # Create save directory if needed
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Enable faulthandler to help diagnose crashes
    faulthandler.enable()
    
    # 4. Create the recorder
    recorder = AudioRecorder(
        save_dir=args.save_dir,
        debug=args.debug,
        timer_interval=args.timer_interval,
        api_key=API_KEY
    )
    
    # 5. Detect devices or exit
    if not recorder.detect():
        sys.exit("No suitable audio devices found.")
    
    # 6. Start recording
    recorder.start()

if __name__ == "__main__":
    main()
