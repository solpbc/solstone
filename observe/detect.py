# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import logging
import threading

import numpy as np
import soundcard as sc

logger = logging.getLogger(__name__)


def input_detect(duration=0.4, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 18000 * t)  # ultrasonic

    try:
        devices = sc.all_microphones(include_loopback=True)
    except Exception:
        logger.warning("Failed to enumerate audio devices")
        return None, None
    if not devices:
        logger.warning("No audio devices found")
        return None, None

    results = {}
    barrier = threading.Barrier(len(devices) + 1)

    def record_mic(mic, results):
        barrier.wait()
        try:
            audio = mic.record(
                samplerate=sample_rate, numframes=int(sample_rate * duration)
            )
            results[mic.name] = audio
        except Exception:
            results[mic.name] = None

    def play_tone():
        barrier.wait()
        try:
            sp = sc.default_speaker()
            sp.play(tone, samplerate=sample_rate)
        except Exception:
            logger.warning("No default speaker available for tone detection")

    threads = []
    for mic in devices:
        thread = threading.Thread(target=record_mic, args=(mic, results))
        thread.start()
        threads.append(thread)

    play_thread = threading.Thread(target=play_tone)
    play_thread.start()
    threads.append(play_thread)

    for thread in threads:
        thread.join()

    # Analyze the recordings with a simple amplitude threshold
    threshold = 0.001
    mic_detected = None
    loopback_detected = None
    for mic in devices:
        audio = results.get(mic.name)
        if audio is not None and np.max(np.abs(audio)) > threshold:
            # First match for each category
            if "microphone" in str(mic).lower() and mic_detected is None:
                mic_detected = mic
            if "loopback" in str(mic).lower() and loopback_detected is None:
                loopback_detected = mic
    return mic_detected, loopback_detected


if __name__ == "__main__":
    mic, loopback = input_detect()
    print("Microphone detection:", mic.name if mic else "None")
    print("Loopback detection:", loopback.name if loopback else "None")
