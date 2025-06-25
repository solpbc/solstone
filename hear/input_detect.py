import threading

import numpy as np
import soundcard as sc


def input_detect(duration=0.2, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 18000 * t)  # ultrasonic

    devices = sc.all_microphones(include_loopback=True)
    if not devices:
        print("No matching devices found:")
        for mic in devices:
            print(mic)
        return None, None

    results = {}
    barrier = threading.Barrier(len(devices) + 1)

    def record_mic(mic, results):
        barrier.wait()
        try:
            audio = mic.record(samplerate=sample_rate, numframes=int(sample_rate * duration))
            results[mic.name] = audio
        except Exception as e:
            results[mic.name] = None

    def play_tone():
        barrier.wait()
        sp = sc.default_speaker()
        sp.play(tone, samplerate=sample_rate)

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
    threshold = 0.01
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
