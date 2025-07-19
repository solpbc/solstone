import threading

import numpy as np
import soundcard as sc
from scipy.signal import correlate

from hear.input_detect import input_detect


def input_delay(mic_device, system_device, duration=0.2, sample_rate=44100):
    # Generate test signal: 10ms gated 18kHz burst for clear cross-correlation peak
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    gate = np.zeros_like(t)
    burst_frames = int(0.010 * sample_rate)
    gate[:burst_frames] = 1.0
    tone = 0.5 * np.sin(2 * np.pi * 18000 * t) * gate

    # Storage for recording results
    results = {}
    barrier = threading.Barrier(3)  # mic, system, and playback threads

    def record_device(device, device_key):
        barrier.wait()
        try:
            audio = device.record(
                samplerate=sample_rate, numframes=int(sample_rate * duration)
            )
            results[device_key] = audio
        except Exception as e:
            print(f"Recording error on {device.name}: {e}")
            results[device_key] = None

    def play_tone():
        barrier.wait()
        try:
            speaker = sc.default_speaker()
            speaker.play(tone, samplerate=sample_rate)
        except Exception as e:
            print(f"Playback error: {e}")

    # Start recording and playback threads
    mic_thread = threading.Thread(target=record_device, args=(mic_device, "mic"))
    system_thread = threading.Thread(
        target=record_device, args=(system_device, "system")
    )
    play_thread = threading.Thread(target=play_tone)

    threads = [mic_thread, system_thread, play_thread]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Calculate delay using cross-correlation
    mic_audio = results.get("mic")
    system_audio = results.get("system")

    if mic_audio is None or system_audio is None:
        print("Failed to record from one or both devices")
        return None

    # Flatten signals for cross-correlation
    mic_sig = mic_audio.flatten()
    system_sig = system_audio.flatten()

    # Cross-correlate to find delay: peak index gives sample lag
    correlation = correlate(mic_sig, system_sig, mode="full", method="fft")
    lag_samples = correlation.argmax() - (len(system_sig) - 1)
    delay_ms = lag_samples * 1000 / sample_rate

    return delay_ms


if __name__ == "__main__":
    # Use input_detect to find the best microphone and system devices
    print("Detecting input devices...")
    mic_device, system_device = input_detect()

    if not mic_device or not system_device:
        print("Missing device(s)")
        exit(1)

    # Run multiple measurements
    for rep in range(30):
        print(f"Run {rep + 1} of 30")
        delay = input_delay(mic_device, system_device)
        if delay is not None:
            print(f"System → Mic delay: {delay:.2f} ms")
        else:
            print("Failed to measure delay")
