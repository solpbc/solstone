import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def input_graph(flac_file, chunk_duration=0.05):
    """
    Load a FLAC file and graph RMS values for each chunk.
    Left channel is treated as microphone, right channel as system audio.

    Args:
        flac_file: Path to the FLAC file
        chunk_duration: Duration of each RMS chunk in seconds (default 50ms)
    """
    try:
        # Load the FLAC file
        audio_data, sample_rate = sf.read(flac_file)
        print(f"Loaded {flac_file}: {audio_data.shape[0]} samples at {sample_rate} Hz")

        # Check if stereo
        if len(audio_data.shape) != 2 or audio_data.shape[1] != 2:
            print("Error: FLAC file must be stereo (2 channels)")
            return

        # Extract left (mic) and right (system) channels
        mic_sig = audio_data[:, 0]  # Left channel
        system_sig = audio_data[:, 1]  # Right channel

        duration = len(mic_sig) / sample_rate

    except Exception as e:
        print(f"Error loading FLAC file: {e}")
        return

    # Calculate chunk parameters
    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = len(mic_sig) // chunk_samples

    # Calculate RMS for each chunk
    mic_rms = []
    system_rms = []
    time_points = []

    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples

        # Calculate RMS for this chunk
        mic_chunk = mic_sig[start_idx:end_idx]
        system_chunk = system_sig[start_idx:end_idx]

        mic_rms.append(np.sqrt(np.mean(mic_chunk**2)))
        system_rms.append(np.sqrt(np.mean(system_chunk**2)))
        time_points.append(i * chunk_duration)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_points, mic_rms, label="Microphone RMS (Left)", alpha=0.8, linewidth=2
    )
    plt.plot(
        time_points,
        system_rms,
        label="System Audio RMS (Right)",
        alpha=0.8,
        linewidth=2,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Amplitude")
    plt.title(f"Audio RMS over {duration:.1f}s (50ms chunks)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show statistics
    print(
        f"\nMicrophone RMS (Left) - Mean: {np.mean(mic_rms):.4f}, Max: {np.max(mic_rms):.4f}"
    )
    print(
        f"System Audio RMS (Right) - Mean: {np.mean(system_rms):.4f}, Max: {np.max(system_rms):.4f}"
    )

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python input_graph.py <flac_file>")
        exit(1)

    flac_file = sys.argv[1]
    print(f"Processing FLAC file: {flac_file}")
    input_graph(flac_file)
