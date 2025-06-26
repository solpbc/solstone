import numpy as np


def merge_best(
    sys_data: np.ndarray,
    mic_data: np.ndarray,
    sample_rate: int,
    window_ms: int = 50,
    threshold: float = 0.005,
) -> np.ndarray:
    """Mix system and microphone audio, muting mic when both exceed threshold to avoid feedback."""

    length = min(len(sys_data), len(mic_data))
    if length == 0:
        return np.array([], dtype=np.float32)

    sys_data = sys_data[:length]
    mic_data = mic_data[:length]

    window_samples = max(1, int(sample_rate * window_ms / 1000))

    output = np.zeros(length, dtype=np.float32)

    for start in range(0, length, window_samples):
        end = min(length, start + window_samples)
        sys_win = sys_data[start:end]
        mic_win = mic_data[start:end]
        sys_rms = float(np.sqrt(np.mean(sys_win**2))) if len(sys_win) else 0.0
        mic_rms = float(np.sqrt(np.mean(mic_win**2))) if len(mic_win) else 0.0

        if sys_rms > threshold and mic_rms > threshold:
            # Both channels active - mute mic to avoid interference
            output[start:end] = sys_win
            mic_muted = True
        else:
            # Mix both channels together
            output[start:end] = sys_win + mic_win
            mic_muted = False

        print(f"sys_rms: {sys_rms:.4f}, mic_rms: {mic_rms:.4f}, "
              f"mic_muted: {mic_muted}")

    return output
