"""Audio processing utilities for gemini transcription."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from silero_vad import get_speech_timestamps

SAMPLE_RATE = 16000


def merge_streams(
    sys_data: np.ndarray,
    mic_data: np.ndarray,
    sample_rate: int,
    window_ms: int = 50,
    threshold: float = 0.005,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mix system and microphone audio while avoiding feedback."""

    length = min(len(sys_data), len(mic_data))
    if length == 0:
        return np.array([], dtype=np.float32), []

    sys_data = sys_data[:length]
    mic_data = mic_data[:length]

    sys_rms_full = float(np.sqrt(np.mean(sys_data**2))) if len(sys_data) else 0.0
    if np.isclose(sys_rms_full, 0.0):
        mic_range = (0.0, length / sample_rate)
        return mic_data, [mic_range]
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    output = np.zeros(length, dtype=np.float32)
    mic_ranges: list[tuple[float, float]] = []
    in_range = False
    range_start = 0
    consecutive_mic_windows = 0

    for start in range(0, length, window_samples):
        end = min(length, start + window_samples)
        sys_win = sys_data[start:end]
        mic_win = mic_data[start:end]
        sys_rms = float(np.sqrt(np.mean(sys_win**2))) if len(sys_win) else 0.0
        mic_rms = float(np.sqrt(np.mean(mic_win**2))) if len(mic_win) else 0.0

        if sys_rms > threshold and mic_rms > threshold:
            output[start:end] = sys_win
            consecutive_mic_windows = 0
            if in_range:
                in_range = False
                mic_ranges.append((range_start / sample_rate, start / sample_rate))
        else:
            output[start:end] = sys_win + mic_win
            if sys_rms < threshold and mic_rms > threshold:
                consecutive_mic_windows += 1
                if consecutive_mic_windows >= 2 and not in_range:
                    in_range = True
                    range_start = start - window_samples
            else:
                consecutive_mic_windows = 0

    if in_range:
        mic_ranges.append((range_start / sample_rate, length / sample_rate))

    mic_ranges = [(s, e) for s, e in mic_ranges if e - s >= 1.0]

    return output, mic_ranges


def resample_audio(
    audio_data: np.ndarray,
    in_sr: int,
    out_sr: int,
    df_model,
    df_state,
    df_sr: int,
    denoise: bool = False,
) -> np.ndarray:
    """Resample audio and optionally denoise using DeepFilterNet."""

    import torch
    from df.enhance import enhance
    from df.io import resample as df_resample

    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

    if denoise:
        if in_sr != df_sr:
            audio_tensor = df_resample(audio_tensor, in_sr, df_sr, method="kaiser_best")
            in_sr = df_sr
        with torch.no_grad():
            audio_tensor = enhance(df_model, df_state, audio_tensor)[0]

    if in_sr != out_sr:
        audio_tensor = df_resample(
            audio_tensor.unsqueeze(0), in_sr, out_sr, method="kaiser_best"
        )

    return audio_tensor.squeeze().cpu().numpy()


def calculate_mic_overlap(
    seg_start: float, seg_end: float, mic_ranges: List[Tuple[float, float]]
) -> float:
    """Return percentage of a segment overlapping with mic ranges."""

    if not mic_ranges:
        return 0.0

    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return 0.0

    overlap_duration = 0.0
    for mic_start, mic_end in mic_ranges:
        overlap_start = max(seg_start, mic_start)
        overlap_end = min(seg_end, mic_end)
        if overlap_start < overlap_end:
            overlap_duration += overlap_end - overlap_start

    return overlap_duration / seg_duration


def detect_speech(
    vad_model,
    label: str,
    buffer_data: np.ndarray,
    mic_ranges: Optional[List[Tuple[float, float]]] = None,
    no_stash: bool = False,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Detect speech segments using silero VAD."""

    if buffer_data is None or len(buffer_data) == 0:
        logging.info(f"No audio data found in {label} buffer.")
        return [], np.array([], dtype=np.float32)
    try:
        vad_model.reset_states()
        speech_segments = get_speech_timestamps(
            buffer_data,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=70,
            min_silence_duration_ms=100,
            min_speech_duration_ms=200,
            threshold=0.2,
        )
        buffer_seconds = len(buffer_data) / SAMPLE_RATE
        segments = []
        total_duration = len(buffer_data) / SAMPLE_RATE
        unprocessed_data = np.array([], dtype=np.float32)
        for i, seg in enumerate(speech_segments):
            # Skip stash logic if no_stash is True
            if (
                not no_stash
                and i == len(speech_segments) - 1
                and total_duration - seg["end"] < 1
            ):
                start_idx = int(seg["start"] * SAMPLE_RATE)
                unprocessed_data = buffer_data[start_idx:]
                break
            start_idx = int(seg["start"] * SAMPLE_RATE)
            end_idx = int(seg["end"] * SAMPLE_RATE)
            seg_data = buffer_data[start_idx:end_idx]

            mic_overlap = calculate_mic_overlap(
                seg["start"], seg["end"], mic_ranges or []
            )
            in_mic = mic_overlap > 0.8

            segments.append({"offset": seg["start"], "data": seg_data, "mic": in_mic})
        logging.info(
            "Detected %d speech segments in %s of %.1f seconds with %.1f seconds remainder",
            len(speech_segments),
            label,
            buffer_seconds,
            len(unprocessed_data) / SAMPLE_RATE,
        )
        return segments, unprocessed_data
    except Exception as e:
        logging.error(f"Error in detect_speech for {label}: {e}")
        raise
