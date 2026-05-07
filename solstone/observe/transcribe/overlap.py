# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Overlap-fraction inference for speaker-attribution flywheel gating."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

from solstone.observe.utils import SAMPLE_RATE

logger = logging.getLogger(__name__)

WINDOW_S = 10
STRIDE_S = 5
FRAMES_PER_WINDOW = 589
OVERLAP_CLASSES = (4, 5, 6)

_overlap_session: ort.InferenceSession | None = None


def _get_overlap_session() -> ort.InferenceSession:
    """Return a cached ONNX InferenceSession for the pyannote overlap model."""
    global _overlap_session

    if _overlap_session is None:
        from solstone.observe.transcribe.main import (
            PYANNOTE_OVERLAP_MODEL_PATH,
            _select_onnx_providers,
        )

        if not PYANNOTE_OVERLAP_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"pyannote model not found at {PYANNOTE_OVERLAP_MODEL_PATH}. "
                "Run `make install` to verify the bundled asset."
            )

        providers = _select_onnx_providers()
        start = time.monotonic()
        _overlap_session = ort.InferenceSession(
            str(PYANNOTE_OVERLAP_MODEL_PATH),
            providers=providers,
        )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "pyannote overlap session loaded (providers=%s, elapsed=%.1fms)",
            _overlap_session.get_providers(),
            elapsed_ms,
        )

    return _overlap_session


def compute_overlap_fraction(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> float:
    """Compute the speech-conditioned overlap fraction for an audio segment."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"pyannote overlap detector requires {SAMPLE_RATE} Hz audio, got {sample_rate}"
        )

    session = _get_overlap_session()
    input_name = session.get_inputs()[0].name

    window_samples = WINDOW_S * sample_rate
    stride_samples = STRIDE_S * sample_rate

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < window_samples:
        pad = window_samples - len(audio_f32)
        audio_padded = np.concatenate([audio_f32, np.zeros(pad, dtype=np.float32)])
    else:
        audio_padded = audio_f32

    starts = list(range(0, len(audio_padded) - window_samples + 1, stride_samples))
    final_start = max(0, len(audio_padded) - window_samples)
    if not starts:
        starts = [final_start]
    elif starts[-1] != final_start:
        starts.append(final_start)

    samples_per_frame = window_samples / FRAMES_PER_WINDOW
    num_frames = int(np.ceil(len(audio_padded) / samples_per_frame))
    accum = np.zeros((num_frames, 7), dtype=np.float64)
    counts = np.zeros((num_frames,), dtype=np.int32)

    for start_sample in starts:
        chunk = audio_padded[start_sample : start_sample + window_samples][
            None, None, :
        ]
        log_probs = session.run(None, {input_name: chunk})[0][0]
        frame_start = int(round(start_sample / samples_per_frame))
        frame_end = frame_start + log_probs.shape[0]
        if frame_end > num_frames:
            frame_end = num_frames
            log_probs = log_probs[: frame_end - frame_start]
        accum[frame_start:frame_end] += log_probs.astype(np.float64)
        counts[frame_start:frame_end] += 1

    counts = np.maximum(counts, 1)
    avg_log_probs = (accum / counts[:, None]).astype(np.float32)
    argmax = avg_log_probs.argmax(axis=-1)
    speech_count = int((argmax >= 1).sum())
    if speech_count == 0:
        return 0.0

    overlap_count = int(np.isin(argmax, OVERLAP_CLASSES).sum())
    return float(overlap_count / speech_count)


def compute_overlap_fraction_for_wav(path: Path) -> float:
    """Load a WAV file and compute its speech-conditioned overlap fraction."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)
        sample_rate = SAMPLE_RATE
    return compute_overlap_fraction(np.asarray(audio, dtype=np.float32), sample_rate)
