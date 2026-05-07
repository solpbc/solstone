# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for pyannote overlap-fraction inference."""

from __future__ import annotations

import numpy as np
import pytest

from solstone.observe.utils import SAMPLE_RATE


class _Input:
    def __init__(self, name: str):
        self.name = name


class _StubSession:
    def __init__(self, log_probs: np.ndarray):
        self._log_probs = log_probs.astype(np.float32)

    def get_inputs(self):
        return [_Input("input_values")]

    def run(self, _outputs, _inputs):
        return [self._log_probs[None, :, :]]


def _dominant_log_probs(classes: np.ndarray) -> np.ndarray:
    log_probs = np.full((classes.shape[0], 7), -10.0, dtype=np.float32)
    log_probs[np.arange(classes.shape[0]), classes] = 0.0
    return log_probs


def test_compute_overlap_fraction_silent_audio_returns_zero(monkeypatch):
    from solstone.observe.transcribe import overlap

    monkeypatch.setattr(
        overlap,
        "_get_overlap_session",
        lambda: _StubSession(_dominant_log_probs(np.zeros(589, dtype=np.int64))),
    )

    result = overlap.compute_overlap_fraction(
        np.zeros(12 * SAMPLE_RATE, dtype=np.float32)
    )

    assert result == 0.0


def test_compute_overlap_fraction_short_audio_padded(monkeypatch):
    from solstone.observe.transcribe import overlap

    monkeypatch.setattr(
        overlap,
        "_get_overlap_session",
        lambda: _StubSession(_dominant_log_probs(np.zeros(589, dtype=np.int64))),
    )

    result = overlap.compute_overlap_fraction(
        np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    )

    assert isinstance(result, float)
    assert result == 0.0


def test_compute_overlap_fraction_non_aligned_length(monkeypatch):
    from solstone.observe.transcribe import overlap

    monkeypatch.setattr(
        overlap,
        "_get_overlap_session",
        lambda: _StubSession(_dominant_log_probs(np.zeros(589, dtype=np.int64))),
    )

    audio = np.zeros(int(13.7 * SAMPLE_RATE), dtype=np.float32)
    result = overlap.compute_overlap_fraction(audio)

    assert result == 0.0


def test_compute_overlap_fraction_rejects_wrong_sample_rate():
    from solstone.observe.transcribe.overlap import compute_overlap_fraction

    with pytest.raises(ValueError, match="requires 16000 Hz audio"):
        compute_overlap_fraction(np.zeros(16000, dtype=np.float32), sample_rate=8000)


def test_get_overlap_session_loads_and_caches():
    from solstone.observe.transcribe.overlap import _get_overlap_session

    first = _get_overlap_session()
    second = _get_overlap_session()

    assert first is second


def test_compute_overlap_fraction_uses_conditioned_formula(monkeypatch):
    from solstone.observe.transcribe import overlap

    classes = np.concatenate(
        [
            np.full(300, 1, dtype=np.int64),
            np.full(100, 4, dtype=np.int64),
            np.zeros(189, dtype=np.int64),
        ]
    )
    monkeypatch.setattr(
        overlap,
        "_get_overlap_session",
        lambda: _StubSession(_dominant_log_probs(classes)),
    )

    result = overlap.compute_overlap_fraction(
        np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    )

    assert result == pytest.approx(100 / 400)
