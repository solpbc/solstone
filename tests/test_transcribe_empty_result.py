# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for empty-result handling in process_audio."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from observe.utils import SAMPLE_RATE
from observe.vad import VadResult


@pytest.fixture
def raw_path(tmp_path):
    path = tmp_path / "chronicle" / "20260416" / "default" / "120000_300" / "audio.m4a"
    path.parent.mkdir(parents=True)
    path.touch()
    return path


@pytest.fixture
def audio_buffer():
    return np.zeros(10 * SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def vad_result():
    return VadResult(
        duration=10.0,
        speech_duration=5.0,
        has_speech=True,
        speech_segments=[(1.0, 6.0)],
    )


def test_empty_statements_filter_path(raw_path, audio_buffer, vad_result):
    from observe.transcribe.main import process_audio

    backend_module = MagicMock()
    backend_module.get_model_info.return_value = {
        "model": "medium.en",
        "device": "cpu",
        "compute_type": "int8",
    }

    with (
        patch(
            "observe.transcribe.main.get_config",
            return_value={"transcribe": {"preserve_all": False}},
        ),
        patch(
            "observe.transcribe.main.get_journal", return_value=str(raw_path.parents[4])
        ),
        patch("observe.transcribe.main.stt_transcribe", return_value=[]),
        patch("observe.transcribe.main.get_backend", return_value=backend_module),
        patch("observe.transcribe.main.callosum_send") as mock_send,
    ):
        process_audio(raw_path, audio_buffer, vad_result, {}, backend="whisper")

    assert not raw_path.exists()
    assert mock_send.call_args.args[:2] == ("observe", "transcribed")
    assert mock_send.call_args.kwargs["outcome"] == "filtered"


def test_empty_statements_preserve_path(raw_path, audio_buffer, vad_result):
    from observe.transcribe.main import process_audio

    backend_module = MagicMock()
    backend_module.get_model_info.return_value = {
        "model": "medium.en",
        "device": "cpu",
        "compute_type": "int8",
    }

    with (
        patch(
            "observe.transcribe.main.get_config",
            return_value={"transcribe": {"preserve_all": True}},
        ),
        patch(
            "observe.transcribe.main.get_journal", return_value=str(raw_path.parents[4])
        ),
        patch("observe.transcribe.main.stt_transcribe", return_value=[]),
        patch("observe.transcribe.main.get_backend", return_value=backend_module),
        patch("observe.transcribe.main.callosum_send") as mock_send,
    ):
        process_audio(raw_path, audio_buffer, vad_result, {}, backend="whisper")

    assert raw_path.exists()
    assert mock_send.call_args.args[:2] == ("observe", "transcribed")
    assert mock_send.call_args.kwargs["outcome"] == "preserved"


def test_backend_raise_propagates(raw_path, audio_buffer, vad_result):
    from observe.transcribe.main import process_audio

    with patch(
        "observe.transcribe.main.stt_transcribe", side_effect=RuntimeError("rev.ai 502")
    ):
        with pytest.raises(SystemExit) as exc_info:
            process_audio(raw_path, audio_buffer, vad_result, {}, backend="whisper")

    assert exc_info.value.code == 1
    assert raw_path.exists()
