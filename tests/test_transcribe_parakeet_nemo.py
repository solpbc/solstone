# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import platform
from pathlib import Path

import pytest
import soundfile as sf

if platform.system() != "Linux":
    pytest.skip("Linux-only NeMo test", allow_module_level=True)

try:
    import google.protobuf.json_format  # noqa: F401
    import sklearn.metrics.pairwise  # noqa: F401  # real sklearn preempts conftest stub
except ImportError:
    pass

import observe.transcribe._parakeet_nemo as parakeet_nemo


def _require_nemo() -> None:
    try:
        import nemo  # noqa: F401
    except ImportError:
        pytest.skip("NeMo not installed")


@pytest.fixture
def pangram_audio() -> tuple[object, int]:
    fixture_path = Path("tests/fixtures/parakeet_sample.wav")
    return sf.read(fixture_path, dtype="float32")


@pytest.mark.timeout(120)
def test_transcribe_nemo_pangram(pangram_audio: tuple[object, int]):
    _require_nemo()
    audio, sample_rate = pangram_audio

    statements = parakeet_nemo.transcribe(audio, sample_rate, {"device": "cpu"})

    assert statements
    combined_text = " ".join(statement["text"] for statement in statements).lower()
    assert "quick" in combined_text
    assert "fox" in combined_text
    for statement in statements:
        assert all(
            word["word"].startswith(" ") and not word["word"].startswith("  ")
            for word in statement["words"]
        )


def test_get_model_info_shape():
    _require_nemo()

    info = parakeet_nemo.get_model_info({"device": "cpu"})

    assert {"model", "device", "compute_type", "per_word_confidence"} <= set(info)
    assert info["per_word_confidence"] is False


def test_invalid_device_raises(pangram_audio: tuple[object, int]):
    _require_nemo()
    audio, sample_rate = pangram_audio

    with pytest.raises(ValueError, match="auto, cpu, cuda"):
        parakeet_nemo.transcribe(audio, sample_rate, {"device": "tpu"})


def test_unsupported_model_version_raises(pangram_audio: tuple[object, int]):
    _require_nemo()
    audio, sample_rate = pangram_audio

    with pytest.raises(ValueError, match="v2, v3"):
        parakeet_nemo.transcribe(audio, sample_rate, {"model_version": "v9"})
