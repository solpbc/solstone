# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import solstone.observe.transcribe._parakeet_coreml as parakeet
from solstone.observe.transcribe import BACKEND_METADATA, BACKEND_REGISTRY
from solstone.think.install_models import _fixture_audio_path


def _skip_reason() -> str | None:
    if platform.system() != "Darwin":
        return "requires Darwin"
    if platform.machine() != "arm64":
        return "requires arm64"
    try:
        parakeet._resolve_helper_path()
    except RuntimeError as exc:
        return str(exc)
    return None


def test_collapse_empty_input():
    assert parakeet._collapse_subwords_to_words([]) == []


def test_collapse_single_word():
    tokens = [
        {
            "token": "▁hello",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        }
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert words == [{"word": " hello", "start": 0.0, "end": 0.1, "probability": 0.9}]


def test_collapse_two_words_with_boundary():
    tokens = [
        {
            "token": "▁the",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        },
        {
            "token": " quick",
            "token_id": 2,
            "start": 0.1,
            "end": 0.2,
            "confidence": 0.8,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert [word["word"] for word in words] == [" the", " quick"]


def test_collapse_subword_rebuild():
    tokens = [
        {
            "token": "▁the",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        },
        {
            "token": "▁qu",
            "token_id": 2,
            "start": 0.1,
            "end": 0.18,
            "confidence": 0.8,
        },
        {
            "token": "ick",
            "token_id": 3,
            "start": 0.18,
            "end": 0.25,
            "confidence": 0.95,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert len(words) == 2
    assert words[0]["word"] == " the"
    assert words[1]["word"] == " quick"
    assert words[1]["probability"] == pytest.approx(0.8)


def test_collapse_contraction():
    tokens = [
        {
            "token": "▁don",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        },
        {
            "token": "'",
            "token_id": 2,
            "start": 0.1,
            "end": 0.12,
            "confidence": 0.8,
        },
        {
            "token": "t",
            "token_id": 3,
            "start": 0.12,
            "end": 0.18,
            "confidence": 0.85,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert [word["word"] for word in words] == [" don't"]


def test_collapse_trailing_punctuation_attaches():
    tokens = [
        {
            "token": "▁fox",
            "token_id": 1,
            "start": 0.0,
            "end": 0.2,
            "confidence": 0.9,
        },
        {
            "token": ".",
            "token_id": 2,
            "start": 0.2,
            "end": 0.24,
            "confidence": 0.7,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert words == [{"word": " fox.", "start": 0.0, "end": 0.2, "probability": 0.7}]


def test_collapse_confidence_is_min():
    tokens = [
        {
            "token": "▁hel",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        },
        {
            "token": "lo",
            "token_id": 2,
            "start": 0.1,
            "end": 0.2,
            "confidence": 0.5,
        },
        {
            "token": "!",
            "token_id": 3,
            "start": 0.2,
            "end": 0.24,
            "confidence": 0.7,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert words[0]["probability"] == pytest.approx(0.5)


def test_collapse_leading_space_invariant():
    tokens = [
        {
            "token": "▁the",
            "token_id": 1,
            "start": 0.0,
            "end": 0.1,
            "confidence": 0.9,
        },
        {
            "token": " quick",
            "token_id": 2,
            "start": 0.1,
            "end": 0.2,
            "confidence": 0.8,
        },
        {
            "token": " brown",
            "token_id": 3,
            "start": 0.2,
            "end": 0.3,
            "confidence": 0.85,
        },
        {
            "token": ".",
            "token_id": 4,
            "start": 0.3,
            "end": 0.34,
            "confidence": 0.95,
        },
    ]
    words = parakeet._collapse_subwords_to_words(tokens)
    assert all(
        word["word"].startswith(" ") and not word["word"].startswith("  ")
        for word in words
    )


def test_validate_config_bad_model():
    with pytest.raises(ValueError, match="v2, v3"):
        parakeet._validate_config({"model_version": "v4"})


def test_validate_config_bad_timeout():
    with pytest.raises(ValueError, match="> 0"):
        parakeet._validate_config({"timeout_sec": -1})


def test_registry_has_parakeet():
    assert "parakeet" in BACKEND_REGISTRY
    assert "parakeet" in BACKEND_METADATA


def test_metadata_settings_list_of_str():
    assert all(isinstance(key, str) for key in BACKEND_METADATA["parakeet"]["settings"])


def test_resolve_helper_path_packaged_missing_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("SOLSTONE_PARAKEET_HELPER", raising=False)
    monkeypatch.setattr(parakeet, "__file__", str(tmp_path / "_parakeet_coreml.py"))
    monkeypatch.setattr(parakeet, "is_packaged_install", lambda: True)

    with pytest.raises(RuntimeError) as exc_info:
        parakeet._resolve_helper_path()

    message = str(exc_info.value)
    assert "Apple Silicon Macs running macOS 14" in message
    assert "Intel Mac" in message
    assert "source checkout" in message


def test_resolve_helper_path_env_override_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fake = tmp_path / "custom" / "parakeet-helper"
    monkeypatch.setenv("SOLSTONE_PARAKEET_HELPER", str(fake))
    monkeypatch.setattr(parakeet, "__file__", str(tmp_path / "_parakeet_coreml.py"))
    assert parakeet._resolve_helper_path() == fake.expanduser().resolve()


def test_resolve_helper_path_prefers_bundled_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("SOLSTONE_PARAKEET_HELPER", raising=False)
    monkeypatch.setattr(parakeet, "__file__", str(tmp_path / "_parakeet_coreml.py"))
    bundled = tmp_path / "parakeet_helper" / "_bin" / "parakeet-helper"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("")
    assert parakeet._resolve_helper_path() == bundled


def test_resolve_helper_path_falls_back_to_swift_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("SOLSTONE_PARAKEET_HELPER", raising=False)
    monkeypatch.setattr(parakeet, "__file__", str(tmp_path / "_parakeet_coreml.py"))
    monkeypatch.setattr(parakeet, "is_packaged_install", lambda: False)
    expected = tmp_path / "parakeet_helper" / ".build" / "release" / "parakeet-helper"
    assert parakeet._resolve_helper_path() == expected


def test_transcribe_rejects_transcript_without_token_timings(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        parakeet, "_resolve_helper_path", lambda: Path("/tmp/parakeet-helper")
    )
    monkeypatch.setattr(
        parakeet,
        "get_model_info",
        lambda config: {"model": "parakeet-tdt-0.6b-v3"},
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "transcript": "hello world",
                    "token_timings": [],
                    "audio_sec": 1.0,
                    "transcribe_ms": 50,
                    "rtfx": 20.0,
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="transcript text without token timings",
    ):
        parakeet.transcribe(np.zeros(16000, dtype=np.float32), 16000, {})


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.timeout(120)
def test_helper_version_envelope():
    helper_path = parakeet._resolve_helper_path()
    result = subprocess.run(
        [str(helper_path), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["fluidaudio_version"] == "0.14.0"
    assert set(payload) >= {
        "fluidaudio_version",
        "model_version_default",
        "swift_version",
        "hardware",
        "macos_version",
    }


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.timeout(120)
def test_transcribe_pangram_end_to_end():
    audio, sample_rate = sf.read(_fixture_audio_path(), dtype="float32")
    statements = parakeet.transcribe(audio, sample_rate, {})
    assert statements

    combined_text = " ".join(statement["text"] for statement in statements)
    tokens = combined_text.split()
    assert "quick" in tokens
    assert "fox" in tokens

    for statement in statements:
        assert set(statement) >= {"id", "start", "end", "text", "words", "speaker"}
        assert statement["speaker"] is None
        assert statement["words"]
        assert all(
            word["word"].startswith(" ") and not word["word"].startswith("  ")
            for word in statement["words"]
        )


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.timeout(120)
def test_transcribe_empty_audio():
    audio = np.zeros(5 * 16000, dtype=np.float32)
    statements = parakeet.transcribe(audio, 16000, {})
    assert statements == []
