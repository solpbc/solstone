# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Parakeet STT backend via a Swift helper.

This backend shells out to `observe/transcribe/parakeet_helper/` for FluidAudio
inference, then rebuilds repo-standard statements with
`observe.transcribe.utils.build_statements_from_acoustic`.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import string
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from solstone.observe.transcribe.parakeet_hints import PACKAGED_COREML_HINT
from solstone.observe.transcribe.utils import build_statements_from_acoustic
from solstone.think.utils import is_packaged_install

_VERSION_CACHE: dict[str, dict] = {}
_DEFAULT_MODEL_VERSION = "v3"
_DEFAULT_TIMEOUT_SEC = 120.0
_DEFAULT_CACHE_DIR = (
    Path.home() / "Library/Application Support/solstone/parakeet/models"
)
_VALID_MODEL_VERSIONS = frozenset({"v2", "v3"})
_PUNCTUATION_TOKENS = frozenset(string.punctuation) | frozenset(
    {"—", "…", "’", "‘", "“", "”"}
)
_HELPER_ENV_KEY = "SOLSTONE_PARAKEET_HELPER"


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[dict]:
    """Transcribe audio by invoking the Parakeet helper and rebuilding statements."""
    model_version, cache_dir, timeout_sec = _validate_config(config)
    helper_path = _resolve_helper_path()
    model_info = get_model_info(config)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(temp_path, audio_int16, sample_rate, format="WAV", subtype="PCM_16")

        argv = [
            str(helper_path),
            "--cache-dir",
            str(cache_dir),
            "--model",
            model_version,
            str(temp_path),
        ]

        try:
            result = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Parakeet helper timed out after {timeout_sec:.1f}s. Rebuild with "
                f"'make parakeet-helper', increase transcribe.parakeet.timeout_sec, "
                f"or set ${_HELPER_ENV_KEY} to a different helper build."
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            try:
                stderr_json = json.loads(stderr) if stderr else {}
            except json.JSONDecodeError:
                stderr_json = {}
            message = stderr_json.get("message") or stderr or "unknown helper failure"

            if result.returncode == 2:
                raise RuntimeError(
                    f"Parakeet helper rejected validated arguments (internal bug — file an issue): {message}"
                )
            if result.returncode == 3:
                raise RuntimeError(
                    f"Parakeet helper could not prepare cache dir {cache_dir}: {message}"
                )
            if result.returncode == 4:
                raise RuntimeError(
                    f"Parakeet helper failed to download or load model '{model_version}'. "
                    f"Valid values: v2, v3. {message}"
                )
            if result.returncode == 5:
                raise RuntimeError(
                    f"Parakeet helper failed to transcribe audio: {message}"
                )
            raise RuntimeError(
                f"Parakeet helper failed with exit code {result.returncode}: {message}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Parakeet helper returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Parakeet helper returned a non-object JSON payload")

        helper_transcript = str(payload.get("transcript", "")).strip()
        token_timings = payload.get("token_timings", [])
        if not token_timings:
            if helper_transcript:
                raise RuntimeError(
                    "Parakeet helper returned transcript text without token timings "
                    "(internal bug — file an issue)."
                )
            return []

        words = _collapse_subwords_to_words(token_timings)
        if not words:
            if helper_transcript:
                raise RuntimeError(
                    "Parakeet helper returned token timings that collapsed to no words "
                    "(internal bug — file an issue)."
                )
            return []

        acoustic_segments = [
            {
                "id": 1,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": helper_transcript,
                "words": words,
            }
        ]
        statements = build_statements_from_acoustic(acoustic_segments)
        for statement in statements:
            statement["speaker"] = None

        rebuilt_text = " ".join(statement["text"] for statement in statements).strip()
        _log_drift_if_needed(helper_transcript, rebuilt_text)

        audio_sec = float(payload.get("audio_sec", len(audio) / sample_rate))
        transcribe_ms = int(payload.get("transcribe_ms", 0))
        tx_sec = transcribe_ms / 1000.0
        rtfx = float(payload.get("rtfx", 0.0))
        logging.info(
            f"  Transcribed {len(statements)} statements, {audio_sec:.2f}s speech "
            f"in {tx_sec:.2f}s (RTFx: {rtfx:.2f}) [model={model_info['model']}]"
        )

        return statements

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def get_model_info(config: dict) -> dict:
    """Return Parakeet model metadata for transcript JSONL headers."""
    model_version, _cache_dir, _timeout_sec = _validate_config(config)
    helper_path = _resolve_helper_path()
    version_envelope = _probe_helper_version(helper_path)
    return {
        "model": f"parakeet-tdt-0.6b-{model_version}",
        "device": "ane",
        "compute_type": "coreml_fp16",
        "fluidaudio_version": version_envelope["fluidaudio_version"],
        "helper_hardware": version_envelope["hardware"],
    }


def _resolve_helper_path() -> Path:
    """Resolve the Parakeet helper path from env override or default build path."""
    env_path = os.getenv(_HELPER_ENV_KEY)
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if not candidate.exists():
            raise RuntimeError(
                f"Parakeet helper not found at ${_HELPER_ENV_KEY}={candidate}. "
                f"Run 'make parakeet-helper' or point ${_HELPER_ENV_KEY} at a valid executable."
            )
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(
                f"Parakeet helper at ${_HELPER_ENV_KEY}={candidate} is not executable. "
                f"Run 'make parakeet-helper', chmod +x the file, or point "
                f"${_HELPER_ENV_KEY} at a valid executable."
            )
        return candidate

    candidate = (
        Path(__file__).with_name("parakeet_helper")
        / ".build"
        / "release"
        / "parakeet-helper"
    ).resolve()
    if (
        not candidate.exists()
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
    ):
        if is_packaged_install():
            raise RuntimeError(PACKAGED_COREML_HINT)
        raise RuntimeError(
            f"Parakeet helper not found at {candidate}. Run 'make parakeet-helper' "
            f"or set ${_HELPER_ENV_KEY} to a valid executable."
        )
    return candidate


def _probe_helper_version(helper_path: Path) -> dict:
    """Probe and cache the helper version envelope by resolved binary path."""
    cache_key = str(helper_path.resolve())
    if cache_key in _VERSION_CACHE:
        return _VERSION_CACHE[cache_key]

    try:
        result = subprocess.run(
            [str(helper_path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Parakeet helper timed out after 10.0s. Rebuild with "
            f"'make parakeet-helper', increase transcribe.parakeet.timeout_sec, "
            f"or set ${_HELPER_ENV_KEY} to a different helper build."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        try:
            stderr_json = json.loads(stderr) if stderr else {}
        except json.JSONDecodeError:
            stderr_json = {}
        message = stderr_json.get("message") or stderr or "unknown helper failure"
        if result.returncode == 2:
            raise RuntimeError(
                f"Parakeet helper rejected validated arguments (internal bug — file an issue): {message}"
            )
        if result.returncode == 3:
            raise RuntimeError(
                f"Parakeet helper could not prepare cache dir {_DEFAULT_CACHE_DIR}: {message}"
            )
        if result.returncode == 4:
            raise RuntimeError(
                f"Parakeet helper failed to download or load model '{_DEFAULT_MODEL_VERSION}'. "
                f"Valid values: v2, v3. {message}"
            )
        if result.returncode == 5:
            raise RuntimeError(f"Parakeet helper failed to transcribe audio: {message}")
        raise RuntimeError(
            f"Parakeet helper failed with exit code {result.returncode}: {message}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Parakeet helper returned invalid version JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Parakeet helper returned a non-object version payload")

    _VERSION_CACHE[cache_key] = payload
    return payload


def _collapse_subwords_to_words(token_timings: list[dict]) -> list[dict]:
    """Collapse helper token timings into repo-standard word dicts."""
    if not token_timings:
        return []

    words = []
    current_parts: list[str] = []
    current_confidences: list[float] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_parts, current_confidences, current_start, current_end
        if not current_parts or current_start is None or current_end is None:
            current_parts = []
            current_confidences = []
            current_start = None
            current_end = None
            return
        text = "".join(current_parts).lstrip()
        words.append(
            {
                "word": f" {text}",
                "start": current_start,
                "end": current_end,
                "probability": min(current_confidences),
            }
        )
        current_parts = []
        current_confidences = []
        current_start = None
        current_end = None

    for token in token_timings:
        raw = str(token.get("token", ""))
        is_punctuation = bool(raw) and all(
            char in _PUNCTUATION_TOKENS for char in raw if char
        )
        starts_new = raw.startswith("▁") or raw.startswith(" ")

        if starts_new and current_parts and not is_punctuation:
            flush()

        cleaned = raw.lstrip("▁")
        if starts_new:
            cleaned = cleaned.lstrip(" ")

        current_parts.append(cleaned)
        current_confidences.append(float(token.get("confidence", 0.0)))

        if not is_punctuation and current_start is None:
            current_start = float(token["start"])
        if not is_punctuation:
            current_end = float(token["end"])

    flush()
    return words


def _log_drift_if_needed(fluid_transcript: str, rebuilt_text: str) -> None:
    """Warn when helper text and rebuilt text materially diverge."""
    fluid_transcript = fluid_transcript.strip()
    rebuilt_text = rebuilt_text.strip()
    if not fluid_transcript or not rebuilt_text:
        return

    ratio = difflib.SequenceMatcher(None, fluid_transcript, rebuilt_text).ratio()
    if ratio < 0.95:
        logging.warning(
            "Parakeet transcript drift detected (ratio=%.3f): helper=%r rebuilt=%r",
            ratio,
            fluid_transcript,
            rebuilt_text,
        )


def _validate_config(config: dict) -> tuple[str, Path, float]:
    """Validate backend config before spawning the helper."""
    model_version = config.get("model_version", _DEFAULT_MODEL_VERSION)
    if model_version not in _VALID_MODEL_VERSIONS:
        raise ValueError("model_version must be one of: v2, v3")

    raw_timeout = config.get("timeout_sec", _DEFAULT_TIMEOUT_SEC)
    try:
        timeout_sec = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}") from exc
    if timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}")

    raw_cache_dir = config.get("cache_dir")
    cache_dir = (
        Path(raw_cache_dir).expanduser() if raw_cache_dir else _DEFAULT_CACHE_DIR
    )

    return model_version, cache_dir, timeout_sec
