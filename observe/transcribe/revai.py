# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Rev.ai STT backend with speaker diarization.

This module provides cloud-based speech-to-text transcription using Rev.ai,
a professional transcription API with high-quality speaker diarization.

Unlike the local Whisper backend, Rev.ai:
- Provides speaker diarization (identifies who said what)
- Returns per-speaker segments (not per-sentence)
- Requires API credentials and network access
- Processes asynchronously (submit job, poll, fetch)

Configuration keys (passed in config dict):
- language: ISO language code (default: "en")
- model: Rev transcriber ("fusion", "machine", "low_cost"). Default: "fusion"
- diarization_type: Diarization quality ("standard", "premium"). Default: "premium"
- forced_alignment: Improve per-word timestamps (default: False)
- speakers_count: Hint for total unique speakers (optional)
- speaker_channels_count: For multichannel files (optional)
- remove_disfluencies: Remove ums/uhs (default: False)
- filter_profanity: Replace profanities (default: False)
- skip_punctuation: Disable punctuation (default: False)
- entities: Custom vocabulary terms (optional list)
- poll_interval: Seconds between status polls (default: 2.5)
- timeout: Overall timeout in seconds (default: 1800)

Environment:
- REVAI_ACCESS_TOKEN or REV_ACCESS_TOKEN: API access token (required)
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv

API_BASE = "https://api.rev.ai/speechtotext/v1"

# Default configuration
DEFAULT_MODEL = "fusion"
DEFAULT_LANGUAGE = "en"
DEFAULT_DIARIZATION = "premium"
DEFAULT_POLL_INTERVAL = 2.5
DEFAULT_TIMEOUT = 1800  # 30 minutes

# Valid Rev.ai transcriber models
VALID_MODELS = frozenset({"fusion", "machine", "low_cost", "human"})


def _get_token() -> str:
    """Get Rev.ai API token from environment.

    Returns:
        API access token

    Raises:
        ValueError: If token not found in environment
    """
    load_dotenv()
    token = os.getenv("REVAI_ACCESS_TOKEN") or os.getenv("REV_ACCESS_TOKEN")
    if not token:
        raise ValueError("Missing REVAI_ACCESS_TOKEN in environment")
    return token


def submit_job(
    token: str,
    media_path: Path,
    config: dict,
) -> str:
    """Submit transcription job to Rev.ai.

    Args:
        token: API access token
        media_path: Path to audio/video file
        config: Backend configuration dict

    Returns:
        Job ID string

    Raises:
        RuntimeError: If job submission fails
    """
    url = f"{API_BASE}/jobs"
    headers = {"Authorization": f"Bearer {token}"}

    # Build options from config
    # Validate model - use default if not a valid Rev.ai model (e.g., whisper model passed)
    model = config.get("model", DEFAULT_MODEL)
    if model not in VALID_MODELS:
        model = DEFAULT_MODEL

    options = {
        "transcriber": model,
        "skip_diarization": False,
        "diarization_type": config.get("diarization_type", DEFAULT_DIARIZATION),
        "language": config.get("language", DEFAULT_LANGUAGE),
        "forced_alignment": config.get("forced_alignment", False),
        "remove_disfluencies": config.get("remove_disfluencies", False),
        "filter_profanity": config.get("filter_profanity", False),
        "skip_punctuation": config.get("skip_punctuation", False),
    }

    if config.get("speakers_count") is not None:
        options["speakers_count"] = config["speakers_count"]
    if config.get("speaker_channels_count") is not None:
        options["speaker_channels_count"] = config["speaker_channels_count"]
    if config.get("entities"):
        options["custom_vocabularies"] = [{"phrases": config["entities"]}]
        options["strict_custom_vocabulary"] = False

    data = {"options": json.dumps(options)}

    logging.info("Submitting job to Rev.ai: %s", json.dumps(options))

    # Use context manager to ensure file handle is closed
    with open(media_path, "rb") as media_file:
        files = {
            "media": (
                media_path.name,
                media_file,
                mimetypes.guess_type(media_path.name)[0] or "application/octet-stream",
            )
        }
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)

    if resp.status_code >= 300:
        raise RuntimeError(
            f"Rev.ai job submission failed ({resp.status_code}): {resp.text}"
        )

    return resp.json()["id"]


def get_job(token: str, job_id: str) -> dict:
    """Get job status from Rev.ai.

    Args:
        token: API access token
        job_id: Job ID string

    Returns:
        Job status dict

    Raises:
        RuntimeError: If request fails
    """
    url = f"{API_BASE}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Rev.ai get job failed ({resp.status_code}): {resp.text}")
    return resp.json()


def get_transcript_json(token: str, job_id: str) -> dict:
    """Get transcript JSON from completed job.

    Args:
        token: API access token
        job_id: Job ID string

    Returns:
        Transcript dict with monologues

    Raises:
        RuntimeError: If request fails
    """
    url = f"{API_BASE}/jobs/{job_id}/transcript"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.rev.transcript.v1.0+json",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Rev.ai get transcript failed ({resp.status_code}): {resp.text}"
        )
    return resp.json()


def transcribe_file(media_path: Path, config: dict | None = None) -> dict:
    """Transcribe a file using Rev.ai API and return raw JSON.

    This is the low-level function that handles the full API flow:
    submit job → poll for completion → fetch transcript.

    Args:
        media_path: Path to audio/video file
        config: Optional configuration dict

    Returns:
        Raw Rev.ai transcript JSON (monologues format)

    Raises:
        ValueError: If API token not found
        RuntimeError: If job fails
        TimeoutError: If job times out
    """
    config = config or {}
    token = _get_token()

    poll_interval = config.get("poll_interval", DEFAULT_POLL_INTERVAL)
    timeout = config.get("timeout", DEFAULT_TIMEOUT)

    # Submit job
    job_id = submit_job(token, media_path, config)
    logging.info("Rev.ai job submitted: %s", job_id)

    # Poll for completion
    start = time.time()
    status = None
    while True:
        job = get_job(token, job_id)
        new_status = job.get("status")
        if new_status != status:
            logging.info("Rev.ai status: %s", new_status)
            status = new_status

        if new_status in ("transcribed", "completed"):
            break
        if new_status in ("failed", "error"):
            raise RuntimeError(f"Rev.ai job failed: {json.dumps(job, indent=2)}")

        if time.time() - start > timeout:
            raise TimeoutError("Rev.ai transcription timed out")

        time.sleep(poll_interval)

    # Fetch and return transcript
    return get_transcript_json(token, job_id)


def convert_to_segments(revai_json: dict) -> list[dict]:
    """Convert Rev.ai transcript to standard segment format.

    This produces per-speaker segments (one segment per monologue),
    preserving speaker attribution and word-level data.

    Args:
        revai_json: Raw Rev.ai transcript dict with monologues

    Returns:
        List of segment dicts with id, start, end, text, speaker, words
    """
    segments = []

    if "monologues" not in revai_json:
        return segments

    for monologue in revai_json["monologues"]:
        # Rev uses 0-based speakers, we use 1-based
        speaker = monologue.get("speaker", 0) + 1
        elements = monologue.get("elements", [])

        if not elements:
            continue

        # Build text and collect word data
        text_parts = []
        words = []
        start_ts = None
        end_ts = None
        confidences = []

        for elem in elements:
            if elem["type"] == "text":
                value = elem.get("value", "")
                text_parts.append(value)

                # Track timestamps
                ts = elem.get("ts")
                if ts is not None:
                    if start_ts is None:
                        start_ts = ts
                    end_ts = elem.get("end_ts", ts)

                # Build word entry
                word_entry = {"word": value}
                if ts is not None:
                    word_entry["start"] = ts
                if elem.get("end_ts") is not None:
                    word_entry["end"] = elem["end_ts"]
                if elem.get("confidence") is not None:
                    word_entry["probability"] = elem["confidence"]
                    confidences.append(elem["confidence"])
                words.append(word_entry)

            elif elem["type"] == "punct":
                # Append punctuation to text
                text_parts.append(elem.get("value", ""))

        text = "".join(text_parts).strip()
        if not text:
            continue

        # Build segment
        segment = {
            "id": len(segments) + 1,
            "start": start_ts if start_ts is not None else 0.0,
            "end": end_ts if end_ts is not None else 0.0,
            "text": text,
            "speaker": speaker,
            "words": words if words else None,
        }

        # Add average confidence if available
        if confidences:
            segment["confidence"] = sum(confidences) / len(confidences)

        segments.append(segment)

    return segments


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[dict]:
    """Transcribe audio using Rev.ai API.

    This is the standard backend interface. It writes the audio to a temp file,
    submits to Rev.ai, polls for completion, and returns normalized segments.

    Args:
        audio: Audio buffer (float32, mono)
        sample_rate: Sample rate in Hz (typically 16000)
        config: Backend configuration dict

    Returns:
        List of per-speaker segments with id, start, end, text, speaker, words
    """
    temp_path = None
    try:
        # Write audio to temp file for upload
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            temp_path = Path(f.name)

        # Convert to int16 for FLAC encoding
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(temp_path, audio_int16, sample_rate, format="FLAC")

        logging.info(
            "Transcribing audio with Rev.ai (%.1fs)...", len(audio) / sample_rate
        )

        # Get raw transcript
        revai_json = transcribe_file(temp_path, config)

        # Convert to standard segment format
        segments = convert_to_segments(revai_json)

        logging.info("  Rev.ai returned %d speaker segments", len(segments))

        return segments

    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()


def get_model_info(config: dict) -> dict:
    """Get model configuration info for metadata.

    Args:
        config: Backend configuration dict

    Returns:
        Dict with model info for JSONL metadata
    """
    model = config.get("model", DEFAULT_MODEL)
    if model not in VALID_MODELS:
        model = DEFAULT_MODEL

    return {
        "model": f"revai-{model}",
        "device": "cloud",
        "compute_type": "api",
        "diarization": config.get("diarization_type", DEFAULT_DIARIZATION),
    }
