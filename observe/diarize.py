"""Speaker diarization using pyannote pipeline with embedding extraction.

This module provides speaker diarization (who spoke when) using the pyannote
speaker-diarization-3.1 pipeline, along with per-turn speaker embeddings for
future speaker identification.

Requires HUGGINGFACE_API_KEY environment variable for HuggingFace authentication.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PIPELINE_ID = "pyannote/speaker-diarization-3.1"

# Built-in parameters
SEGMENTATION_STEP = 0.2  # 0.1 = 90% overlap (default), 0.2 = 80% overlap (2x faster)
MIN_DURATION_OFF = 0.2  # Minimum silence duration to split turns
MIN_TURN_DURATION = 0.2  # Discard turns shorter than this


class DiarizationError(Exception):
    """Raised when diarization fails."""

    pass


def get_hf_token() -> str:
    """Get HuggingFace token from environment.

    Raises:
        DiarizationError: If HUGGINGFACE_API_KEY is not set.
    """
    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(
            "HUGGINGFACE_API_KEY environment variable required. "
            "Get your token at https://huggingface.co/settings/tokens"
        )
    return token


def diarize(
    audio_path: Path,
) -> tuple[list[dict], dict[str, np.ndarray], dict, list[dict]]:
    """Run speaker diarization and extract per-speaker embeddings.

    Uses exclusive diarization (no overlapping speech in turns) for cleaner
    transcription segments. Overlapping speech regions are reported separately.

    Args:
        audio_path: Path to audio file (FLAC, WAV, etc.)

    Returns:
        Tuple of (turns, speaker_embeddings, timings, overlaps) where:
        - turns: List of dicts with "start", "end", "speaker" keys
          Speaker labels are human-readable: "Speaker 1", "Speaker 2", etc.
          Uses exclusive diarization (no overlapping segments).
        - speaker_embeddings: Dict mapping speaker labels to embedding arrays
        - timings: Dict with timing information for each stage
        - overlaps: List of dicts with "start", "end" keys for overlapping speech

    Raises:
        DiarizationError: If diarization fails (missing token, access denied, etc.)
    """
    from pyannote.audio import Inference, Pipeline

    hf_token = get_hf_token()
    timings: dict[str, float] = {}
    input_path = str(audio_path)

    # Get audio info
    info = sf.info(input_path)
    logging.info(f"Diarizing: {audio_path.name} ({info.duration:.1f}s)")

    # Load pipeline
    logging.info("Loading diarization pipeline...")
    t0 = time.perf_counter()
    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=hf_token)
    except Exception as e:
        if "403" in str(e) or "gated" in str(e).lower():
            raise DiarizationError(
                f"Access denied to {PIPELINE_ID}. "
                f"Accept terms at https://huggingface.co/{PIPELINE_ID}"
            ) from e
        raise DiarizationError(f"Failed to load diarization pipeline: {e}") from e

    # Use GPU if available
    import torch

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        logging.info("  Using CUDA")

    # Apply tunable parameters
    if SEGMENTATION_STEP != 0.1:
        seg_duration = pipeline._segmentation.model.specifications.duration
        pipeline._segmentation = Inference(
            pipeline._segmentation.model,
            duration=seg_duration,
            step=SEGMENTATION_STEP * seg_duration,
            skip_aggregation=True,
            batch_size=pipeline._segmentation.batch_size,
        )
        logging.info(
            f"  Segmentation step: {SEGMENTATION_STEP} "
            f"({int((1 - SEGMENTATION_STEP) * 100)}% overlap)"
        )

    if MIN_DURATION_OFF != 0.0:
        pipeline.segmentation.min_duration_off = MIN_DURATION_OFF
        logging.info(f"  Min silence to split: {MIN_DURATION_OFF}s")

    timings["pipeline_load"] = time.perf_counter() - t0
    logging.info(f"  Pipeline loaded in {timings['pipeline_load']:.2f}s")

    # Run diarization
    logging.info("Running diarization...")
    t0 = time.perf_counter()
    try:
        diarization = pipeline(input_path)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e
    timings["diarization"] = time.perf_counter() - t0
    logging.info(f"  Diarization: {timings['diarization']:.2f}s")

    # Extract turns from exclusive diarization (no overlapping segments)
    # and map speaker labels to human-readable names
    raw_turns: list[dict] = []
    speaker_map: dict[str, str] = {}  # SPEAKER_00 -> Speaker 1

    for turn, _, speaker in diarization.exclusive_speaker_diarization.itertracks(
        yield_label=True
    ):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
        raw_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker_map[speaker],
            }
        )

    # Extract overlapping speech regions from regular diarization
    overlap_timeline = diarization.speaker_diarization.get_overlap()
    overlaps: list[dict] = [
        {"start": float(seg.start), "end": float(seg.end)} for seg in overlap_timeline
    ]
    if overlaps:
        total_overlap = sum(o["end"] - o["start"] for o in overlaps)
        logging.info(f"  Found {len(overlaps)} overlap regions ({total_overlap:.1f}s)")

    if not raw_turns:
        logging.info("No speech detected.")
        timings["total"] = timings["pipeline_load"] + timings["diarization"]
        return [], {}, timings, overlaps

    n_speakers = len(speaker_map)
    logging.info(f"  Found {len(raw_turns)} turns, {n_speakers} speakers")

    # Filter out short turns
    turns = [t for t in raw_turns if t["end"] - t["start"] >= MIN_TURN_DURATION]
    if len(turns) < len(raw_turns):
        logging.info(
            f"  Filtered {len(raw_turns) - len(turns)} turns "
            f"< {MIN_TURN_DURATION}s -> {len(turns)} remaining"
        )

    if not turns:
        logging.info("No turns remaining after filtering.")
        timings["total"] = timings["pipeline_load"] + timings["diarization"]
        return [], {}, timings, overlaps

    # Extract per-speaker embeddings from pipeline output
    # Embeddings are indexed by speaker order matching exclusive_speaker_diarization.labels()
    speaker_embeddings: dict[str, np.ndarray] = {}
    if diarization.speaker_embeddings is not None:
        labels = diarization.exclusive_speaker_diarization.labels()
        for idx, raw_label in enumerate(labels):
            if raw_label in speaker_map and idx < len(diarization.speaker_embeddings):
                emb = diarization.speaker_embeddings[idx]
                # Normalize for cosine similarity
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                speaker_embeddings[speaker_map[raw_label]] = emb.astype(np.float32)
        logging.info(f"  Extracted embeddings for {len(speaker_embeddings)} speakers")

    timings["total"] = timings["pipeline_load"] + timings["diarization"]

    return turns, speaker_embeddings, timings, overlaps


def save_speaker_embeddings(
    output_dir: Path,
    speaker_embeddings: dict[str, np.ndarray],
) -> list[Path]:
    """Save per-speaker embeddings to NPZ files.

    Args:
        output_dir: Directory to save embeddings (e.g., segment_dir/audio_stem/)
        speaker_embeddings: Dict mapping speaker labels to embedding arrays

    Returns:
        List of paths to saved NPZ files
    """
    if not speaker_embeddings:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for speaker, embedding in speaker_embeddings.items():
        emb_path = output_dir / f"{speaker}.npz"
        np.savez_compressed(emb_path, embedding=embedding)
        saved_paths.append(emb_path)
        logging.info(f"  Saved embedding: {emb_path}")

    return saved_paths
