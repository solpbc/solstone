"""Speaker diarization using pyannote pipeline with embedding extraction.

This module provides speaker diarization (who spoke when) using the pyannote
community pipeline, along with per-turn speaker embeddings for future
speaker identification.

Requires HF_TOKEN environment variable for HuggingFace authentication.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PIPELINE_ID = "pyannote/speaker-diarization-3.1"
EMB_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"

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
        DiarizationError: If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise DiarizationError(
            "HF_TOKEN environment variable required. "
            "Get your token at https://huggingface.co/settings/tokens"
        )
    return token


def diarize(
    audio_path: Path,
) -> tuple[list[dict], np.ndarray, dict]:
    """Run speaker diarization and extract per-turn embeddings.

    Args:
        audio_path: Path to audio file (FLAC, WAV, etc.)

    Returns:
        Tuple of (turns, embeddings, timings) where:
        - turns: List of dicts with "start", "end", "speaker" keys
          Speaker labels are human-readable: "Speaker 1", "Speaker 2", etc.
        - embeddings: numpy array of shape (num_turns, 256)
        - timings: Dict with timing information for each stage

    Raises:
        DiarizationError: If diarization fails (missing token, access denied, etc.)
    """
    from pyannote.audio import Inference, Model, Pipeline
    from pyannote.core import Segment

    hf_token = get_hf_token()
    timings: dict[str, float] = {}
    input_path = str(audio_path)

    # Get audio info
    info = sf.info(input_path)
    audio_duration = info.duration
    logging.info(f"Diarizing: {audio_path.name} ({audio_duration:.1f}s)")

    # Load pipeline
    logging.info("Loading diarization pipeline...")
    t0 = time.perf_counter()
    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=hf_token)
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

    # Extract turns and map speaker labels to human-readable names
    raw_turns: list[dict] = []
    speaker_map: dict[str, str] = {}  # SPEAKER_00 -> Speaker 1

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
        raw_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker_map[speaker],
            }
        )

    if not raw_turns:
        logging.info("No speech detected.")
        timings["total"] = timings["pipeline_load"] + timings["diarization"]
        return [], np.array([]), timings

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
        return [], np.array([]), timings

    # Load embedding model
    logging.info("Loading embedding model...")
    t0 = time.perf_counter()
    emb_model = Model.from_pretrained(EMB_MODEL_ID, use_auth_token=hf_token)
    if torch.cuda.is_available():
        emb_model.to(torch.device("cuda"))
    emb_infer = Inference(emb_model, window="whole")
    timings["emb_model_load"] = time.perf_counter() - t0

    # Extract embeddings for each turn
    logging.info(f"Extracting embeddings for {len(turns)} turns...")
    t0 = time.perf_counter()
    emb_list: list[np.ndarray] = []

    for i, turn in enumerate(turns):
        try:
            emb_vec = emb_infer.crop(input_path, Segment(turn["start"], turn["end"]))
            if emb_vec.ndim > 1:
                emb_vec = emb_vec.mean(axis=0)
            emb_list.append(emb_vec.astype(np.float32))
        except Exception as e:
            logging.warning(f"Failed to extract embedding for turn {i}: {e}")
            emb_list.append(np.zeros(256, dtype=np.float32))

        if (i + 1) % 20 == 0:
            logging.info(f"  Embedded {i + 1}/{len(turns)} turns")

    embeddings = np.stack(emb_list, axis=0)
    timings["embedding"] = time.perf_counter() - t0
    logging.info(f"  Embeddings: {timings['embedding']:.2f}s ({embeddings.shape})")

    timings["total"] = (
        timings["pipeline_load"]
        + timings["diarization"]
        + timings["emb_model_load"]
        + timings["embedding"]
    )

    return turns, embeddings, timings


def save_speaker_embeddings(
    output_dir: Path,
    turns: list[dict],
    embeddings: np.ndarray,
) -> list[Path]:
    """Save per-speaker mean embeddings to NPZ files.

    Args:
        output_dir: Directory to save embeddings (e.g., segment_dir/audio_stem/)
        turns: List of turn dicts with "speaker" key
        embeddings: Array of shape (num_turns, 256)

    Returns:
        List of paths to saved NPZ files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group embeddings by speaker
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    for turn, emb in zip(turns, embeddings):
        speaker = turn["speaker"]
        if speaker not in speaker_embeddings:
            speaker_embeddings[speaker] = []
        speaker_embeddings[speaker].append(emb)

    # Save mean embedding per speaker
    saved_paths: list[Path] = []
    for speaker, embs in speaker_embeddings.items():
        mean_emb = np.mean(embs, axis=0).astype(np.float32)
        # Normalize for cosine similarity
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        emb_path = output_dir / f"{speaker}.npz"
        np.savez_compressed(emb_path, embedding=mean_emb)
        saved_paths.append(emb_path)
        logging.info(f"  Saved embedding: {emb_path}")

    return saved_paths
