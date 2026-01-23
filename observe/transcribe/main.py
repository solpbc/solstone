# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files with pluggable STT backends and sentence-level embeddings.

Transcription pipeline:
1. VAD stage: Run Silero VAD to detect speech and filter silent files early
2. Audio reduction: Trim long silence gaps for faster processing
3. Transcription: Dispatch to STT backend (default: whisper)
4. Enrichment: Extract topics, setting, emotions, and warnings via LLM (optional)
5. Embeddings: Generate voice embeddings for each sentence using resemblyzer
6. Output: JSONL format compatible with format_audio() in observe/hear.py

Output files:
- <stem>.jsonl: Transcript with HH:MM:SS timestamps, topics, setting, emotions
- <stem>.npz: Sentence-level voice embeddings indexed by statement id

Configuration (journal config transcribe section):
- transcribe.backend: STT backend ("whisper", "revai", "gemini"). Default: "whisper"
- transcribe.enrich: Enable/disable LLM enrichment (default: true)
- transcribe.preserve_all: Keep audio files even when no speech detected (default: false)
- transcribe.min_speech_seconds: Minimum speech duration to proceed. Default: 1.0
- transcribe.noise_upgrade: Auto-switch to Rev.ai for noisy recordings (default: true)

Whisper backend settings (transcribe.whisper):
- device: Device for inference ("auto", "cpu", "cuda"). Default: "auto"
- model: Whisper model size (e.g., "medium.en"). Default: "medium.en"
- compute_type: Precision ("default", "float32", "float16", "int8"). Default: "default"
  Auto-selects: float16 for CUDA, int8 for CPU (including Apple Silicon).

Rev.ai backend settings (transcribe.revai):
- model: Rev.ai transcriber ("fusion", "machine", "low_cost"). Default: "fusion"
- Automatically loads recent entity names as custom vocabulary for improved recognition

Gemini backend settings (transcribe.gemini):
- No configuration needed (model resolved by muse.models context system)
- Includes speaker diarization

Platform optimizations (Whisper):
- CUDA GPU: Uses float16 for GPU-optimized inference
- Apple Silicon: Uses int8 for Whisper (~2x faster), MPS for embeddings (~16x faster)
- Other CPU: Uses int8 for best performance
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from observe.transcribe import (
    BACKEND_REGISTRY,
    get_backend,
)
from observe.transcribe import transcribe as stt_transcribe
from observe.transcribe.utils import is_apple_silicon
from observe.transcribe.whisper import DEFAULT_COMPUTE, DEFAULT_DEVICE, DEFAULT_MODEL
from observe.utils import SAMPLE_RATE, get_segment_key
from observe.vad import (
    AudioReduction,
    VadResult,
    reduce_audio,
    restore_statement_timestamps,
    run_vad,
)
from think.callosum import callosum_send
from think.utils import get_config, get_journal, setup_cli

# Re-export defaults for backwards compatibility
__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_DEVICE",
    "DEFAULT_COMPUTE",
    "DEFAULT_MIN_SPEECH_SECONDS",
    "MIN_STATEMENT_DURATION",
    "main",
]

# Default transcription settings
DEFAULT_BACKEND = "whisper"
DEFAULT_MIN_SPEECH_SECONDS = 1.0

# Minimum statement duration for embedding (seconds)
MIN_STATEMENT_DURATION = 0.3

# Number of recent entity names to load for transcription context
ENTITY_NAMES_LIMIT = 40

# Module-level voice encoder cache
_voice_encoder = None


def _get_optimal_encoder_device() -> str:
    """Get optimal device for VoiceEncoder (resemblyzer).

    On Apple Silicon, MPS provides ~16x speedup over CPU for embeddings.

    Returns:
        Device string: "mps" on Apple Silicon with MPS available, "cpu" otherwise
    """
    if is_apple_silicon():
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    return "cpu"


def _get_voice_encoder(whisper_device: str = "cpu"):
    """Get or create VoiceEncoder with caching.

    Args:
        whisper_device: Device used by whisper (affects encoder device selection)

    Returns:
        VoiceEncoder instance
    """
    global _voice_encoder
    from resemblyzer import VoiceEncoder

    if _voice_encoder is not None:
        return _voice_encoder

    # VoiceEncoder: use MPS on Apple Silicon for ~16x speedup, otherwise CPU
    # (CUDA auto-detection handled by resemblyzer when device=None)
    if whisper_device == "cuda":
        encoder_device = None  # Let resemblyzer auto-detect CUDA
    else:
        encoder_device = _get_optimal_encoder_device()

    logging.info("Loading resemblyzer VoiceEncoder...")
    t0 = time.perf_counter()
    _voice_encoder = VoiceEncoder(device=encoder_device)
    logging.info(
        f"  VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s "
        f"(device={_voice_encoder.device})"
    )

    return _voice_encoder


def _get_jsonl_path(audio_path: Path) -> Path:
    """Generate the corresponding JSONL path."""
    return audio_path.with_suffix(".jsonl")


def _get_embeddings_path(audio_path: Path) -> Path:
    """Generate the corresponding embeddings path."""
    return audio_path.with_suffix(".npz")


def _build_base_event(
    audio_path: Path,
    vad_result: VadResult,
    segment: str | None = None,
    remote: str | None = None,
) -> dict:
    """Build base event dict for callosum emission.

    Args:
        audio_path: Path to the audio file
        vad_result: VAD result with speech detection info
        segment: Optional segment key (e.g., "143022_300")
        remote: Optional remote name

    Returns:
        Event dict with common fields for observe.transcribed events
    """
    journal_path = Path(get_journal())
    day = audio_path.parent.parent.name

    try:
        rel_input = audio_path.relative_to(journal_path)
    except ValueError:
        rel_input = audio_path

    event = {
        "input": str(rel_input),
        "vad_duration": round(vad_result.duration, 1),
        "vad_speech": round(vad_result.speech_duration, 1),
        "noisy": vad_result.is_noisy(),
    }

    # Add RMS values if available
    if vad_result.noisy_rms is not None:
        event["noisy_rms"] = round(vad_result.noisy_rms, 4)
        event["noisy_s"] = round(vad_result.noisy_s, 1)

    if day:
        event["day"] = day
    if segment:
        event["segment"] = segment
    if remote:
        event["remote"] = remote

    return event


def _embed_statements(
    audio: np.ndarray,
    statements: list[dict],
    sample_rate: int,
    whisper_device: str = "cpu",
) -> dict[str, np.ndarray] | None:
    """Generate voice embeddings for each statement.

    Args:
        audio: Audio buffer (float32, mono)
        statements: List of statements
        sample_rate: Sample rate in Hz
        whisper_device: Device used by whisper (affects encoder device selection)

    Returns:
        Dict with embedding data or None on error:
        - embeddings: (N, 256) float32 array
        - statement_ids: (N,) int32 array of statement IDs
    """
    try:
        voice_encoder = _get_voice_encoder(whisper_device)

        audio_duration = len(audio) / sample_rate

        # Filter statements with valid timestamps and sufficient duration
        # Defensive: handle None timestamps, clamp to audio bounds
        valid_statements = []
        for s in statements:
            start = s.get("start")
            end = s.get("end")

            # Skip if timestamps are None or invalid
            if start is None or end is None:
                continue
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue

            # Clamp to audio bounds
            start = max(0.0, min(start, audio_duration))
            end = max(0.0, min(end, audio_duration))

            # Check duration after clamping
            if end - start >= MIN_STATEMENT_DURATION:
                valid_statements.append({"id": s["id"], "start": start, "end": end})

        if not valid_statements:
            logging.info("No statements with sufficient duration for embedding")
            return None

        logging.info(f"Embedding {len(valid_statements)} statements...")
        t0 = time.perf_counter()

        embeddings = []
        statement_ids = []
        skipped = 0

        for stmt in valid_statements:
            start_sample = int(stmt["start"] * sample_rate)
            end_sample = int(stmt["end"] * sample_rate)
            stmt_audio = audio[start_sample:end_sample]

            # Skip if too short after slicing
            if len(stmt_audio) < int(MIN_STATEMENT_DURATION * sample_rate):
                skipped += 1
                continue

            try:
                emb = voice_encoder.embed_utterance(stmt_audio)
                embeddings.append(emb)
                statement_ids.append(stmt["id"])
            except Exception:
                skipped += 1
                continue

        embed_time = time.perf_counter() - t0

        if not embeddings:
            logging.warning("No embeddings generated")
            return None

        logging.info(
            f"  Embedded {len(embeddings)} statements "
            f"(skipped {skipped}) in {embed_time:.2f}s"
        )

        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "statement_ids": np.array(statement_ids, dtype=np.int32),
        }

    except Exception as e:
        logging.error(f"Embedding failed: {e}", exc_info=True)
        return None


def _statements_to_jsonl(
    statements: list[dict],
    raw_filename: str,
    base_datetime: datetime.datetime,
    model_info: dict,
    source: str | None = None,
    remote: str | None = None,
    enrichment: dict | None = None,
    vad_result: VadResult | None = None,
    segment_meta: dict | None = None,
    backend: str | None = None,
) -> list[str]:
    """Convert statements to JSONL lines.

    Args:
        statements: List of statements
        raw_filename: Original audio filename for metadata
        base_datetime: Base datetime for timestamp calculation
        model_info: Dict with model, device, compute_type from backend
        source: Optional source label (e.g., "mic", "sys")
        remote: Optional remote name for metadata
        enrichment: Optional enrichment data with topics, setting, warning, and
            per-statement corrected text and emotions
        vad_result: Optional VAD result for noise detection metadata
        segment_meta: Optional metadata dict from SEGMENT_META env var
            (facet, setting, host, platform, etc.). Setting overrides enrichment.
        backend: Optional STT backend name (e.g., "whisper", "revai")

    Returns:
        List of JSON strings (metadata line first, then entries)
    """
    # Build metadata line with transcription config
    metadata = {
        "raw": raw_filename,
        "backend": backend or "unknown",
        "model": model_info.get("model", "unknown"),
        "device": model_info.get("device", "unknown"),
        "compute_type": model_info.get("compute_type", "unknown"),
    }
    if remote:
        metadata["remote"] = remote

    # Add noise detection metadata if available
    if vad_result:
        metadata["noisy"] = vad_result.is_noisy()
        if vad_result.noisy_rms is not None:
            metadata["noisy_rms"] = round(vad_result.noisy_rms, 4)
            metadata["noisy_s"] = round(vad_result.noisy_s, 1)

    # Add enrichment metadata if available
    if enrichment:
        if "topics" in enrichment:
            metadata["topics"] = enrichment["topics"]
        if "setting" in enrichment:
            metadata["setting"] = enrichment["setting"]
        if "warning" in enrichment and enrichment["warning"]:
            metadata["warning"] = enrichment["warning"]

    # Add segment metadata (from SEGMENT_META env var)
    # These fields override any enrichment values (e.g., setting)
    if segment_meta:
        for key, value in segment_meta.items():
            metadata[key] = value

    lines = [json.dumps(metadata)]

    # Get enriched statements list (positional matching)
    enriched_statements = []
    if enrichment and "statements" in enrichment:
        enriched_statements = enrichment["statements"]

    # Build entry lines
    for i, stmt in enumerate(statements):
        # Calculate absolute timestamp (handle None for invalid timestamps)
        start_seconds = stmt["start"] if stmt["start"] is not None else 0.0
        stmt_dt = base_datetime + datetime.timedelta(seconds=start_seconds)
        timestamp_str = stmt_dt.strftime("%H:%M:%S")

        entry = {
            "start": timestamp_str,
            "text": stmt["text"],
        }
        if source:
            entry["source"] = source

        # Pass through speaker ID if present (from diarized backends like Rev.ai, Gemini)
        if "speaker" in stmt:
            entry["speaker"] = stmt["speaker"]

        # Add corrected text and emotion from enrichment by position
        if i < len(enriched_statements):
            enriched = enriched_statements[i]
            if isinstance(enriched, dict):
                # Add corrected text only if different from original
                corrected = enriched.get("corrected", "")
                if corrected and corrected != stmt["text"]:
                    entry["corrected"] = corrected
                # Add emotion (overrides emotion from statement)
                emotion = enriched.get("emotion", "")
                if emotion:
                    entry["emotion"] = emotion

        lines.append(json.dumps(entry))

    return lines


def process_audio(
    raw_path: Path,
    vad_result: VadResult,
    backend_config: dict,
    redo: bool = False,
    reduction: AudioReduction | None = None,
    reduced_audio: np.ndarray | None = None,
    backend: str = DEFAULT_BACKEND,
    entity_names: list[str] | None = None,
) -> None:
    """Process a raw audio file with pre-computed VAD.

    This is the main orchestration function that coordinates:
    - STT backend dispatch
    - Enrichment (optional)
    - Embedding generation
    - Output file writing
    - Event emission

    Args:
        raw_path: Path to audio file in journal segment directory (HHMMSS_LEN/)
        vad_result: Pre-computed VAD result from run_vad()
        backend_config: Configuration for STT backend
        redo: If True, skip "already processed" check
        reduction: Optional AudioReduction mapping for timestamp restoration
        reduced_audio: Optional reduced audio buffer (used if reduction provided)
        backend: STT backend name (default: "whisper")
        entity_names: Optional list of entity names for STT and enrichment context
    """
    from faster_whisper.audio import decode_audio

    start_time = time.time()

    # Derive segment from path
    segment = get_segment_key(raw_path)

    # Skip if already processed (unless redo mode)
    jsonl_path = _get_jsonl_path(raw_path)
    if not redo and jsonl_path.exists():
        logging.info(f"Already processed: {raw_path}")
        return

    # Get remote name once for use in metadata and events
    remote = os.getenv("REMOTE_NAME")

    # Get segment metadata (from sense.py via SEGMENT_META env var)
    segment_meta = None
    segment_meta_str = os.getenv("SEGMENT_META")
    if segment_meta_str:
        try:
            segment_meta = json.loads(segment_meta_str)
        except json.JSONDecodeError:
            logging.warning(f"Invalid SEGMENT_META JSON: {segment_meta_str[:100]}")

    # Prepare audio buffer for processing
    if reduced_audio is not None:
        audio_buffer = reduced_audio
    else:
        audio_buffer = decode_audio(str(raw_path), sampling_rate=SAMPLE_RATE)

    try:
        # Dispatch to STT backend
        statements = stt_transcribe(backend, audio_buffer, SAMPLE_RATE, backend_config)

        # Get model info for metadata (dynamic import based on backend)
        backend_module = get_backend(backend)
        model_info = backend_module.get_model_info(backend_config)

        # Sanity check: if VAD detected speech but we got no statements, something is wrong
        if vad_result.has_speech and not statements:
            raise RuntimeError(
                f"VAD detected {vad_result.speech_duration:.1f}s of speech "
                f"(from {vad_result.duration:.1f}s total) but transcription produced "
                f"0 statements. This indicates a transcription failure, not silence."
            )

        # Load config for preserve_all setting
        config = get_config()
        preserve_all = config.get("transcribe", {}).get("preserve_all", False)

        # Build base event fields (always emitted as observe.transcribed)
        event = _build_base_event(raw_path, vad_result, segment, remote)

        # Handle no speech detected
        if not statements:
            if preserve_all:
                event["outcome"] = "preserved"
                logging.info(
                    f"No speech detected in {raw_path}, preserving file "
                    f"(preserve_all=true, VAD: {vad_result.speech_duration:.1f}s "
                    f"of {vad_result.duration:.1f}s)"
                )
            else:
                event["outcome"] = "filtered"
                logging.info(f"No speech detected in {raw_path}, removing file")
                raw_path.unlink()

            callosum_send("observe", "transcribed", **event)
            return

        # Extract date and time from path structure
        journal_path = Path(get_journal())
        day = raw_path.parent.parent.name
        time_part = segment.split("_")[0] if segment else "000000"
        base_dt = datetime.datetime.strptime(f"{day}_{time_part}", "%Y%m%d_%H%M%S")

        # Extract source from <source>_audio pattern
        source = None
        suffix = raw_path.stem
        if suffix.endswith("_audio") and suffix != "audio":
            source = suffix[:-6]  # Remove "_audio" suffix

        # Run enrichment if enabled (extracts topics, setting, emotions, corrections)
        enrichment = None
        enrich_enabled = config.get("transcribe", {}).get("enrich", True)
        if enrich_enabled:
            from observe.enrich import enrich_transcript

            enrichment = enrich_transcript(
                audio_buffer, SAMPLE_RATE, statements, entity_names=entity_names
            )

        # Generate embeddings before timestamp restoration
        # Use reduced audio buffer if available for consistent timestamps
        embeddings_data = _embed_statements(
            audio_buffer, statements, SAMPLE_RATE, model_info.get("device", "cpu")
        )

        # Restore original timestamps if audio was reduced
        if reduction:
            statements = restore_statement_timestamps(statements, reduction)
            logging.info(
                f"  Restored timestamps from reduced audio "
                f"({reduction.reduced_duration:.1f}s -> {reduction.original_duration:.1f}s)"
            )

        # Convert to JSONL format (now with original timestamps)
        raw_filename = f"{raw_path.stem}{raw_path.suffix}"
        jsonl_lines = _statements_to_jsonl(
            statements,
            raw_filename,
            base_dt,
            model_info,
            source,
            remote,
            enrichment,
            vad_result,
            segment_meta,
            backend,
        )

        # Write JSONL
        jsonl_path.write_text("\n".join(jsonl_lines) + "\n")
        logging.info(f"Transcribed {raw_path} -> {jsonl_path}")

        # Save embeddings
        if embeddings_data:
            embeddings_path = _get_embeddings_path(raw_path)
            np.savez_compressed(embeddings_path, **embeddings_data)
            logging.info(f"Saved embeddings: {embeddings_path}")
        else:
            logging.warning(f"No embeddings generated for {raw_path}")

        # Add completion fields and emit event
        event["outcome"] = "transcribed"
        event["duration_ms"] = int((time.time() - start_time) * 1000)
        try:
            rel_output = jsonl_path.relative_to(journal_path)
        except ValueError:
            rel_output = jsonl_path
        event["output"] = str(rel_output)

        callosum_send("observe", "transcribed", **event)

    except Exception as e:
        logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
        raise SystemExit(1) from e


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using pluggable STT with sentence embeddings"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file in journal segment directory, e.g. HHMMSS_LEN/audio.flac",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Reprocess file, overwriting existing outputs",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (overrides config, uses int8 compute)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Whisper model to use (overrides config, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_REGISTRY.keys()),
        help=f"STT backend to use (overrides config, default: {DEFAULT_BACKEND})",
    )
    args = setup_cli(parser)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    # Validate supported formats
    supported_formats = {".flac", ".m4a", ".ogg", ".opus"}
    if audio_path.suffix.lower() not in supported_formats:
        parser.error(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Files must be in segment directories (YYYYMMDD/HHMMSS_LEN/)
    segment = get_segment_key(audio_path)
    if segment is None:
        parser.error(
            f"Audio file must be in a segment directory (HHMMSS_LEN/), "
            f"but parent is: {audio_path.parent.name}"
        )

    # Load transcription settings from journal config
    config = get_config()
    transcribe_config = config.get("transcribe", {})

    # Get min_speech_seconds threshold
    min_speech_seconds = transcribe_config.get(
        "min_speech_seconds", DEFAULT_MIN_SPEECH_SECONDS
    )
    preserve_all = transcribe_config.get("preserve_all", False)

    logging.info(f"Processing audio: {audio_path}")

    # Stage 1: Run VAD to detect speech (lightweight, before loading STT model)
    vad_result = run_vad(audio_path, min_speech_seconds=min_speech_seconds)

    # Early exit if no speech detected (skip loading heavy STT model)
    if not vad_result.has_speech:
        remote = os.getenv("REMOTE_NAME")
        event = _build_base_event(audio_path, vad_result, segment, remote)

        if preserve_all:
            event["outcome"] = "preserved"
            logging.info(
                f"Insufficient speech in {audio_path}, preserving file "
                f"(preserve_all=true, VAD: {vad_result.speech_duration:.1f}s "
                f"of {vad_result.duration:.1f}s, threshold: {min_speech_seconds:.1f}s)"
            )
        else:
            event["outcome"] = "filtered"
            logging.info(
                f"Insufficient speech in {audio_path}, removing file "
                f"(VAD: {vad_result.speech_duration:.1f}s of {vad_result.duration:.1f}s, "
                f"threshold: {min_speech_seconds:.1f}s)"
            )
            audio_path.unlink()

        callosum_send("observe", "transcribed", **event)
        return

    # Stage 2: Reduce audio by trimming long silence gaps (>2s)
    reduced_audio, reduction = reduce_audio(audio_path, vad_result)

    # Stage 3: Determine backend and build backend config
    # CLI --backend flag overrides config, otherwise use config or default
    backend = args.backend or transcribe_config.get("backend", DEFAULT_BACKEND)

    # Check for noise upgrade: auto-switch to Rev.ai for noisy recordings
    # Only applies when:
    # - No explicit CLI --backend flag (respect user's explicit choice)
    # - Not already using Rev.ai
    # - noise_upgrade is enabled (default: true)
    # - Audio is noisy
    # - Rev.ai token is available
    noise_upgrade = transcribe_config.get("noise_upgrade", True)
    if (
        not args.backend
        and noise_upgrade
        and backend != "revai"
        and vad_result.is_noisy()
    ):
        from observe.transcribe.revai import has_token

        if has_token():
            logging.info(
                f"Noisy audio detected (RMS={vad_result.noisy_rms:.4f}), "
                f"upgrading to Rev.ai backend"
            )
            backend = "revai"

    # Load entity names once for use by both STT backend and enrichment
    from think.entities import load_recent_entity_names

    entity_names = load_recent_entity_names(limit=ENTITY_NAMES_LIMIT)
    if entity_names:
        logging.info(f"Loaded {len(entity_names)} entities for transcription context")

    # Get backend-specific config from nested structure
    if backend == "whisper":
        whisper_config = transcribe_config.get("whisper", {})
        if args.cpu:
            device = "cpu"
            compute_type = "int8"
        else:
            device = whisper_config.get("device", DEFAULT_DEVICE)
            compute_type = whisper_config.get("compute_type", DEFAULT_COMPUTE)
        model = args.model or whisper_config.get("model", DEFAULT_MODEL)
        backend_config = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
        }
    elif backend == "revai":
        from observe.transcribe.revai import DEFAULT_MODEL as REVAI_DEFAULT_MODEL

        revai_config = transcribe_config.get("revai", {})
        model = revai_config.get("model", REVAI_DEFAULT_MODEL)
        backend_config = {
            "model": model,
        }
        # Pass entities to Rev.ai for custom vocabulary
        if entity_names:
            backend_config["entities"] = entity_names
    elif backend == "gemini":
        # Gemini backend - model resolved by muse.models based on context
        # Entity names handled by enrich step, not passed to transcription
        backend_config = {}
    else:
        # Unknown backend - let get_backend() raise the error
        backend_config = {}

    # Stage 4: Process audio with STT backend
    process_audio(
        audio_path,
        vad_result,
        backend_config,
        redo=args.redo,
        reduction=reduction,
        reduced_audio=reduced_audio,
        backend=backend,
        entity_names=entity_names,
    )


if __name__ == "__main__":
    main()
