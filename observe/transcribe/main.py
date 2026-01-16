# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files with pluggable STT backends and sentence-level embeddings.

Transcription pipeline:
1. VAD stage: Run Silero VAD to detect speech and filter silent files early
2. Audio reduction: Trim long silence gaps for faster processing
3. Transcription: Dispatch to STT backend (default: whisper)
4. Enrichment: Extract topics, setting, and audio descriptions via LLM (optional)
5. Embeddings: Generate voice embeddings for each sentence using resemblyzer
6. Output: JSONL format compatible with format_audio() in observe/hear.py

Output files:
- <stem>.jsonl: Transcript with HH:MM:SS timestamps, topics, setting, descriptions
- <stem>.npz: Sentence-level voice embeddings indexed by segment id

Configuration (journal config):
- transcribe.device: Device for inference ("auto", "cpu", "cuda"). Default: "auto"
- transcribe.model: Whisper model size (e.g., "medium.en"). Default: "medium.en"
- transcribe.compute_type: Precision ("default", "float32", "float16", "int8").
  When "default", auto-selects: float16 for CUDA, int8 for CPU (including Apple Silicon).
- transcribe.enrich: Enable/disable LLM enrichment (default: true)
- transcribe.preserve_all: Keep audio files even when no speech detected (default: false)
- transcribe.min_speech_seconds: Minimum speech duration to proceed with transcription.
  Files with less speech are filtered early, before loading the STT model. Default: 1.0

Platform optimizations:
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

from observe.transcribe import transcribe as stt_transcribe
from observe.transcribe.utils import is_apple_silicon
from observe.transcribe.whisper import DEFAULT_COMPUTE, DEFAULT_DEVICE, DEFAULT_MODEL
from observe.utils import SAMPLE_RATE, get_segment_key, prepare_audio_file
from observe.vad import (
    AudioReduction,
    VadResult,
    reduce_audio,
    restore_segment_timestamps,
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
    "MIN_SEGMENT_DURATION",
    "main",
]

# Default transcription settings
DEFAULT_MIN_SPEECH_SECONDS = 1.0

# Minimum segment duration for embedding (seconds)
MIN_SEGMENT_DURATION = 0.3

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


def _embed_segments(
    audio: np.ndarray,
    segments: list[dict],
    sample_rate: int,
    whisper_device: str = "cpu",
) -> dict[str, np.ndarray] | None:
    """Generate voice embeddings for each sentence segment.

    Args:
        audio: Audio buffer (float32, mono)
        segments: List of sentence segments
        sample_rate: Sample rate in Hz
        whisper_device: Device used by whisper (affects encoder device selection)

    Returns:
        Dict with embedding data or None on error:
        - embeddings: (N, 256) float32 array
        - segment_ids: (N,) int32 array of segment IDs
    """
    try:
        voice_encoder = _get_voice_encoder(whisper_device)

        # Filter segments by duration
        valid_segments = [
            s for s in segments if s["end"] - s["start"] >= MIN_SEGMENT_DURATION
        ]

        if not valid_segments:
            logging.info("No segments with sufficient duration for embedding")
            return None

        logging.info(f"Embedding {len(valid_segments)} segments...")
        t0 = time.perf_counter()

        embeddings = []
        segment_ids = []
        skipped = 0

        for seg in valid_segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Skip if too short after slicing
            if len(segment_audio) < int(MIN_SEGMENT_DURATION * sample_rate):
                skipped += 1
                continue

            try:
                emb = voice_encoder.embed_utterance(segment_audio)
                embeddings.append(emb)
                segment_ids.append(seg["id"])
            except Exception:
                skipped += 1
                continue

        embed_time = time.perf_counter() - t0

        if not embeddings:
            logging.warning("No embeddings generated")
            return None

        logging.info(
            f"  Embedded {len(embeddings)} segments "
            f"(skipped {skipped}) in {embed_time:.2f}s"
        )

        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "segment_ids": np.array(segment_ids, dtype=np.int32),
        }

    except Exception as e:
        logging.error(f"Embedding failed: {e}", exc_info=True)
        return None


def _segments_to_jsonl(
    segments: list[dict],
    raw_filename: str,
    base_datetime: datetime.datetime,
    model_info: dict,
    source: str | None = None,
    remote: str | None = None,
    enrichment: dict | None = None,
    vad_result: VadResult | None = None,
) -> list[str]:
    """Convert segments to JSONL lines.

    Args:
        segments: List of sentence segments
        raw_filename: Original audio filename for metadata
        base_datetime: Base datetime for timestamp calculation
        model_info: Dict with model, device, compute_type from backend
        source: Optional source label (e.g., "mic", "sys")
        remote: Optional remote name for metadata
        enrichment: Optional enrichment data with topics, setting, and
            per-segment corrected text and descriptions
        vad_result: Optional VAD result for noise detection metadata

    Returns:
        List of JSON strings (metadata line first, then entries)
    """
    # Build metadata line with transcription config
    metadata = {
        "raw": raw_filename,
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

    lines = [json.dumps(metadata)]

    # Get enriched segments list (positional matching)
    enriched_segments = []
    if enrichment and "segments" in enrichment:
        enriched_segments = enrichment["segments"]

    # Build entry lines
    for i, seg in enumerate(segments):
        # Calculate absolute timestamp
        seg_dt = base_datetime + datetime.timedelta(seconds=seg["start"])
        timestamp_str = seg_dt.strftime("%H:%M:%S")

        entry = {
            "start": timestamp_str,
            "text": seg["text"],
        }
        if source:
            entry["source"] = source

        # Pass through speaker ID if present (from diarized backends like Rev.ai)
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]

        # Add corrected text and description from enrichment by position
        if i < len(enriched_segments):
            enriched = enriched_segments[i]
            if isinstance(enriched, dict):
                # Add corrected text only if different from original
                corrected = enriched.get("corrected", "")
                if corrected and corrected != seg["text"]:
                    entry["corrected"] = corrected
                # Add description
                description = enriched.get("description", "")
                if description:
                    entry["description"] = description

        lines.append(json.dumps(entry))

    return lines


def process_audio(
    raw_path: Path,
    vad_result: VadResult,
    backend_config: dict,
    redo: bool = False,
    reduction: AudioReduction | None = None,
    reduced_audio: np.ndarray | None = None,
) -> None:
    """Process a raw audio file with pre-computed VAD.

    This is the main orchestration function that coordinates:
    - STT backend dispatch
    - Enrichment (optional)
    - Embedding generation
    - Output file writing
    - Event emission

    Args:
        raw_path: Path to audio file in segment directory
        vad_result: Pre-computed VAD result from run_vad()
        backend_config: Configuration for STT backend
        redo: If True, skip "already processed" check
        reduction: Optional AudioReduction mapping for timestamp restoration
        reduced_audio: Optional reduced audio buffer (used if reduction provided)
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

    # Prepare audio buffer for processing
    if reduced_audio is not None:
        audio_buffer = reduced_audio
    else:
        audio_buffer = decode_audio(str(raw_path), sampling_rate=SAMPLE_RATE)

    # For enrichment, we need a file path - use temp if we have reduced audio
    import soundfile as sf

    processing_audio_path: Path | None = None
    is_temp_audio = False

    if reduced_audio is not None:
        # Write reduced buffer to temp file for enrichment
        processing_audio_path = raw_path.with_suffix(".reduced.flac")
        audio_int16 = (np.clip(reduced_audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(processing_audio_path, audio_int16, SAMPLE_RATE, format="FLAC")
        is_temp_audio = True
    else:
        # Use original file (convert M4A if needed)
        processing_audio_path = prepare_audio_file(raw_path, SAMPLE_RATE)
        is_temp_audio = processing_audio_path != raw_path

    try:
        # Dispatch to STT backend
        segments = stt_transcribe("whisper", audio_buffer, SAMPLE_RATE, backend_config)

        # Get model info for metadata
        from observe.transcribe.whisper import get_model_info

        model_info = get_model_info(backend_config)

        # Sanity check: if VAD detected speech but we got no segments, something is wrong
        if vad_result.has_speech and not segments:
            raise RuntimeError(
                f"VAD detected {vad_result.speech_duration:.1f}s of speech "
                f"(from {vad_result.duration:.1f}s total) but transcription produced "
                f"0 segments. This indicates a transcription failure, not silence."
            )

        # Load config for preserve_all setting
        config = get_config()
        preserve_all = config.get("transcribe", {}).get("preserve_all", False)

        # Build base event fields (always emitted as observe.transcribed)
        event = _build_base_event(raw_path, vad_result, segment, remote)

        # Handle no speech detected
        if not segments:
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

        # Run enrichment if enabled in config
        enrichment = None
        enrich_enabled = config.get("transcribe", {}).get("enrich", True)
        if enrich_enabled:
            from observe.enrich import enrich_transcript

            enrichment = enrich_transcript(processing_audio_path, segments)

        # Generate embeddings before timestamp restoration
        # Use reduced audio buffer if available for consistent timestamps
        embeddings_data = _embed_segments(
            audio_buffer, segments, SAMPLE_RATE, model_info.get("device", "cpu")
        )

        # Restore original timestamps if audio was reduced
        if reduction:
            segments = restore_segment_timestamps(segments, reduction)
            logging.info(
                f"  Restored timestamps from reduced audio "
                f"({reduction.reduced_duration:.1f}s -> {reduction.original_duration:.1f}s)"
            )

        # Convert to JSONL format (now with original timestamps)
        raw_filename = f"{raw_path.stem}{raw_path.suffix}"
        jsonl_lines = _segments_to_jsonl(
            segments,
            raw_filename,
            base_dt,
            model_info,
            source,
            remote,
            enrichment,
            vad_result,
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
    finally:
        # Clean up temp files
        if is_temp_audio and processing_audio_path and processing_audio_path.exists():
            processing_audio_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using pluggable STT with sentence embeddings"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file in segment directory (.flac or .m4a)",
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
    args = setup_cli(parser)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    # Validate supported formats
    supported_formats = {".flac", ".m4a"}
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

    # Stage 3: Build backend config
    if args.cpu:
        device = "cpu"
        compute_type = "int8"
    else:
        device = transcribe_config.get("device", DEFAULT_DEVICE)
        compute_type = transcribe_config.get("compute_type", DEFAULT_COMPUTE)

    model = args.model or transcribe_config.get("model", DEFAULT_MODEL)

    backend_config = {
        "model": model,
        "device": device,
        "compute_type": compute_type,
    }

    # Stage 4: Process audio with STT backend
    process_audio(
        audio_path,
        vad_result,
        backend_config,
        redo=args.redo,
        reduction=reduction,
        reduced_audio=reduced_audio,
    )


if __name__ == "__main__":
    main()
