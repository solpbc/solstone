# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files with pluggable STT backends and sentence-level embeddings.

Transcription pipeline:
1. VAD stage: Run Silero VAD to detect speech and filter silent files early
2. Audio reduction: Trim long silence gaps for faster processing
3. Transcription: Dispatch to the configured STT backend (default: parakeet)
4. Enrichment: Extract topics, setting, emotions, and warnings via LLM (optional)
5. Embeddings: Generate voice embeddings for each sentence using wespeaker-resnet34
6. Output: JSONL format compatible with format_audio() in observe/hear.py

Output files:
- <stem>.jsonl: Transcript with HH:MM:SS timestamps, topics, setting, emotions
- <stem>.npz: Sentence-level voice embeddings indexed by statement id

Configuration (journal config transcribe section):
- transcribe.backend: STT backend ("parakeet", "whisper", "revai", "gemini"). Default: "parakeet"
- transcribe.enrich: Enable/disable LLM enrichment (default: true)
- transcribe.preserve_all: Keep audio files even when no speech detected (default: false)
- transcribe.min_speech_seconds: Minimum speech duration to proceed. Default: 1.0
- transcribe.noise_upgrade: Auto-switch to Rev.ai for noisy recordings (default: true)
- transcribe.noise_upgrade_min_speech_ratio: Min speech/loud ratio required for noisy upgrade (default: 0.3). Filters out music and other non-speech noise.

Parakeet backend settings (transcribe.parakeet):
- model_version: Parakeet model version ("v2", "v3"). Default: "v3"
- cache_dir: Optional helper cache directory
- timeout_sec: Helper timeout in seconds. Default: 120.0

Whisper backend settings (transcribe.whisper):
- device: Device for inference ("auto", "cpu", "cuda"). Default: "auto"
- model: Whisper model size (e.g., "medium.en"). Default: "medium.en"
- compute_type: Precision ("default", "float32", "float16", "int8"). Default: "default"
  Auto-selects: float16 for CUDA, int8 for CPU (including Apple Silicon).
- Whisper remains available as the rollback/local alternative backend.

Rev.ai backend settings (transcribe.revai):
- model: Rev.ai transcriber ("fusion", "machine", "low_cost"). Default: "fusion"
- Automatically loads recent entity names as custom vocabulary for improved recognition

Gemini backend settings (transcribe.gemini):
- No configuration needed (model resolved by think.models context system)
- Includes speaker diarization

Platform optimizations (Whisper):
- CUDA GPU: Uses float16 for GPU-optimized inference
- Apple hosts: Uses int8 for Whisper on CPU and CoreML/CPU for embeddings
- Other CPU: Uses int8 for best performance
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from observe.transcribe import (
    BACKEND_REGISTRY,
    get_backend,
)
from observe.transcribe import transcribe as stt_transcribe
from observe.transcribe.whisper import DEFAULT_COMPUTE, DEFAULT_DEVICE, DEFAULT_MODEL
from observe.utils import SAMPLE_RATE, get_segment_key, load_audio
from observe.vad import (
    AudioReduction,
    VadResult,
    reduce_audio,
    restore_statement_timestamps,
    run_vad,
)
from think.callosum import callosum_send
from think.media import AUDIO_EXTENSIONS as SUPPORTED_AUDIO_FORMATS
from think.utils import (
    day_dirs,
    day_from_path,
    get_config,
    get_journal,
    iter_segments,
    journal_relative_path,
    require_solstone,
    resolve_journal_path,
    setup_cli,
)

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
DEFAULT_BACKEND = "parakeet"
DEFAULT_MIN_SPEECH_SECONDS = 1.0

# Minimum statement duration for embedding (seconds)
MIN_STATEMENT_DURATION = 0.3

# WeSpeaker embedder asset
EMBEDDER_NAME = "wespeaker-resnet34-256"
WESPEAKER_MODEL_SHA256 = (
    "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94"
)
WESPEAKER_MODEL_PATH = Path(__file__).parent / "assets" / "wespeaker-resnet34-256.onnx"

# Number of recent entity names to load for transcription context
ENTITY_NAMES_LIMIT = 40

# Module-level embedder cache
_embedder_session: ort.InferenceSession | None = None


def _select_onnx_providers() -> list[str]:
    """Return the ONNX Runtime provider list for this host.

    Darwin (any arch) prefers CoreML with CPU fallback; elsewhere, CPU only.
    """
    if platform.system() == "Darwin":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_embedder_session() -> ort.InferenceSession:
    """Return a cached ONNX InferenceSession for the WeSpeaker encoder."""
    global _embedder_session

    if _embedder_session is None:
        if not WESPEAKER_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"WeSpeaker model asset not found at {WESPEAKER_MODEL_PATH}. "
                "Run `make install` to verify the bundled asset."
            )
        providers = _select_onnx_providers()
        start = time.monotonic()
        _embedder_session = ort.InferenceSession(
            str(WESPEAKER_MODEL_PATH),
            providers=providers,
        )
        elapsed = time.monotonic() - start
        logging.info(
            "wespeaker session loaded providers=%s elapsed=%.2fs",
            _embedder_session.get_providers(),
            elapsed,
        )

    return _embedder_session


def _compute_wespeaker_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute Kaldi-style fbank features for the bundled WeSpeaker encoder."""
    import kaldi_native_fbank as knf

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"WeSpeaker embedder requires {SAMPLE_RATE} Hz audio, got {sample_rate}"
        )

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0
    opts.use_energy = False

    fbank = knf.OnlineFbank(opts)
    scaled = (audio.astype(np.float32) * 32768.0).tolist()
    fbank.accept_waveform(float(sample_rate), scaled)
    fbank.input_finished()

    frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    if not frames:
        return np.zeros((0, 80), dtype=np.float32)

    feats = np.stack(frames, axis=0).astype(np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)
    return feats


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
    observer: str | None = None,
) -> dict:
    """Build base event dict for callosum emission.

    Args:
        audio_path: Path to the audio file
        vad_result: VAD result with speech detection info
        segment: Optional segment key (e.g., "143022_300")
        observer: Optional observer name

    Returns:
        Event dict with common fields for observe.transcribed events
    """
    journal_path = Path(get_journal())
    day = day_from_path(audio_path)

    try:
        rel_input = journal_relative_path(journal_path, audio_path)
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
    if vad_result.loud_windows > 0:
        event["loud_windows"] = vad_result.loud_windows
        event["speech_loud_windows"] = vad_result.speech_loud_windows
        ratio = vad_result.loud_speech_ratio
        if ratio is not None:
            event["loud_speech_ratio"] = round(ratio, 2)

    if day:
        event["day"] = day
    if segment:
        event["segment"] = segment
    if observer:
        event["observer"] = observer

    return event


def _embed_statements(
    audio: np.ndarray,
    statements: list[dict],
    sample_rate: int,
) -> dict[str, np.ndarray] | None:
    """Generate voice embeddings for each statement.

    Args:
        audio: Audio buffer (float32, mono)
        statements: List of statements
        sample_rate: Sample rate in Hz

    Returns:
        Dict with embedding data or None on error:
        - embeddings: (N, 256) float32 array
        - statement_ids: (N,) int32 array of statement IDs
        - encoder: 0-d array naming the embedder
    """
    try:
        session = _get_embedder_session()
        audio_duration = len(audio) / sample_rate
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

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
                feats = _compute_wespeaker_features(stmt_audio, sample_rate)
                if feats.shape[0] == 0:
                    skipped += 1
                    continue
                emb = session.run([output_name], {input_name: feats[None, :, :]})[0]
                embeddings.append(emb[0].astype(np.float32))
                statement_ids.append(stmt["id"])
            except Exception:
                logging.exception(
                    "wespeaker embedding failed for statement %s", stmt["id"]
                )
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
            "embeddings": np.stack(embeddings, axis=0).astype(np.float32),
            "statement_ids": np.asarray(statement_ids, dtype=np.int32),
            "encoder": np.array(EMBEDDER_NAME),
        }

    except Exception:
        logging.exception("failed to load WeSpeaker embedder")
        return None


def _statements_to_jsonl(
    statements: list[dict],
    raw_filename: str,
    base_datetime: datetime.datetime,
    model_info: dict,
    source: str | None = None,
    observer: str | None = None,
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
        observer: Optional observer name for metadata
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
    if observer:
        metadata["observer"] = observer

    # Add noise detection metadata if available
    if vad_result:
        metadata["noisy"] = vad_result.is_noisy()
        if vad_result.noisy_rms is not None:
            metadata["noisy_rms"] = round(vad_result.noisy_rms, 4)
            metadata["noisy_s"] = round(vad_result.noisy_s, 1)
        if vad_result.loud_windows > 0:
            metadata["loud_windows"] = vad_result.loud_windows
            metadata["speech_loud_windows"] = vad_result.speech_loud_windows
            ratio = vad_result.loud_speech_ratio
            if ratio is not None:
                metadata["loud_speech_ratio"] = round(ratio, 2)

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
    audio_buffer: np.ndarray,
    vad_result: VadResult,
    backend_config: dict,
    redo: bool = False,
    reduction: AudioReduction | None = None,
    reduced_audio: np.ndarray | None = None,
    backend: str | None = None,
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
        audio_buffer: Full audio waveform (float32 mono at SAMPLE_RATE)
        vad_result: Pre-computed VAD result from run_vad()
        backend_config: Configuration for STT backend
        redo: If True, skip "already processed" check
        reduction: Optional AudioReduction mapping for timestamp restoration
        reduced_audio: Optional reduced audio buffer (used if reduction provided)
        backend: STT backend name. If omitted, uses DEFAULT_BACKEND.
        entity_names: Optional list of entity names for STT and enrichment context
    """
    start_time = time.time()
    resolved_backend = backend or DEFAULT_BACKEND

    # Derive segment from path
    segment = get_segment_key(raw_path)

    # Skip if already processed (unless redo mode)
    jsonl_path = _get_jsonl_path(raw_path)
    if not redo and jsonl_path.exists():
        logging.info(f"Already processed: {raw_path}")
        return

    # Get observer name once for use in metadata and events
    observer = os.getenv("OBSERVER_NAME")

    # Get segment metadata (from sense.py via SEGMENT_META env var)
    segment_meta = None
    segment_meta_str = os.getenv("SEGMENT_META")
    if segment_meta_str:
        try:
            segment_meta = json.loads(segment_meta_str)
        except json.JSONDecodeError:
            logging.warning(f"Invalid SEGMENT_META JSON: {segment_meta_str[:100]}")

    # Gemini uses chunk-based transcription with VAD segments for timestamp accuracy
    # Other backends use reduced audio with post-hoc timestamp restoration
    use_gemini_chunks = backend == "gemini" and vad_result.speech_segments

    if use_gemini_chunks:
        # Gemini: use full audio buffer with VAD segments for chunking
        stt_buffer = audio_buffer
    elif reduced_audio is not None:
        # Other backends: use reduced audio
        stt_buffer = reduced_audio
    else:
        stt_buffer = audio_buffer

    try:
        # Dispatch to STT backend
        if use_gemini_chunks:
            # Pass VAD segments to Gemini for chunk-based transcription
            statements = stt_transcribe(
                resolved_backend,
                stt_buffer,
                SAMPLE_RATE,
                backend_config,
                speech_segments=vad_result.speech_segments,
            )
        else:
            statements = stt_transcribe(
                resolved_backend, stt_buffer, SAMPLE_RATE, backend_config
            )

        # Get model info for metadata (dynamic import based on backend)
        backend_module = get_backend(resolved_backend)
        model_info = backend_module.get_model_info(backend_config)

        # Load config for preserve_all setting
        config = get_config()
        preserve_all = config.get("transcribe", {}).get("preserve_all", False)

        # Build base event fields (always emitted as observe.transcribed)
        event = _build_base_event(raw_path, vad_result, segment, observer)

        # Handle no speech detected
        if not statements:
            logging.info(
                "STT backend returned 0 statements, treating as silence "
                "(VAD: %.1fs speech of %.1fs)",
                vad_result.speech_duration,
                vad_result.duration,
            )
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
        day = day_from_path(raw_path)
        time_part = segment.split("_")[0] if segment else "000000"
        if day is None:
            logging.error(f"Could not extract day from path: {raw_path}")
            time_obj = datetime.datetime.strptime(time_part, "%H%M%S").time()
            base_dt = datetime.datetime.combine(datetime.date.today(), time_obj)
        else:
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
                stt_buffer, SAMPLE_RATE, statements, entity_names=entity_names
            )

        # Generate embeddings before timestamp restoration
        # Use reduced audio buffer if available for consistent timestamps
        embeddings_data = _embed_statements(stt_buffer, statements, SAMPLE_RATE)

        # Restore original timestamps if audio was reduced (non-Gemini backends only)
        # Gemini with chunks already has timestamps in original audio time
        if reduction and not use_gemini_chunks:
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
            observer,
            enrichment,
            vad_result,
            segment_meta,
            resolved_backend,
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
            rel_output = journal_relative_path(journal_path, jsonl_path)
        except ValueError:
            rel_output = jsonl_path
        event["output"] = rel_output

        callosum_send("observe", "transcribed", **event)

    except Exception as e:
        logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
        from think.models import IncompleteJSONError

        if isinstance(e, IncompleteJSONError) and e.partial_text:
            text = e.partial_text
            logging.error(f"Partial response ({len(text)} chars) HEAD: {text[:1000]}")
            logging.error(f"Partial response TAIL: {text[-1000:]}")
        raise SystemExit(1) from e


def _process_one(
    audio_path: Path,
    args: argparse.Namespace,
    transcribe_config: dict,
    entity_names: list[str],
) -> None:
    """Run the full transcription pipeline for a single audio file."""
    min_speech_seconds = transcribe_config.get(
        "min_speech_seconds", DEFAULT_MIN_SPEECH_SECONDS
    )
    preserve_all = transcribe_config.get("preserve_all", False)

    logging.info(f"Processing audio: {audio_path}")

    # Load audio once - handles M4A multi-stream mixing
    audio_buffer = load_audio(audio_path)

    # Stage 1: Run VAD to detect speech (lightweight, before loading STT model)
    vad_result = run_vad(audio_buffer, min_speech_seconds=min_speech_seconds)

    # Early exit if no speech detected (skip loading heavy STT model)
    if not vad_result.has_speech:
        observer = os.getenv("OBSERVER_NAME")
        segment = get_segment_key(audio_path)
        event = _build_base_event(audio_path, vad_result, segment, observer)

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
    # Skip reduction for noisy clips with >70% speech — the "silence" gaps are
    # mostly noise and VAD boundaries are less reliable, so process the full audio.
    if vad_result.is_noisy() and vad_result.speech_ratio >= 0.7:
        logging.info(
            f"  Skipping audio reduction: noisy clip with "
            f"{vad_result.speech_ratio:.0%} speech"
        )
        reduced_audio, reduction = None, None
    else:
        reduced_audio, reduction = reduce_audio(audio_buffer, vad_result)

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
    min_ratio = transcribe_config.get("noise_upgrade_min_speech_ratio", 0.3)
    if (
        not args.backend
        and noise_upgrade
        and backend != "revai"
        and vad_result.is_noisy()
    ):
        from observe.transcribe.revai import has_token

        ratio = vad_result.loud_speech_ratio
        if ratio is not None and ratio < min_ratio:
            logging.info(
                "Noisy audio (RMS=%.4f) looks like non-speech (loud_speech_ratio=%.2f < %.2f), "
                "skipping Rev.ai upgrade",
                vad_result.noisy_rms,
                ratio,
                min_ratio,
            )
        elif has_token():
            logging.info(
                "Noisy audio detected (RMS=%.4f, loud_speech_ratio=%s), upgrading to Rev.ai backend",
                vad_result.noisy_rms,
                f"{ratio:.2f}" if ratio is not None else "n/a",
            )
            backend = "revai"

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
    elif backend == "parakeet":
        parakeet_config = transcribe_config.get("parakeet", {})
        backend_config = {
            k: v
            for k, v in parakeet_config.items()
            if k in ("model_version", "cache_dir", "timeout_sec", "device", "precision")
        }
    elif backend == "gemini":
        # Gemini backend - model resolved by think.models based on context
        # Entity names handled by enrich step, not passed to transcription
        backend_config = {}
    else:
        # Unknown backend - let get_backend() raise the error
        backend_config = {}

    # Stage 4: Process audio with STT backend
    process_audio(
        audio_path,
        audio_buffer,
        vad_result,
        backend_config,
        redo=args.redo,
        reduction=reduction,
        reduced_audio=reduced_audio,
        backend=backend,
        entity_names=entity_names,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using pluggable STT with sentence embeddings"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        type=str,
        help="Path to audio file in journal segment directory, e.g. HHMMSS_LEN/audio.flac",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all",
        help="Batch-transcribe all unprocessed audio segments in the journal",
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
    require_solstone()

    if args.all and args.audio_path:
        parser.error("--all and audio_path are mutually exclusive")
    if not args.all and not args.audio_path:
        parser.error("provide audio_path or --all")

    config = get_config()
    transcribe_config = config.get("transcribe", {})

    from think.entities import load_recent_entity_names

    entity_names = load_recent_entity_names(limit=ENTITY_NAMES_LIMIT)
    if entity_names:
        logging.info(f"Loaded {len(entity_names)} entities for transcription context")

    if args.all:
        processed = 0
        skipped = 0
        failed = 0

        for day_name, _day_path_str in sorted(day_dirs().items()):
            for _stream_name, _seg_key, seg_path in iter_segments(day_name):
                for audio_file in sorted(seg_path.iterdir()):
                    if audio_file.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
                        continue
                    jsonl_path = audio_file.with_suffix(".jsonl")
                    if jsonl_path.exists() and not args.redo:
                        logging.info(f"Skipping (already transcribed): {audio_file}")
                        skipped += 1
                        continue
                    try:
                        logging.info(f"Transcribing: {audio_file}")
                        _process_one(audio_file, args, transcribe_config, entity_names)
                        processed += 1
                    except Exception:
                        logging.error(
                            f"Failed to transcribe {audio_file}", exc_info=True
                        )
                        failed += 1

        summary = f"{processed} processed, {skipped} skipped (already transcribed)"
        if failed:
            summary += f", {failed} failed"
        print(summary)
        return

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        if audio_path.is_absolute():
            journal_relative = Path(get_journal()) / audio_path.as_posix().lstrip("/")
        else:
            journal_relative = resolve_journal_path(get_journal(), args.audio_path)
        if journal_relative.exists():
            audio_path = journal_relative
        else:
            parser.error(
                f"Audio file not found.\n"
                f"  Tried absolute:         {audio_path}\n"
                f"  Tried journal-relative: {journal_relative}"
            )

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        parser.error(
            f"Unsupported audio format: {audio_path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )

    segment = get_segment_key(audio_path)
    if segment is None:
        parser.error(
            f"Audio file must be in a segment directory (HHMMSS_LEN/), "
            f"but parent is: {audio_path.parent.name}"
        )

    _process_one(audio_path, args, transcribe_config, entity_names)


if __name__ == "__main__":
    main()
