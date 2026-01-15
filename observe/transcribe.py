# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files using faster-whisper with sentence-level embeddings.

Transcription pipeline:
1. VAD stage: Run Silero VAD to detect speech and filter silent files early
2. Transcription: Transcribe speech segments using faster-whisper with word timestamps
3. Sentence re-segmentation: Re-segment by punctuation boundaries (not acoustic pauses)
4. Enrichment: Extract topics, setting, and audio descriptions via Gemini Lite (optional)
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
- transcribe.enrich: Enable/disable Gemini Lite enrichment (default: true)
- transcribe.preserve_all: Keep audio files even when no speech detected (default: false)
- transcribe.min_speech_seconds: Minimum speech duration to proceed with transcription.
  Files with less speech are filtered early, before loading the Whisper model. Default: 1.0

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
import platform
import time
from pathlib import Path

import librosa
import numpy as np

from observe.utils import (
    SAMPLE_RATE,
    get_segment_key,
    prepare_audio_file,
)
from observe.vad import (
    AudioReduction,
    VadResult,
    reduce_audio,
    restore_segment_timestamps,
    run_vad,
)
from think.callosum import callosum_send
from think.utils import get_config, get_journal, setup_cli

# Default transcription settings
DEFAULT_MODEL = "medium.en"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE = "default"
DEFAULT_MIN_SPEECH_SECONDS = 1.0

# Style prompt to establish punctuation pattern for Whisper
# Whisper is autoregressive and can get stuck in "no-punctuation mode" without this
STYLE_PROMPT = "Okay, let's get started. Here's what we've been working on."

# Minimum segment duration for embedding (seconds)
MIN_SEGMENT_DURATION = 0.3

# Sentence-ending punctuation marks
SENTENCE_ENDINGS = frozenset(".?!")


def _is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_cuda() -> bool:
    """Check if CUDA is available via CTranslate2."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _get_optimal_compute_type(device: str) -> str:
    """Get optimal compute type for the current platform.

    When compute_type is "default", CTranslate2 auto-selects but makes suboptimal
    choices on some platforms. This function provides better defaults:

    - CUDA GPU: float16 for GPU-optimized inference
    - CPU (including Apple Silicon): int8 for ~2x faster inference and faster model load

    Args:
        device: The device being used ("cpu", "cuda", "auto")

    Returns:
        Optimal compute type string
    """
    # If CUDA is explicitly requested or auto-detected, float16 is optimal
    if device == "cuda" or (device == "auto" and _has_cuda()):
        return "float16"

    # For CPU (including Apple Silicon), int8 is fastest
    # This provides ~2x speedup and 76x faster model loading
    return "int8"


def _get_optimal_encoder_device() -> str:
    """Get optimal device for VoiceEncoder (resemblyzer).

    On Apple Silicon, MPS provides ~16x speedup over CPU for embeddings.

    Returns:
        Device string: "mps" on Apple Silicon with MPS available, "cpu" otherwise
    """
    if _is_apple_silicon():
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    return "cpu"


def resegment_by_sentences(segments: list[dict]) -> list[dict]:
    """Re-segment transcript by sentence boundaries instead of acoustic pauses.

    Whisper segments on speech pauses (VAD-driven), but we want segments aligned
    to sentence boundaries (punctuation-driven). This function flattens all words
    and re-segments based on sentence-ending punctuation.

    Args:
        segments: List of segment dicts with 'words' containing word-level data

    Returns:
        New list of segments aligned to sentence boundaries
    """
    # Flatten all words from all segments
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return segments

    # Build new segments based on sentence-ending punctuation
    new_segments = []
    current_words = []

    for word in all_words:
        current_words.append(word)

        # Check if word ends with sentence-ending punctuation
        word_text = word.get("word", "").strip()
        if word_text and word_text[-1] in SENTENCE_ENDINGS:
            # Complete this segment
            new_segments.append(_build_segment(len(new_segments) + 1, current_words))
            current_words = []

    # Handle any remaining words (incomplete final sentence)
    if current_words:
        new_segments.append(_build_segment(len(new_segments) + 1, current_words))

    return new_segments


def _build_segment(segment_id: int, words: list[dict]) -> dict:
    """Build a segment dict from a list of words.

    Args:
        segment_id: Sequential segment ID
        words: List of word dicts with 'word', 'start', 'end', 'probability'

    Returns:
        Segment dict with id, start, end, text, words
    """
    # Join words - Whisper includes leading spaces in word text
    text = "".join(w.get("word", "") for w in words).strip()

    return {
        "id": segment_id,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


class Transcriber:
    """Transcribes audio using faster-whisper and generates sentence embeddings."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE,
    ):
        """Initialize transcriber with models.

        Args:
            model_size: faster-whisper model size (e.g., "medium.en", "small.en")
            device: Device for inference ("auto", "cpu", "cuda")
            compute_type: Compute type ("float32", "float16", "int8", "default")
        """
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder

        # Resolve "default" compute_type to platform-optimal setting
        # CTranslate2's auto-selection falls back to float32 on CPU, but int8 is faster
        if compute_type == "default":
            compute_type = _get_optimal_compute_type(device)
            logging.info(
                f"Auto-selected compute_type={compute_type} for device={device}"
            )

        logging.info(f"Loading faster-whisper model ({model_size})...")
        t0 = time.perf_counter()
        self.whisper_model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        whisper_actual_device = self.whisper_model.model.device
        whisper_actual_compute = self.whisper_model.model.compute_type
        logging.info(
            f"  Whisper loaded in {time.perf_counter() - t0:.2f}s "
            f"(device={whisper_actual_device}, compute={whisper_actual_compute})"
        )

        # VoiceEncoder: use MPS on Apple Silicon for ~16x speedup, otherwise CPU
        # (CUDA auto-detection handled by resemblyzer when device=None)
        if whisper_actual_device == "cuda":
            encoder_device = None  # Let resemblyzer auto-detect CUDA
        else:
            encoder_device = _get_optimal_encoder_device()

        logging.info("Loading resemblyzer VoiceEncoder...")
        t0 = time.perf_counter()
        self.voice_encoder = VoiceEncoder(device=encoder_device)
        logging.info(
            f"  VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s "
            f"(device={self.voice_encoder.device})"
        )

        # Store configuration for metadata
        self.model_size = model_size
        self.device = str(whisper_actual_device)
        self.compute_type = str(whisper_actual_compute)

    def _get_jsonl_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSONL path.

        E.g., YYYYMMDD/HHMMSS_LEN/audio.flac -> YYYYMMDD/HHMMSS_LEN/audio.jsonl
        """
        return audio_path.with_suffix(".jsonl")

    def _get_embeddings_path(self, audio_path: Path) -> Path:
        """Generate the corresponding embeddings path.

        E.g., YYYYMMDD/HHMMSS_LEN/audio.flac -> YYYYMMDD/HHMMSS_LEN/audio.npz
        """
        return audio_path.with_suffix(".npz")

    def _transcribe(
        self,
        audio_path: Path,
        vad_result: VadResult,
        initial_prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe audio using faster-whisper with VAD filtering.

        Args:
            audio_path: Path to audio file (FLAC)
            vad_result: Pre-computed VAD result (used for sanity check)
            initial_prompt: Optional prompt with entity names for context

        Returns:
            List of sentence-aligned segments with word-level data

        Raises:
            RuntimeError: If VAD detected speech but transcription produced no segments
        """
        logging.info(f"Transcribing {audio_path.name}...")
        t0 = time.perf_counter()

        # Build transcribe kwargs - let faster-whisper run its own VAD
        transcribe_kwargs = {
            "language": "en",
            "vad_filter": True,
            "beam_size": 5,
            "word_timestamps": True,
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        segments_gen, info = self.whisper_model.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )

        # Consume generator and build output
        segment_list = []
        for seg in segments_gen:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                    )

            segment_list.append(
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": words,
                }
            )

        transcribe_time = time.perf_counter() - t0

        # Get duration from last segment or 0
        duration = max((s["end"] for s in segment_list), default=0)

        # Sanity check: if VAD detected speech but we got no segments, something is wrong
        # This protects against silent transcription failures (e.g., CUDA errors)
        if vad_result.has_speech and not segment_list:
            raise RuntimeError(
                f"VAD detected {vad_result.speech_duration:.1f}s of speech "
                f"(from {vad_result.duration:.1f}s total) but transcription produced "
                f"0 segments. This indicates a transcription failure, not silence."
            )

        # Log transcription stats
        logging.info(
            f"  Transcribed {len(segment_list)} segments, "
            f"{duration:.1f}s speech in {transcribe_time:.2f}s "
            f"(RTF: {transcribe_time / max(duration, 0.1):.3f}x)"
        )

        # Re-segment by sentence boundaries instead of acoustic pauses
        whisper_segments = len(segment_list)
        sentence_segments = resegment_by_sentences(segment_list)
        logging.info(
            f"  Re-segmented {whisper_segments} acoustic segments "
            f"to {len(sentence_segments)} sentences"
        )

        return sentence_segments

    def _embed_segments(
        self, audio_path: Path, segments: list[dict]
    ) -> dict[str, np.ndarray] | None:
        """Generate voice embeddings for each sentence segment.

        Args:
            audio_path: Path to audio file
            segments: List of sentence segments from _transcribe()

        Returns:
            Dict with embedding data or None on error:
            - embeddings: (N, 256) float32 array
            - segment_ids: (N,) int32 array of segment IDs
        """
        try:
            # Load audio
            logging.info("Loading audio for embeddings...")
            t0 = time.perf_counter()
            wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            logging.info(f"  Audio loaded in {time.perf_counter() - t0:.2f}s")

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
                start_sample = int(seg["start"] * SAMPLE_RATE)
                end_sample = int(seg["end"] * SAMPLE_RATE)
                segment_audio = wav[start_sample:end_sample]

                # Skip if too short after slicing
                if len(segment_audio) < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
                    skipped += 1
                    continue

                try:
                    emb = self.voice_encoder.embed_utterance(segment_audio)
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
            logging.error(f"Embedding failed for {audio_path}: {e}", exc_info=True)
            return None

    def _segments_to_jsonl(
        self,
        segments: list[dict],
        raw_filename: str,
        base_datetime: datetime.datetime,
        source: str | None = None,
        remote: str | None = None,
        enrichment: dict | None = None,
    ) -> list[str]:
        """Convert segments to JSONL lines.

        Args:
            segments: List of sentence segments
            raw_filename: Original audio filename for metadata
            base_datetime: Base datetime for timestamp calculation
            source: Optional source label (e.g., "mic", "sys")
            remote: Optional remote name for metadata
            enrichment: Optional enrichment data with topics, setting, and
                per-segment corrected text and descriptions

        Returns:
            List of JSON strings (metadata line first, then entries)
        """
        # Build metadata line with transcription config
        metadata = {
            "raw": raw_filename,
            "model": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if remote:
            metadata["remote"] = remote

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

    def _handle_raw(
        self,
        raw_path: Path,
        vad_result: VadResult,
        redo: bool = False,
        reduction: AudioReduction | None = None,
        reduced_audio: np.ndarray | None = None,
    ) -> None:
        """Process a raw audio file with pre-computed VAD.

        Args:
            raw_path: Path to audio file in segment directory
            vad_result: Pre-computed VAD result from run_vad()
            redo: If True, skip "already processed" check
            reduction: Optional AudioReduction mapping for timestamp restoration
            reduced_audio: Optional reduced audio buffer (used if reduction provided)
        """
        import soundfile as sf

        start_time = time.time()

        # Derive segment from path
        segment = get_segment_key(raw_path)

        # Skip if already processed (unless redo mode)
        jsonl_path = self._get_jsonl_path(raw_path)
        if not redo and jsonl_path.exists():
            logging.info(f"Already processed: {raw_path}")
            return

        # Prepare audio for processing
        # If we have a reduced audio buffer, write it to temp file (avoids redundant M4A decode)
        # Otherwise, convert M4A if needed
        reduced_temp_path: Path | None = None
        if reduced_audio is not None:
            # Write reduced buffer to temp file for downstream operations
            reduced_temp_path = raw_path.with_suffix(".reduced.flac")
            audio_int16 = (np.clip(reduced_audio, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(reduced_temp_path, audio_int16, SAMPLE_RATE, format="FLAC")
            processing_audio = reduced_temp_path
            audio_path = raw_path  # Keep reference for cleanup logic
            is_temp = False  # No M4A conversion temp file
        else:
            audio_path = prepare_audio_file(raw_path, SAMPLE_RATE)
            is_temp = audio_path != raw_path
            processing_audio = audio_path

        # Get remote name once for use in metadata and events
        remote = os.getenv("REMOTE_NAME")

        try:
            # Transcribe with faster-whisper using pre-computed VAD
            segments = self._transcribe(processing_audio, vad_result, STYLE_PROMPT)

            # Load config for preserve_all setting
            config = get_config()
            preserve_all = config.get("transcribe", {}).get("preserve_all", False)

            # Build base event fields (always emitted as observe.transcribed)
            journal_path = Path(get_journal())
            day = raw_path.parent.parent.name
            try:
                rel_input = raw_path.relative_to(journal_path)
            except ValueError:
                rel_input = raw_path

            event = {
                "input": str(rel_input),
                "vad_duration": round(vad_result.duration, 1),
                "vad_speech": round(vad_result.speech_duration, 1),
            }
            if day:
                event["day"] = day
            if segment:
                event["segment"] = segment
            if remote:
                event["remote"] = remote

            # Handle no speech detected - safe to skip/delete since _transcribe() already
            # validated that VAD also detected minimal speech (raises RuntimeError otherwise)
            if not segments:
                # Determine outcome based on preserve_all config
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

            # Extract date and time from path structure (day already set above)
            # Files are always in segment directories: YYYYMMDD/HHMMSS_LEN/audio.flac
            time_part = segment.split("_")[0] if segment else "000000"
            base_dt = datetime.datetime.strptime(f"{day}_{time_part}", "%Y%m%d_%H%M%S")

            # Extract source from <source>_audio pattern
            # mic_audio -> "mic", sys_audio -> "sys", etc.
            source = None
            suffix = raw_path.stem
            if suffix.endswith("_audio") and suffix != "audio":
                source = suffix[:-6]  # Remove "_audio" suffix

            # Run enrichment if enabled in config (config already loaded above)
            # Use processing_audio (reduced if available) for consistent timestamps
            enrichment = None
            enrich_enabled = config.get("transcribe", {}).get("enrich", True)
            if enrich_enabled:
                from observe.enrich import enrich_transcript

                enrichment = enrich_transcript(processing_audio, segments)

            # Generate embeddings before timestamp restoration
            # Use processing_audio (reduced if available) for consistent timestamps
            embeddings_data = self._embed_segments(processing_audio, segments)

            # Restore original timestamps if audio was reduced
            if reduction:
                segments = restore_segment_timestamps(segments, reduction)
                logging.info(
                    f"  Restored timestamps from reduced audio "
                    f"({reduction.reduced_duration:.1f}s -> {reduction.original_duration:.1f}s)"
                )

            # Convert to JSONL format (now with original timestamps)
            raw_filename = f"{raw_path.stem}{raw_path.suffix}"
            jsonl_lines = self._segments_to_jsonl(
                segments, raw_filename, base_dt, source, remote, enrichment
            )

            # Write JSONL
            jsonl_path.write_text("\n".join(jsonl_lines) + "\n")
            logging.info(f"Transcribed {raw_path} -> {jsonl_path}")

            # Save embeddings
            if embeddings_data:
                embeddings_path = self._get_embeddings_path(raw_path)
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
            if is_temp and audio_path.exists():
                audio_path.unlink()
            if reduced_temp_path and reduced_temp_path.exists():
                reduced_temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper with sentence embeddings"
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

    # Stage 1: Run VAD to detect speech (lightweight, before loading Whisper)
    vad_result = run_vad(audio_path, min_speech_seconds=min_speech_seconds)

    # Early exit if no speech detected (skip loading heavy Whisper model)
    if not vad_result.has_speech:
        # Build event for early filtering
        journal_path = Path(get_journal())
        day = audio_path.parent.parent.name
        try:
            rel_input = audio_path.relative_to(journal_path)
        except ValueError:
            rel_input = audio_path

        remote = os.getenv("REMOTE_NAME")
        event = {
            "input": str(rel_input),
            "vad_duration": round(vad_result.duration, 1),
            "vad_speech": round(vad_result.speech_duration, 1),
        }
        if day:
            event["day"] = day
        if segment:
            event["segment"] = segment
        if remote:
            event["remote"] = remote

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
    # This optimization works with any STT backend
    reduced_audio, reduction = reduce_audio(audio_path, vad_result)

    # Stage 3: Load Whisper and transcribe (only if speech detected)
    if args.cpu:
        device = "cpu"
        compute_type = "int8"
    else:
        device = transcribe_config.get("device", DEFAULT_DEVICE)
        compute_type = transcribe_config.get("compute_type", DEFAULT_COMPUTE)

    model = args.model or transcribe_config.get("model", DEFAULT_MODEL)

    transcriber = Transcriber(
        model_size=model, device=device, compute_type=compute_type
    )
    transcriber._handle_raw(
        audio_path,
        vad_result,
        redo=args.redo,
        reduction=reduction,
        reduced_audio=reduced_audio,
    )


if __name__ == "__main__":
    main()
