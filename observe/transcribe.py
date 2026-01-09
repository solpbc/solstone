# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcribe audio files using faster-whisper with sentence-level embeddings.

Transcription pipeline:
- Transcribes audio using faster-whisper with word timestamps
- Re-segments by sentence boundaries (not acoustic pauses)
- Generates voice embeddings for each sentence using resemblyzer
- Outputs JSONL format compatible with format_audio() in observe/hear.py

Output files:
- <stem>.jsonl: Transcript with HH:MM:SS timestamps
- <stem>.npz: Sentence-level voice embeddings indexed by segment id
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np

from observe.utils import (
    SAMPLE_RATE,
    get_segment_key,
    prepare_audio_file,
)
from think.callosum import callosum_send
from think.entities import load_entity_names
from think.utils import get_config, get_journal, setup_cli

# Default transcription settings
DEFAULT_MODEL = "medium.en"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE = "default"

# Minimum segment duration for embedding (seconds)
MIN_SEGMENT_DURATION = 0.3

# Sentence-ending punctuation marks
SENTENCE_ENDINGS = frozenset(".?!")


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


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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

        # VoiceEncoder follows whisper device: None auto-detects CUDA, "cpu" forces CPU
        encoder_device = None if device in ("auto", "cuda") else "cpu"

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

        logging.info("Loading resemblyzer VoiceEncoder...")
        t0 = time.perf_counter()
        self.voice_encoder = VoiceEncoder(device=encoder_device)
        logging.info(
            f"  VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s "
            f"(device={self.voice_encoder.device})"
        )

        self.model_size = model_size

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

    def _transcribe(self, audio_path: Path, initial_prompt: str | None) -> list[dict]:
        """Transcribe audio using faster-whisper.

        Args:
            audio_path: Path to audio file (FLAC)
            initial_prompt: Optional prompt with entity names for context

        Returns:
            List of sentence-aligned segments with word-level data

        Raises:
            RuntimeError: If VAD detected speech but transcription produced no segments
        """
        logging.info(f"Transcribing {audio_path.name}...")
        t0 = time.perf_counter()

        # Build transcribe kwargs
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
        vad_detected_speech = (
            info.duration_after_vad > 1.0
        )  # More than 1 second of speech
        if vad_detected_speech and not segment_list:
            raise RuntimeError(
                f"VAD detected {info.duration_after_vad:.1f}s of speech "
                f"(from {info.duration:.1f}s total) but transcription produced 0 segments. "
                f"This indicates a transcription failure, not silence."
            )

        # Log transcription stats including VAD info
        vad_removed = info.duration - info.duration_after_vad
        logging.info(
            f"  Transcribed {len(segment_list)} segments, "
            f"{duration:.1f}s speech in {transcribe_time:.2f}s "
            f"(VAD: {info.duration:.1f}s -> {info.duration_after_vad:.1f}s, "
            f"removed {vad_removed:.1f}s, RTF: {transcribe_time / max(duration, 0.1):.3f}x)"
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
    ) -> list[str]:
        """Convert segments to JSONL lines.

        Args:
            segments: List of sentence segments
            raw_filename: Original audio filename for metadata
            base_datetime: Base datetime for timestamp calculation
            source: Optional source label (e.g., "mic", "sys")
            remote: Optional remote name for metadata

        Returns:
            List of JSON strings (metadata line first, then entries)
        """
        # Build metadata line
        metadata = {"raw": raw_filename}
        if remote:
            metadata["remote"] = remote

        lines = [json.dumps(metadata)]

        # Build entry lines
        for seg in segments:
            # Calculate absolute timestamp
            seg_dt = base_datetime + datetime.timedelta(seconds=seg["start"])
            timestamp_str = seg_dt.strftime("%H:%M:%S")

            entry = {
                "start": timestamp_str,
                "text": seg["text"],
            }
            if source:
                entry["source"] = source

            lines.append(json.dumps(entry))

        return lines

    def _handle_raw(self, raw_path: Path, redo: bool = False) -> None:
        """Process a raw audio file.

        Args:
            raw_path: Path to audio file in segment directory
            redo: If True, skip "already processed" check
        """
        start_time = time.time()

        # Derive segment from path
        segment = get_segment_key(raw_path)

        # Skip if already processed (unless redo mode)
        jsonl_path = self._get_jsonl_path(raw_path)
        if not redo and jsonl_path.exists():
            logging.info(f"Already processed: {raw_path}")
            return

        # Prepare audio (convert M4A if needed)
        audio_path = prepare_audio_file(raw_path, SAMPLE_RATE)
        is_temp = audio_path != raw_path

        # Get remote name once for use in metadata and events
        remote = os.getenv("REMOTE_NAME")

        try:
            # Load entity names for initial prompt
            entity_names = load_entity_names(spoken=True)
            initial_prompt = None
            if entity_names:
                initial_prompt = ", ".join(entity_names)
                logging.info(f"Using {len(entity_names)} entity names as prompt hints")

            # Transcribe with faster-whisper
            segments = self._transcribe(audio_path, initial_prompt)

            # Skip if no speech detected - safe to delete since _transcribe() already
            # validated that VAD also detected minimal speech (raises RuntimeError otherwise)
            if not segments:
                logging.info(f"No speech detected in {raw_path}, removing file")
                raw_path.unlink()
                return

            # Extract date and time from path structure
            # Files are always in segment directories: YYYYMMDD/HHMMSS_LEN/audio.flac
            time_part = segment.split("_")[0] if segment else "000000"
            day_str = raw_path.parent.parent.name

            base_dt = datetime.datetime.strptime(
                f"{day_str}_{time_part}", "%Y%m%d_%H%M%S"
            )

            # Extract source from <source>_audio pattern
            # mic_audio -> "mic", sys_audio -> "sys", etc.
            source = None
            suffix = raw_path.stem
            if suffix.endswith("_audio") and suffix != "audio":
                source = suffix[:-6]  # Remove "_audio" suffix

            # Convert to JSONL format
            raw_filename = f"{raw_path.stem}{raw_path.suffix}"
            jsonl_lines = self._segments_to_jsonl(
                segments, raw_filename, base_dt, source, remote
            )

            # Write JSONL
            jsonl_path.write_text("\n".join(jsonl_lines) + "\n")
            logging.info(f"Transcribed {raw_path} -> {jsonl_path}")

            # Generate and save embeddings
            embeddings_data = self._embed_segments(audio_path, segments)
            if embeddings_data:
                embeddings_path = self._get_embeddings_path(raw_path)
                np.savez_compressed(embeddings_path, **embeddings_data)
                logging.info(f"Saved embeddings: {embeddings_path}")
            else:
                logging.warning(f"No embeddings generated for {raw_path}")

            # Emit completion event
            journal_path = Path(get_journal())
            duration_ms = int((time.time() - start_time) * 1000)

            try:
                rel_input = raw_path.relative_to(journal_path)
                rel_output = jsonl_path.relative_to(journal_path)
            except ValueError:
                rel_input = raw_path
                rel_output = jsonl_path

            # Extract day from audio path (grandparent is day dir)
            day = raw_path.parent.parent.name

            event_fields = {
                "input": str(rel_input),
                "output": str(rel_output),
                "duration_ms": duration_ms,
            }
            if day:
                event_fields["day"] = day
            if segment:
                event_fields["segment"] = segment
            if remote:
                event_fields["remote"] = remote
            callosum_send("observe", "transcribed", **event_fields)

        except Exception as e:
            logging.error(f"Failed to transcribe {raw_path}: {e}", exc_info=True)
            raise SystemExit(1) from e
        finally:
            # Clean up temp file if created
            if is_temp and audio_path.exists():
                audio_path.unlink()


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
    if get_segment_key(audio_path) is None:
        parser.error(
            f"Audio file must be in a segment directory (HHMMSS_LEN/), "
            f"but parent is: {audio_path.parent.name}"
        )

    # Load transcription settings from journal config
    config = get_config()
    transcribe_config = config.get("transcribe", {})

    # Determine settings: CLI args override config, config overrides defaults
    if args.cpu:
        device = "cpu"
        compute_type = "int8"
    else:
        device = transcribe_config.get("device", DEFAULT_DEVICE)
        compute_type = transcribe_config.get("compute_type", DEFAULT_COMPUTE)

    model = args.model or transcribe_config.get("model", DEFAULT_MODEL)

    logging.info(f"Processing audio: {audio_path}")

    transcriber = Transcriber(
        model_size=model, device=device, compute_type=compute_type
    )
    transcriber._handle_raw(audio_path, redo=args.redo)


if __name__ == "__main__":
    main()
