# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Interpret audio files using faster-whisper with word-level embeddings.

Phase 1 of the new transcription pipeline:
- Transcribes audio using faster-whisper with word timestamps
- Re-segments by sentence boundaries (not acoustic pauses)
- Generates voice embeddings for each word using resemblyzer
- Saves intermediate output for Phase 2 speaker correlation

Output files are saved to <day>/<segment>/<stem>/:
- transcript.json: Sentence-aligned segments with word-level timestamps
- embeddings.npz: Voice embeddings indexed by word start time
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np

from observe.utils import (
    SAMPLE_RATE,
    get_output_dir,
    get_segment_key,
    prepare_audio_file,
)
from think.callosum import callosum_send
from think.utils import get_journal, setup_cli

# Default model for faster-whisper
DEFAULT_MODEL = "medium.en"

# Minimum word duration for embedding (seconds)
MIN_WORD_DURATION = 0.1

# Sentence-ending punctuation marks
SENTENCE_ENDINGS = frozenset(".?!")


def resegment_by_sentences(transcript: dict) -> dict:
    """Re-segment transcript by sentence boundaries instead of acoustic pauses.

    Whisper segments on speech pauses (VAD-driven), but we want segments aligned
    to sentence boundaries (punctuation-driven). This function flattens all words
    and re-segments based on sentence-ending punctuation.

    Args:
        transcript: Transcript dict with 'segments' containing word-level data

    Returns:
        New transcript dict with segments aligned to sentence boundaries
    """
    # Flatten all words from all segments
    all_words = []
    for seg in transcript.get("segments", []):
        all_words.extend(seg.get("words", []))

    if not all_words:
        return transcript

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

    # Return new transcript with resegmented data
    return {
        **transcript,
        "segments": new_segments,
    }


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


class Interpreter:
    """Interprets audio using faster-whisper and generates word embeddings."""

    def __init__(self, model_size: str = DEFAULT_MODEL):
        """Initialize interpreter with models.

        Args:
            model_size: faster-whisper model size (e.g., "medium.en", "small.en")
        """
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder

        logging.info(f"Loading faster-whisper model ({model_size})...")
        t0 = time.perf_counter()
        self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logging.info(f"  Whisper loaded in {time.perf_counter() - t0:.2f}s")

        logging.info("Loading resemblyzer VoiceEncoder...")
        t0 = time.perf_counter()
        self.voice_encoder = VoiceEncoder(device="cpu")
        logging.info(f"  VoiceEncoder loaded in {time.perf_counter() - t0:.2f}s")

        self.model_size = model_size

    def _get_output_dir(self, audio_path: Path) -> Path:
        """Get output directory for interpreted files."""
        return get_output_dir(audio_path)

    def _transcribe(self, audio_path: Path) -> dict | None:
        """Transcribe audio using faster-whisper.

        Args:
            audio_path: Path to audio file (FLAC)

        Returns:
            Dict with transcript data or None on error:
            - info: language, probability, duration, model
            - raw: source filename
            - segments: list of segment dicts with words
        """
        try:
            logging.info(f"Transcribing {audio_path.name}...")
            t0 = time.perf_counter()

            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                language="en",
                vad_filter=True,
                beam_size=5,
                word_timestamps=True,
            )

            # Consume generator and build output
            segment_list = []
            for seg in segments:
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

            logging.info(
                f"  Transcribed {len(segment_list)} segments, "
                f"{duration:.1f}s audio in {transcribe_time:.2f}s "
                f"(RTF: {transcribe_time / max(duration, 0.1):.3f}x)"
            )

            transcript = {
                "info": {
                    "language": info.language,
                    "probability": info.language_probability,
                    "duration": duration,
                    "transcribe_time": transcribe_time,
                    "model": self.model_size,
                },
                "raw": f"{audio_path.stem}{audio_path.suffix}",
                "segments": segment_list,
            }

            # Re-segment by sentence boundaries instead of acoustic pauses
            whisper_segments = len(segment_list)
            transcript = resegment_by_sentences(transcript)
            logging.info(
                f"  Re-segmented {whisper_segments} acoustic segments "
                f"to {len(transcript['segments'])} sentences"
            )

            return transcript

        except Exception as e:
            logging.error(f"Transcription failed for {audio_path}: {e}", exc_info=True)
            return None

    def _embed_words(self, audio_path: Path, transcript: dict) -> dict | None:
        """Generate voice embeddings for words in transcript.

        Args:
            audio_path: Path to audio file
            transcript: Transcript dict from _transcribe()

        Returns:
            Dict with embedding data or None on error:
            - embeddings: (N, 256) float32 array
            - starts: (N,) float32 array of word start times
        """
        try:
            # Load audio without VAD preprocessing to preserve timing
            logging.info(f"Loading audio for embeddings...")
            t0 = time.perf_counter()
            wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            logging.info(f"  Audio loaded in {time.perf_counter() - t0:.2f}s")

            # Collect all words with sufficient duration
            all_words = []
            for seg in transcript.get("segments", []):
                for w in seg.get("words", []):
                    duration = w["end"] - w["start"]
                    if duration >= MIN_WORD_DURATION:
                        all_words.append(w)

            if not all_words:
                logging.info("No words with sufficient duration for embedding")
                return None

            logging.info(f"Embedding {len(all_words)} words...")
            t0 = time.perf_counter()

            embeddings = []
            starts = []
            skipped = 0

            for i, w in enumerate(all_words):
                start_sample = int(w["start"] * SAMPLE_RATE)
                end_sample = int(w["end"] * SAMPLE_RATE)
                word_audio = wav[start_sample:end_sample]

                # Skip if too short after slicing
                if len(word_audio) < int(MIN_WORD_DURATION * SAMPLE_RATE):
                    skipped += 1
                    continue

                try:
                    emb = self.voice_encoder.embed_utterance(word_audio)
                    embeddings.append(emb)
                    starts.append(w["start"])
                except Exception:
                    skipped += 1
                    continue

                if (i + 1) % 100 == 0:
                    logging.debug(f"  {i + 1}/{len(all_words)} words...")

            embed_time = time.perf_counter() - t0

            if not embeddings:
                logging.warning("No embeddings generated")
                return None

            logging.info(
                f"  Embedded {len(embeddings)} words "
                f"(skipped {skipped}) in {embed_time:.2f}s"
            )

            return {
                "embeddings": np.array(embeddings, dtype=np.float32),
                "starts": np.array(starts, dtype=np.float32),
            }

        except Exception as e:
            logging.error(f"Embedding failed for {audio_path}: {e}", exc_info=True)
            return None

    def _handle_raw(self, raw_path: Path, redo: bool = False) -> None:
        """Process a raw audio file.

        Args:
            raw_path: Path to audio file in segment directory
            redo: If True, skip "already processed" check
        """
        start_time = time.time()

        # Get output directory and check if already processed
        output_dir = self._get_output_dir(raw_path)
        transcript_path = output_dir / "transcript.json"

        if not redo and transcript_path.exists():
            logging.info(f"Already processed: {raw_path}")
            return

        # Derive segment from path
        segment = get_segment_key(raw_path)

        # Prepare audio (convert M4A if needed)
        audio_path = prepare_audio_file(raw_path, SAMPLE_RATE)
        is_temp = audio_path != raw_path

        try:
            # Transcribe with faster-whisper
            transcript = self._transcribe(audio_path)
            if transcript is None:
                raise SystemExit(1)

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Add remote origin if set
            remote = os.getenv("REMOTE_NAME")
            if remote:
                transcript["remote"] = remote

            # Save transcript (always, regardless of embedding success)
            transcript_path.write_text(json.dumps(transcript, indent=2))
            logging.info(f"Saved transcript: {transcript_path}")

            # Generate and save embeddings
            embeddings_data = self._embed_words(audio_path, transcript)
            if embeddings_data:
                embeddings_path = output_dir / "embeddings.npz"
                np.savez_compressed(embeddings_path, **embeddings_data)
                logging.info(f"Saved embeddings: {embeddings_path}")
            else:
                logging.warning(f"No embeddings generated for {raw_path}")

            # Emit completion event
            journal_path = Path(get_journal())
            duration_ms = int((time.time() - start_time) * 1000)

            try:
                rel_input = raw_path.relative_to(journal_path)
                rel_output = output_dir.relative_to(journal_path)
            except ValueError:
                rel_input = raw_path
                rel_output = output_dir

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

            callosum_send("observe", "interpreted", **event_fields)

        finally:
            # Clean up temp file if created
            if is_temp and audio_path.exists():
                audio_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Interpret audio files using faster-whisper with word embeddings"
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

    logging.info(f"Processing audio: {audio_path}")

    interpreter = Interpreter()
    interpreter._handle_raw(audio_path, redo=args.redo)


if __name__ == "__main__":
    main()
