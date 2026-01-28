# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Voice Activity Detection (VAD) module using Silero VAD.

Provides a standalone VAD stage that can be run before transcription to:
- Filter out silent/low-speech audio files early (before loading heavy STT models)
- Provide speech duration metrics for logging and events
- Reduce audio by trimming long silence gaps (>2s trimmed to 2s with 1s buffers)
- Compute RMS of non-speech regions for background noise detection

Uses Silero VAD via faster-whisper's bundled implementation.

Audio Reduction:
When there are long gaps (>2s) between speech segments, the audio can be
"reduced" by trimming those gaps to a maximum of 2s (1s buffer on each side).
This creates a shorter audio buffer that any STT backend can process more
efficiently. A mapping is preserved to restore original timestamps after
transcription.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from observe.utils import SAMPLE_RATE

# Minimum silence gap to reduce (seconds)
MIN_GAP_TO_REDUCE = 2.0

# Buffer to keep on each side of trimmed gap (seconds)
GAP_BUFFER = 1.0

# Minimum non-speech segment duration for RMS calculation (seconds)
MIN_NONSPEECH_SEGMENT = 0.5


def get_nonspeech_segments(
    speech_segments: list[tuple[float, float]], total_duration: float
) -> list[tuple[float, float]]:
    """Invert speech segments to get non-speech (silence/noise) gaps.

    Args:
        speech_segments: List of (start, end) tuples for speech
        total_duration: Total audio duration in seconds

    Returns:
        List of (start, end) tuples for non-speech regions
    """
    nonspeech = []

    # Leading silence (before first speech)
    if speech_segments and speech_segments[0][0] > 0:
        nonspeech.append((0.0, speech_segments[0][0]))

    # Gaps between speech segments
    for i in range(len(speech_segments) - 1):
        gap_start = speech_segments[i][1]  # end of current
        gap_end = speech_segments[i + 1][0]  # start of next
        if gap_end > gap_start:
            nonspeech.append((gap_start, gap_end))

    # Trailing silence (after last speech)
    if speech_segments and speech_segments[-1][1] < total_duration:
        nonspeech.append((speech_segments[-1][1], total_duration))

    return nonspeech


def compute_nonspeech_rms(
    audio: np.ndarray,
    speech_segments: list[tuple[float, float]],
    sample_rate: int,
    min_segment: float = MIN_NONSPEECH_SEGMENT,
) -> tuple[float | None, float]:
    """Compute RMS level of non-speech regions.

    Args:
        audio: Audio samples as numpy array
        speech_segments: List of (start, end) tuples for speech
        sample_rate: Audio sample rate in Hz
        min_segment: Minimum non-speech segment duration to include (seconds)

    Returns:
        Tuple of (rms_value, duration_used):
        - rms_value: Mean RMS across qualifying non-speech segments, or None if none found
        - duration_used: Total seconds of non-speech audio used for calculation
    """
    duration = len(audio) / sample_rate
    nonspeech = get_nonspeech_segments(speech_segments, duration)

    # Filter to segments >= min_segment seconds
    long_segments = [(s, e) for s, e in nonspeech if e - s >= min_segment]

    if not long_segments:
        return None, 0.0

    # Compute RMS for each segment
    rms_values = []
    total_duration = 0.0
    for start, end in long_segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio[start_sample:end_sample]
        rms = np.sqrt(np.mean(segment_audio**2))
        rms_values.append(rms)
        total_duration += end - start

    return float(np.mean(rms_values)), total_duration


@dataclass
class SpeechSegment:
    """A segment of speech with original and reduced timestamps.

    Attributes:
        original_start: Start time in original audio (seconds)
        original_end: End time in original audio (seconds)
        reduced_start: Start time in reduced audio (seconds)
        reduced_end: End time in reduced audio (seconds)
    """

    original_start: float
    original_end: float
    reduced_start: float
    reduced_end: float


@dataclass
class AudioReduction:
    """Mapping between reduced and original audio timestamps.

    Contains the speech segments that were preserved and provides methods
    to restore original timestamps from reduced timestamps.

    Attributes:
        segments: List of speech segments with both original and reduced times
        original_duration: Duration of original audio (seconds)
        reduced_duration: Duration of reduced audio (seconds)
    """

    segments: list[SpeechSegment] = field(default_factory=list)
    original_duration: float = 0.0
    reduced_duration: float = 0.0

    def restore_timestamp(self, reduced_time: float) -> float:
        """Convert a timestamp from reduced audio to original audio time.

        Args:
            reduced_time: Timestamp in reduced audio (seconds)

        Returns:
            Corresponding timestamp in original audio (seconds)
        """
        # Handle timestamps before first segment (in leading buffer region)
        if self.segments:
            first = self.segments[0]
            if reduced_time < first.reduced_start:
                # Map offset from first segment back to original leading region
                offset = first.reduced_start - reduced_time
                return first.original_start - offset

        for seg in self.segments:
            if seg.reduced_start <= reduced_time <= seg.reduced_end:
                # Linear mapping within segment
                offset = reduced_time - seg.reduced_start
                return seg.original_start + offset

            # Check if in gap between this segment and next
            # (gaps are preserved at reduced size, map to middle of original gap)
            seg_idx = self.segments.index(seg)
            if seg_idx < len(self.segments) - 1:
                next_seg = self.segments[seg_idx + 1]
                if seg.reduced_end < reduced_time < next_seg.reduced_start:
                    # In the reduced gap - map proportionally to original gap
                    reduced_gap = next_seg.reduced_start - seg.reduced_end
                    original_gap = next_seg.original_start - seg.original_end
                    gap_progress = (reduced_time - seg.reduced_end) / reduced_gap
                    return seg.original_end + (gap_progress * original_gap)

        # After all segments - extrapolate from last segment
        if self.segments:
            last = self.segments[-1]
            offset = reduced_time - last.reduced_end
            return last.original_end + offset

        # No segments - return as-is
        return reduced_time


@dataclass
class VadResult:
    """Result of Voice Activity Detection analysis.

    Attributes:
        duration: Total audio duration in seconds
        speech_duration: Duration of detected speech in seconds
        has_speech: Whether speech duration meets minimum threshold
        speech_segments: List of (start, end) tuples for each speech segment
        noisy_rms: RMS level of non-speech regions (None if not computable)
        noisy_s: Duration of non-speech audio used for RMS calculation
    """

    duration: float
    speech_duration: float
    has_speech: bool
    speech_segments: list[tuple[float, float]] = field(default_factory=list)
    noisy_rms: float | None = None
    noisy_s: float = 0.0

    def is_noisy(self, threshold: float = 0.01) -> bool:
        """Check if background noise level exceeds threshold.

        Args:
            threshold: RMS threshold for noisy background (default: 0.01)

        Returns:
            True if non-speech RMS exceeds threshold, False otherwise
        """
        return self.noisy_rms is not None and self.noisy_rms > threshold


def run_vad(
    audio: np.ndarray,
    min_speech_seconds: float,
) -> VadResult:
    """Run Voice Activity Detection on an audio buffer.

    Runs Silero VAD to identify speech segments. This can be used to
    filter silent files before loading heavier transcription models.
    Also computes RMS of non-speech regions for background noise detection.

    Args:
        audio: Audio waveform (float32 mono at SAMPLE_RATE)
        min_speech_seconds: Minimum speech duration to set has_speech=True

    Returns:
        VadResult with duration info, has_speech flag, speech segment boundaries,
        and non-speech RMS level for noise detection
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    logging.info("Running VAD...")
    t0 = time.perf_counter()

    duration = len(audio) / SAMPLE_RATE

    # Run Silero VAD with default options
    # threshold=0.5 for speech start, min_silence_duration_ms=1000
    vad_options = VadOptions(min_silence_duration_ms=1000)
    speech_chunks = get_speech_timestamps(audio, vad_options, sampling_rate=SAMPLE_RATE)

    # Calculate speech duration and extract segment boundaries
    speech_samples = sum(chunk["end"] - chunk["start"] for chunk in speech_chunks)
    speech_duration = speech_samples / SAMPLE_RATE

    # Convert sample indices to time boundaries
    speech_segments = [
        (chunk["start"] / SAMPLE_RATE, chunk["end"] / SAMPLE_RATE)
        for chunk in speech_chunks
    ]

    has_speech = speech_duration >= min_speech_seconds

    # Compute RMS of non-speech regions (for noise detection)
    noisy_rms, noisy_s = compute_nonspeech_rms(audio, speech_segments, SAMPLE_RATE)

    vad_time = time.perf_counter() - t0
    rms_str = f", rms={noisy_rms:.4f}" if noisy_rms is not None else ""
    logging.info(
        f"  VAD complete in {vad_time:.2f}s: "
        f"{duration:.1f}s total, {speech_duration:.1f}s speech, "
        f"{len(speech_chunks)} chunks, has_speech={has_speech}{rms_str}"
    )

    return VadResult(
        duration=duration,
        speech_duration=speech_duration,
        has_speech=has_speech,
        speech_segments=speech_segments,
        noisy_rms=noisy_rms,
        noisy_s=noisy_s,
    )


def reduce_audio(
    audio: np.ndarray,
    vad_result: VadResult,
) -> tuple[np.ndarray | None, AudioReduction | None]:
    """Reduce audio by trimming long silence gaps.

    Gaps longer than MIN_GAP_TO_REDUCE (2s) are trimmed to 2s total
    (GAP_BUFFER of 1s on each side). This creates a shorter audio buffer
    that any STT backend can process more efficiently.

    Args:
        audio: Audio waveform (float32 mono at SAMPLE_RATE)
        vad_result: VAD result with speech segment boundaries

    Returns:
        Tuple of (reduced_audio_array, reduction_mapping):
        - If no reduction needed (no gaps > 2s), returns (None, None)
        - Otherwise returns the reduced audio numpy array and the mapping
    """
    if not vad_result.speech_segments:
        return None, None

    # Calculate gaps between speech segments
    gaps_to_reduce = []
    for i in range(len(vad_result.speech_segments) - 1):
        current_end = vad_result.speech_segments[i][1]
        next_start = vad_result.speech_segments[i + 1][0]
        gap = next_start - current_end
        if gap > MIN_GAP_TO_REDUCE:
            gaps_to_reduce.append((i, current_end, next_start, gap))

    # Also check leading silence (before first speech)
    first_start = vad_result.speech_segments[0][0]
    if first_start > MIN_GAP_TO_REDUCE:
        gaps_to_reduce.insert(0, (-1, 0.0, first_start, first_start))

    # Check trailing silence (after last speech)
    last_end = vad_result.speech_segments[-1][1]
    trailing_gap = vad_result.duration - last_end
    if trailing_gap > MIN_GAP_TO_REDUCE:
        gaps_to_reduce.append(
            (
                len(vad_result.speech_segments) - 1,
                last_end,
                vad_result.duration,
                trailing_gap,
            )
        )

    if not gaps_to_reduce:
        logging.info("  No gaps > 2s to reduce")
        return None, None

    # Build reduced audio by copying segments and trimmed gaps
    reduced_chunks = []
    reduction_segments = []
    current_reduced_time = 0.0

    # Process each speech segment and the gap after it
    for i, (seg_start, seg_end) in enumerate(vad_result.speech_segments):
        # Handle leading gap (before first segment)
        if i == 0 and first_start > MIN_GAP_TO_REDUCE:
            # Keep only GAP_BUFFER before first speech
            buffer_start_sample = int((seg_start - GAP_BUFFER) * SAMPLE_RATE)
            seg_start_sample = int(seg_start * SAMPLE_RATE)
            reduced_chunks.append(audio[buffer_start_sample:seg_start_sample])
            current_reduced_time = GAP_BUFFER
        elif i == 0:
            # Keep all leading audio (gap <= 2s)
            seg_start_sample = int(seg_start * SAMPLE_RATE)
            reduced_chunks.append(audio[:seg_start_sample])
            current_reduced_time = seg_start

        # Copy speech segment
        seg_start_sample = int(seg_start * SAMPLE_RATE)
        seg_end_sample = int(seg_end * SAMPLE_RATE)
        reduced_chunks.append(audio[seg_start_sample:seg_end_sample])

        # Record mapping for this speech segment
        seg_duration = seg_end - seg_start
        reduction_segments.append(
            SpeechSegment(
                original_start=seg_start,
                original_end=seg_end,
                reduced_start=current_reduced_time,
                reduced_end=current_reduced_time + seg_duration,
            )
        )
        current_reduced_time += seg_duration

        # Handle gap after this segment
        if i < len(vad_result.speech_segments) - 1:
            next_start = vad_result.speech_segments[i + 1][0]
            gap = next_start - seg_end

            if gap > MIN_GAP_TO_REDUCE:
                # Trim to 2s: keep GAP_BUFFER after current segment and GAP_BUFFER before next
                after_sample = int((seg_end + GAP_BUFFER) * SAMPLE_RATE)
                before_sample = int((next_start - GAP_BUFFER) * SAMPLE_RATE)
                reduced_chunks.append(audio[seg_end_sample:after_sample])
                reduced_chunks.append(
                    audio[before_sample : int(next_start * SAMPLE_RATE)]
                )
                current_reduced_time += 2 * GAP_BUFFER  # 2s total gap
            else:
                # Keep full gap
                gap_end_sample = int(next_start * SAMPLE_RATE)
                reduced_chunks.append(audio[seg_end_sample:gap_end_sample])
                current_reduced_time += gap

    # Handle trailing gap
    if trailing_gap > MIN_GAP_TO_REDUCE:
        # Keep only GAP_BUFFER at start of trailing gap
        trail_sample = int((last_end + GAP_BUFFER) * SAMPLE_RATE)
        reduced_chunks.append(audio[int(last_end * SAMPLE_RATE) : trail_sample])
        current_reduced_time += GAP_BUFFER
    else:
        # Keep all trailing audio
        reduced_chunks.append(audio[int(last_end * SAMPLE_RATE) :])
        current_reduced_time += trailing_gap

    # Concatenate reduced audio
    reduced_audio = np.concatenate(reduced_chunks)
    reduced_duration = len(reduced_audio) / SAMPLE_RATE

    reduction = AudioReduction(
        segments=reduction_segments,
        original_duration=vad_result.duration,
        reduced_duration=reduced_duration,
    )

    time_saved = vad_result.duration - reduced_duration
    logging.info(
        f"  Reduced audio: {vad_result.duration:.1f}s -> {reduced_duration:.1f}s "
        f"(saved {time_saved:.1f}s, {len(gaps_to_reduce)} gaps trimmed)"
    )

    return reduced_audio, reduction


def restore_statement_timestamps(
    statements: list[dict],
    reduction: AudioReduction,
) -> list[dict]:
    """Restore original timestamps to statements transcribed from reduced audio.

    Args:
        statements: List of statement dicts with 'start', 'end', and optionally 'words'
        reduction: AudioReduction mapping from reduce_audio()

    Returns:
        New list of statements with timestamps restored to original audio time
    """
    restored = []
    for stmt in statements:
        new_stmt = stmt.copy()

        # Restore statement-level timestamps
        new_stmt["start"] = reduction.restore_timestamp(stmt["start"])
        new_stmt["end"] = reduction.restore_timestamp(stmt["end"])

        # Restore word-level timestamps if present
        if "words" in stmt and stmt["words"]:
            new_words = []
            for word in stmt["words"]:
                new_word = word.copy()
                new_word["start"] = reduction.restore_timestamp(word["start"])
                new_word["end"] = reduction.restore_timestamp(word["end"])
                new_words.append(new_word)
            new_stmt["words"] = new_words

        restored.append(new_stmt)

    return restored
