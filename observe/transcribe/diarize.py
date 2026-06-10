# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Local speaker diarization using pyannote segmentation + WeSpeaker + AHC.

Pipeline:
  1. Run pyannote-segmentation-3.0 on the raw audio at frame level.
  2. Find intervals where a single local speaker dominates (classes 1–3).
  3. Extract one WeSpeaker embedding per interval (clean, single-speaker audio).
  4. Cluster interval embeddings with AHC → global speaker IDs.
  5. Map sentences to intervals by timestamp overlap.

Returns a list of integer speaker labels (1-indexed) parallel to the input
sentences list, matching the format of the `speaker` field Gemini writes into
audio.jsonl.  Sentences with no single-speaker interval coverage get None.

Public API:
    from observe.transcribe.diarize import diarize, diarize_auto_k

    labels = diarize(wav_path, sentences)            # auto-estimate k
    labels = diarize(wav_path, sentences, n_speakers=4)  # known k
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent / "assets"
PYANNOTE_MODEL_PATH = ASSETS_DIR / "pyannote-segmentation-3.0.onnx"
WESPEAKER_MODEL_PATH = ASSETS_DIR / "wespeaker-resnet34-256.onnx"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
WINDOW_S = 10
STRIDE_S = 2
FRAMES_PER_WINDOW = 589

# pyannote class layout: 0=silence, 1=spkA, 2=spkB, 3=spkC, 4=A+B, 5=A+C, 6=B+C
SINGLE_SPEAKER_CLASSES = frozenset({1, 2, 3})

MIN_INTERVAL_S = 0.5
MIN_FRAME_CONFIDENCE = 0.50

AHC_LINKAGE = "average"
AHC_METRIC = "cosine"
MAX_K = 8
SILHOUETTE_IMPROVEMENT = 0.03

# ---------------------------------------------------------------------------
# Module-level session cache
# ---------------------------------------------------------------------------

_pyannote_session = None
_wespeaker_session = None


def _get_pyannote_session():
    global _pyannote_session
    if _pyannote_session is None:
        import onnxruntime as ort

        if not PYANNOTE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"pyannote model not found at {PYANNOTE_MODEL_PATH}"
            )
        _pyannote_session = ort.InferenceSession(
            str(PYANNOTE_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        logger.debug("pyannote session loaded")
    return _pyannote_session


def _get_wespeaker_session():
    global _wespeaker_session
    if _wespeaker_session is None:
        import onnxruntime as ort

        if not WESPEAKER_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"WeSpeaker model not found at {WESPEAKER_MODEL_PATH}"
            )
        _wespeaker_session = ort.InferenceSession(
            str(WESPEAKER_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        logger.debug("wespeaker session loaded")
    return _wespeaker_session


# ---------------------------------------------------------------------------
# Step 1: run pyannote → per-frame averaged log-probabilities
# ---------------------------------------------------------------------------


def _run_pyannote(audio: np.ndarray) -> np.ndarray:
    """Return averaged log-probs (num_frames, 7) across overlapping windows."""
    session = _get_pyannote_session()
    input_name = session.get_inputs()[0].name

    window_samples = WINDOW_S * SAMPLE_RATE
    stride_samples = STRIDE_S * SAMPLE_RATE
    samples_per_frame = window_samples / FRAMES_PER_WINDOW

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < window_samples:
        pad = window_samples - len(audio_f32)
        audio_f32 = np.concatenate([audio_f32, np.zeros(pad, dtype=np.float32)])

    starts = list(range(0, len(audio_f32) - window_samples + 1, stride_samples))
    final_start = max(0, len(audio_f32) - window_samples)
    if not starts:
        starts = [final_start]
    elif starts[-1] != final_start:
        starts.append(final_start)

    num_frames = int(np.ceil(len(audio_f32) / samples_per_frame))
    accum = np.zeros((num_frames, 7), dtype=np.float64)
    counts = np.zeros(num_frames, dtype=np.int32)

    for start in starts:
        chunk = audio_f32[start : start + window_samples][None, None, :]
        log_probs = session.run(None, {input_name: chunk})[0][0]
        frame_start = int(round(start / samples_per_frame))
        frame_end = min(frame_start + log_probs.shape[0], num_frames)
        log_probs = log_probs[: frame_end - frame_start]
        accum[frame_start:frame_end] += log_probs.astype(np.float64)
        counts[frame_start:frame_end] += 1

    counts = np.maximum(counts, 1)
    avg_log_probs = (accum / counts[:, None]).astype(np.float32)
    return avg_log_probs


# ---------------------------------------------------------------------------
# Step 2: find single-speaker intervals
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _find_intervals(
    avg_log_probs: np.ndarray,
    audio_len_samples: int,
) -> list[tuple[float, float, int]]:
    """Find single-speaker intervals; returns (start_s, end_s, local_class) triples."""
    num_frames = avg_log_probs.shape[0]
    probs = _softmax(avg_log_probs)
    argmax = avg_log_probs.argmax(axis=-1)
    confidence = probs[np.arange(num_frames), argmax]

    samples_per_frame = (WINDOW_S * SAMPLE_RATE) / FRAMES_PER_WINDOW
    audio_duration_s = audio_len_samples / SAMPLE_RATE

    intervals: list[tuple[float, float, int]] = []
    run_class: int | None = None
    run_start_frame: int = 0

    def _flush(end_frame: int) -> None:
        if run_class is None:
            return
        start_s = (run_start_frame * samples_per_frame) / SAMPLE_RATE
        end_s = min((end_frame * samples_per_frame) / SAMPLE_RATE, audio_duration_s)
        if end_s - start_s >= MIN_INTERVAL_S:
            intervals.append((start_s, end_s, run_class))

    for i in range(num_frames):
        cls = int(argmax[i])
        conf = float(confidence[i])
        is_single = cls in SINGLE_SPEAKER_CLASSES and conf >= MIN_FRAME_CONFIDENCE

        if is_single:
            if cls != run_class:
                _flush(i)
                run_class = cls
                run_start_frame = i
        else:
            _flush(i)
            run_class = None

    _flush(num_frames)
    return intervals


# ---------------------------------------------------------------------------
# Step 3: WeSpeaker embedding per interval
# ---------------------------------------------------------------------------


def _wespeaker_features(audio_slice: np.ndarray) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(SAMPLE_RATE)
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0
    opts.use_energy = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(
        float(SAMPLE_RATE),
        (audio_slice.astype(np.float32) * 32768.0).tolist(),
    )
    fbank.input_finished()

    frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    if not frames:
        return np.zeros((0, 80), dtype=np.float32)
    feats = np.stack(frames).astype(np.float32)
    feats -= feats.mean(axis=0, keepdims=True)
    return feats


def _embed_interval(
    audio: np.ndarray, start_s: float, end_s: float
) -> np.ndarray | None:
    session = _get_wespeaker_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_sample = int(start_s * SAMPLE_RATE)
    end_sample = int(end_s * SAMPLE_RATE)
    slice_audio = audio[start_sample:end_sample]

    if len(slice_audio) < int(MIN_INTERVAL_S * SAMPLE_RATE):
        return None

    feats = _wespeaker_features(slice_audio)
    if feats.shape[0] == 0:
        return None

    emb = session.run([output_name], {input_name: feats[None, :, :]})[0][0]
    return emb.astype(np.float32)


def _embed_all_intervals(
    audio: np.ndarray,
    intervals: list[tuple[float, float, int]],
) -> tuple[list[tuple[float, float, int]], np.ndarray]:
    valid: list[tuple[float, float, int]] = []
    embs: list[np.ndarray] = []

    for ivl in intervals:
        start_s, end_s, local_cls = ivl
        emb = _embed_interval(audio, start_s, end_s)
        if emb is not None:
            valid.append(ivl)
            embs.append(emb)

    if not embs:
        return [], np.zeros((0, 256), dtype=np.float32)
    return valid, np.stack(embs)


# ---------------------------------------------------------------------------
# Step 4: AHC clustering of interval embeddings
# ---------------------------------------------------------------------------


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms > 1e-9, norms, 1.0)


def _ahc(embs_n: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        n_clusters=k, metric=AHC_METRIC, linkage=AHC_LINKAGE
    ).fit_predict(embs_n)


def _silhouette(embs_n: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    n_cls = len(np.unique(labels))
    if n_cls < 2 or len(embs_n) <= n_cls:
        return -1.0
    dist = np.clip(1.0 - embs_n @ embs_n.T, 0.0, None)
    return float(silhouette_score(dist, labels, metric="precomputed"))


def _pick_k_silhouette(embs_n: np.ndarray, max_k: int) -> int:
    n = len(embs_n)
    effective_max = min(max_k, n - 1)
    if effective_max < 2:
        return 1
    best_k, best_s = 1, -1.0
    for k in range(2, effective_max + 1):
        labels = _ahc(embs_n, k)
        s = _silhouette(embs_n, labels)
        if s > best_s + SILHOUETTE_IMPROVEMENT:
            best_s, best_k = s, k
    return best_k


def _cluster_intervals(embs: np.ndarray, n_speakers: int | None) -> np.ndarray:
    embs_n = _normalize_rows(embs)
    k = n_speakers if n_speakers is not None else _pick_k_silhouette(embs_n, MAX_K)
    k = max(1, min(k, len(embs) - 1 if len(embs) > 1 else 1))
    if k <= 1:
        return np.zeros(len(embs), dtype=np.int32)
    return _ahc(embs_n, k).astype(np.int32)


# ---------------------------------------------------------------------------
# Step 5: map sentences to interval speaker labels
# ---------------------------------------------------------------------------


def _assign_sentences(
    sentences: list[dict],
    intervals: list[tuple[float, float, int]],
    global_labels: np.ndarray,
) -> list[int | None]:
    """Assign each sentence a global speaker ID (1-indexed) by max overlap."""
    result: list[int | None] = []

    for sent in sentences:
        start = sent.get("start")
        end = sent.get("end")
        if start is None or end is None or end <= start:
            result.append(None)
            continue

        best_overlap = 0.0
        best_label: int | None = None

        for idx, (ivl_start, ivl_end, _) in enumerate(intervals):
            overlap = max(0.0, min(end, ivl_end) - max(start, ivl_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = int(global_labels[idx]) + 1  # 1-indexed to match Gemini

        result.append(best_label)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diarize(
    wav_path: "Path | str",
    sentences: list[dict],
    n_speakers: int | None = None,
    avg_log_probs: "np.ndarray | None" = None,
    audio: "np.ndarray | None" = None,
) -> list[int | None]:
    """Assign speaker labels to sentences from raw audio.

    Args:
        wav_path: Path to the .wav file (16kHz mono preferred; resampled if not).
        sentences: List of sentence dicts with 'start', 'end' keys in seconds
                   from audio start (same format as Parakeet/Whisper output).
        n_speakers: Known number of speakers. If None, estimated via silhouette.
        avg_log_probs: Pre-computed pyannote log-probs (num_frames, 7). When
                       provided the pyannote inference pass is skipped entirely.
        audio: Pre-loaded 16kHz mono audio array. When provided the wav file is
               not re-read from disk.

    Returns:
        List of integer speaker labels (1-indexed) parallel to sentences.
        Sentences with no single-speaker interval coverage get None.
    """
    t0 = time.perf_counter()

    if audio is None:
        import soundfile as sf
        from scipy.signal import resample_poly

        raw_audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raw_audio = resample_poly(raw_audio, SAMPLE_RATE, sr).astype(np.float32)
        audio = raw_audio

    if avg_log_probs is None:
        avg_log_probs = _run_pyannote(audio)

    intervals = _find_intervals(avg_log_probs, len(audio))
    if not intervals:
        logger.debug("diarize: no single-speaker intervals found")
        return [None] * len(sentences)

    valid_intervals, embs = _embed_all_intervals(audio, intervals)
    if len(embs) == 0:
        logger.debug("diarize: no interval embeddings produced")
        return [None] * len(sentences)

    global_labels = _cluster_intervals(embs, n_speakers)
    labels = _assign_sentences(sentences, valid_intervals, global_labels)

    elapsed = time.perf_counter() - t0
    assigned = sum(1 for lb in labels if lb is not None)
    logger.debug(
        "diarize: %d intervals → %d clusters → %d/%d sentences assigned in %.1fs",
        len(valid_intervals),
        int(global_labels.max()) + 1 if len(global_labels) > 0 else 0,
        assigned,
        len(sentences),
        elapsed,
    )

    return labels


def diarize_auto_k(
    wav_path: "Path | str",
    sentences: list[dict],
    avg_log_probs: "np.ndarray | None" = None,
    audio: "np.ndarray | None" = None,
) -> list[int | None]:
    """Diarize with silhouette-estimated number of speakers."""
    return diarize(
        wav_path, sentences, n_speakers=None, avg_log_probs=avg_log_probs, audio=audio
    )
