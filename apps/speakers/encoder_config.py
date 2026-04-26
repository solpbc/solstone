# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Encoder-specific calibration constants. Source of truth for owner-separation, acoustic-match, and owner-bootstrap quality thresholds. Changes require CPO spec revision; see cpo/specs/in-flight/speaker-attribution-wespeaker.md."""

ENCODER_ID: str = "wespeaker-resnet34-256"

OWNER_THRESHOLD: float = 0.43
ACOUSTIC_HIGH: float = 0.36
ACOUSTIC_MEDIUM: float = 0.22

OWNER_BOOTSTRAP_MIN_STMTS: int = 30
OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S: float = 1.5
OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25: float = 0.30
# Smallest manual-tag set that meaningfully constrains the contamination centroid; below this the no-op default holds.
OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS: int = 5

NOISY_FLYWHEEL_OVERLAP_MAX: float = 0.10
OVERLAP_DETECTOR_ID: str = "pyannote-segmentation-3.0-onnx"
OVERLAP_DETECTOR_SHA256: str = (
    "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25"
)
