# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Test the constant-import contract for encoder_config."""

from solstone.apps.speakers import attribution, encoder_config, owner
from solstone.observe.transcribe.main import (
    OVERLAP_DETECTOR_ID as MAIN_OVERLAP_DETECTOR_ID,
)
from solstone.observe.transcribe.main import PYANNOTE_OVERLAP_MODEL_SHA256


def test_locked_constants():
    assert encoder_config.ENCODER_ID == "wespeaker-resnet34-256"
    assert encoder_config.OWNER_THRESHOLD == 0.43
    assert encoder_config.ACOUSTIC_HIGH == 0.36
    assert encoder_config.ACOUSTIC_MEDIUM == 0.22
    assert encoder_config.OWNER_BOOTSTRAP_MIN_STMTS == 30
    assert encoder_config.OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S == 1.5
    assert encoder_config.OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25 == 0.30
    assert encoder_config.OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS == 5
    assert encoder_config.NOISY_FLYWHEEL_OVERLAP_MAX == 0.10
    assert encoder_config.OVERLAP_DETECTOR_ID == MAIN_OVERLAP_DETECTOR_ID
    assert encoder_config.OVERLAP_DETECTOR_SHA256 == PYANNOTE_OVERLAP_MODEL_SHA256


def test_attribution_imports_acoustic_constants():
    assert attribution.ACOUSTIC_HIGH is encoder_config.ACOUSTIC_HIGH
    assert attribution.ACOUSTIC_MEDIUM is encoder_config.ACOUSTIC_MEDIUM


def test_owner_imports_constants():
    assert owner.OWNER_THRESHOLD is encoder_config.OWNER_THRESHOLD
    assert owner.OWNER_BOOTSTRAP_MIN_STMTS is encoder_config.OWNER_BOOTSTRAP_MIN_STMTS
    assert (
        owner.OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S
        is encoder_config.OWNER_BOOTSTRAP_MIN_MEDIAN_DURATION_S
    )
    assert (
        owner.OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25
        is encoder_config.OWNER_BOOTSTRAP_MIN_INTRA_COSINE_P25
    )
    assert (
        owner.OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS
        is encoder_config.OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS
    )
