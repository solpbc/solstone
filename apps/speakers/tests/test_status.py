# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speaker subsystem status."""

from __future__ import annotations


def test_status_all_sections(speakers_env):
    from apps.speakers.status import get_speakers_status

    speakers_env()
    result = get_speakers_status()
    assert "embeddings" in result
    assert "owner" in result
    assert "speakers" in result
    assert "clusters" in result
    assert "imports" in result
    assert "attribution" in result


def test_status_single_section(speakers_env):
    from apps.speakers.status import get_speakers_status

    speakers_env()
    result = get_speakers_status(section="owner")
    assert "status" in result
    assert "centroid_saved" in result


def test_status_unknown_section(speakers_env):
    from apps.speakers.status import get_speakers_status

    speakers_env()
    result = get_speakers_status(section="nonexistent")
    assert "error" in result


def test_status_embeddings_with_data(speakers_env):
    from apps.speakers.status import get_speakers_status

    env = speakers_env()
    env.create_segment("20240101", "090000_300", ["mic_audio"])
    env.create_segment("20240101", "091000_300", ["sys_audio"])
    env.create_segment("20240102", "090000_300", ["audio"])

    result = get_speakers_status(section="embeddings")
    assert result["segments"] == 3
    assert result["days"] == 2
    assert result["date_range"] == ["20240101", "20240102"]


def test_status_attribution_with_labels(speakers_env):
    from apps.speakers.status import get_speakers_status

    env = speakers_env()
    env.create_speaker_labels(
        "20240101",
        "090000_300",
        [
            {
                "sentence_id": 1,
                "speaker": "alice",
                "confidence": "high",
                "method": "voiceprint",
            },
            {
                "sentence_id": 2,
                "speaker": None,
                "confidence": "low",
                "method": "unmatched",
            },
        ],
    )

    result = get_speakers_status(section="attribution")
    assert result["files"] == 1
    assert result["labels"] == 2
    assert result["by_confidence"]["high"] == 1
    assert result["by_confidence"]["low"] == 1
    assert result["by_method"]["voiceprint"] == 1
    assert result["by_method"]["unmatched"] == 1
