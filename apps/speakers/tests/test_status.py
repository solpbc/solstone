# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speaker subsystem status."""

from __future__ import annotations

import numpy as np


def _save_principal_manual_tags(env, principal_id: str, count: int) -> None:
    from apps.speakers.routes import _save_voiceprint

    embeddings = np.zeros((count, 256), dtype=np.float32)
    embeddings[:, 0] = 1.0
    env.create_segment("20240101", "090000_300", ["audio"], embeddings=embeddings)
    env.create_speaker_labels(
        "20240101",
        "090000_300",
        [
            {
                "sentence_id": idx,
                "speaker": principal_id,
                "confidence": "high",
                "method": "user_assigned",
            }
            for idx in range(1, count + 1)
        ],
    )
    for idx, embedding in enumerate(embeddings, start=1):
        _save_voiceprint(
            principal_id,
            embedding,
            "20240101",
            "090000_300",
            "audio",
            idx,
            stream="test",
        )


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


def test_status_owner_includes_bootstrap_diagnostics(speakers_env):
    from apps.speakers.status import get_speakers_status

    env = speakers_env()
    env.create_entity("Self Person", is_principal=True)
    _save_principal_manual_tags(env, "self_person", 7)

    result = get_speakers_status(section="owner")

    assert result["status"] == "none"
    assert result["manual_tags_count"] == 7
    assert result["segments_available"] == 1
    assert result["embeddings_available"] == 7
    assert result["streams_represented"] == 1
    assert result["can_build_from_tags"] is False


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
