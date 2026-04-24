# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for speaker artifact wipe reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.speakers.wipe import wipe_speaker_artifacts

FILE_BYTES = b"test-data"


@pytest.fixture
def wipe_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[Path]]:
    journal = tmp_path / "journal"
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    import think.utils

    think.utils._journal_path_cache = None

    targets = [
        journal / "chronicle/20240101/test/120000_300/mic_audio.npz",
        # labels + corrections live under both agents/ (historical) and talents/
        # (current writes); the wipe must catch both.
        journal / "chronicle/20240101/test/120000_300/agents/speaker_labels.json",
        journal / "chronicle/20240101/test/120000_300/agents/speaker_corrections.json",
        journal / "chronicle/20240102/test/120000_300/talents/speaker_labels.json",
        journal / "chronicle/20240102/test/120000_300/talents/speaker_corrections.json",
        journal / "entities/alice/voiceprints.npz",
        journal / "entities/alice/owner_centroid.npz",
        journal / "entities/bob/owner_centroid.npz",
        journal / "awareness/owner_candidate.npz",
    ]

    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FILE_BYTES)

    yield journal, targets
    think.utils._journal_path_cache = None


def test_wipe_dry_run(wipe_journal: tuple[Path, list[Path]]) -> None:
    _, targets = wipe_journal

    report = wipe_speaker_artifacts(dry_run=True)

    assert report.segment_embeddings.count == 1
    assert report.speaker_labels.count == 2  # agents/ + talents/
    assert report.speaker_corrections.count == 2  # agents/ + talents/
    assert report.entity_voiceprints.count == 1
    assert report.owner_centroids.count == 2
    assert report.owner_candidate.count == 1
    assert report.total_files == 9
    assert report.total_bytes == 9 * len(FILE_BYTES)

    for path in targets:
        assert path.exists()


def test_wipe_commit(wipe_journal: tuple[Path, list[Path]]) -> None:
    _, targets = wipe_journal

    report = wipe_speaker_artifacts(dry_run=False)

    assert report.segment_embeddings.count == 1
    assert report.speaker_labels.count == 2  # agents/ + talents/
    assert report.speaker_corrections.count == 2  # agents/ + talents/
    assert report.entity_voiceprints.count == 1
    assert report.owner_centroids.count == 2
    assert report.owner_candidate.count == 1
    assert report.total_files == 9
    assert report.total_bytes == 9 * len(FILE_BYTES)

    for path in targets:
        assert not path.exists()
