# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Wipe legacy speaker artifacts from the journal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from think.utils import get_journal


@dataclass
class WipeCategory:
    count: int = 0
    bytes: int = 0
    paths: list[str] = field(default_factory=list)


@dataclass
class WipeReport:
    segment_embeddings: WipeCategory = field(default_factory=WipeCategory)
    speaker_labels: WipeCategory = field(default_factory=WipeCategory)
    speaker_corrections: WipeCategory = field(default_factory=WipeCategory)
    entity_voiceprints: WipeCategory = field(default_factory=WipeCategory)
    owner_centroids: WipeCategory = field(default_factory=WipeCategory)
    owner_candidate: WipeCategory = field(default_factory=WipeCategory)
    total_files: int = 0
    total_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def wipe_speaker_artifacts(dry_run: bool = True) -> WipeReport:
    """Remove legacy speaker artifacts from the journal."""
    journal = Path(get_journal())
    report = WipeReport()

    # Labels and corrections currently live under agents/ for most historical
    # segments; the attribution module now writes new files under talents/.
    # Cover both paths so no resemblyzer-space artifacts leak through the wipe.
    label_patterns = [
        "chronicle/*/*/*/agents/speaker_labels.json",
        "chronicle/*/*/*/talents/speaker_labels.json",
    ]
    correction_patterns = [
        "chronicle/*/*/*/agents/speaker_corrections.json",
        "chronicle/*/*/*/talents/speaker_corrections.json",
    ]

    def _expand(patterns: list[str]) -> list[Path]:
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(journal.glob(pattern))
        return sorted(paths)

    categories: list[tuple[WipeCategory, list[Path]]] = [
        (report.segment_embeddings, sorted(journal.glob("chronicle/*/*/*/*.npz"))),
        (report.speaker_labels, _expand(label_patterns)),
        (report.speaker_corrections, _expand(correction_patterns)),
        (report.entity_voiceprints, sorted(journal.glob("entities/*/voiceprints.npz"))),
        (report.owner_centroids, sorted(journal.glob("entities/*/owner_centroid.npz"))),
        (report.owner_candidate, [journal / "awareness" / "owner_candidate.npz"]),
    ]

    for category, paths in categories:
        for path in paths:
            if not path.is_file():
                continue
            category.count += 1
            category.bytes += path.stat().st_size
            category.paths.append(path.relative_to(journal).as_posix())
            if not dry_run:
                path.unlink()

    tracked_categories = [
        report.segment_embeddings,
        report.speaker_labels,
        report.speaker_corrections,
        report.entity_voiceprints,
        report.owner_centroids,
        report.owner_candidate,
    ]
    report.total_files = sum(category.count for category in tracked_categories)
    report.total_bytes = sum(category.bytes for category in tracked_categories)
    return report
