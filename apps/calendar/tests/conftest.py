# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for calendar app tests."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def calendar_env(tmp_path, monkeypatch):
    """Create a temporary journal facet with optional calendar entries."""

    def _create(
        entries: list[dict] | None = None,
        day: str = "20240101",
        facet: str = "work",
    ):
        calendar_dir = tmp_path / "facets" / facet / "calendar"
        calendar_dir.mkdir(parents=True, exist_ok=True)
        calendar_path = calendar_dir / f"{day}.jsonl"
        if entries is not None:
            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            calendar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        monkeypatch.setenv("SOL_DAY", day)
        monkeypatch.setenv("SOL_FACET", facet)
        return day, facet, calendar_path

    return _create


@pytest.fixture
def facet_env(tmp_path, monkeypatch):
    """Create a temporary facet with full structure for testing."""
    journal = tmp_path / "journal"
    journal.mkdir()

    def _create(facet: str = "test_facet"):
        facet_path = journal / "facets" / facet
        facet_path.mkdir(parents=True)

        facet_json = facet_path / "facet.json"
        facet_json.write_text(
            json.dumps({"title": f"Test {facet}", "description": "Test facet"}),
            encoding="utf-8",
        )

        (facet_path / "calendar").mkdir()
        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        monkeypatch.setenv("SOL_FACET", facet)
        return journal, facet

    return _create


@pytest.fixture
def move_env(tmp_path, monkeypatch):
    """Create a two-facet environment for move tests."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    def _create(
        entries: list[dict] | None = None,
        day: str = "20240101",
        src_facet: str = "work",
        dst_facet: str = "personal",
    ):
        for facet in [src_facet, dst_facet]:
            facet_dir = tmp_path / "facets" / facet
            facet_dir.mkdir(parents=True, exist_ok=True)
            (facet_dir / "facet.json").write_text(
                json.dumps({"title": f"Test {facet}", "description": "Test facet"}),
                encoding="utf-8",
            )

        calendar_dir = tmp_path / "facets" / src_facet / "calendar"
        calendar_dir.mkdir(parents=True, exist_ok=True)
        calendar_path = calendar_dir / f"{day}.jsonl"
        if entries:
            lines = []
            for entry in entries:
                lines.append(json.dumps(entry, ensure_ascii=False))
            calendar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return tmp_path, src_facet, dst_facet

    return _create
