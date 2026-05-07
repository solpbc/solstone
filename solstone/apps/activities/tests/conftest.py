# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

import pytest


@pytest.fixture
def activities_env(tmp_path, monkeypatch):
    """Create a temporary journal facet with activity config and day records."""

    def _create(
        entries: list[dict] | None = None,
        *,
        day: str = "20240101",
        facet: str = "work",
        activity_config: list[dict] | None = None,
    ):
        facet_dir = tmp_path / "facets" / facet
        activities_dir = facet_dir / "activities"
        activities_dir.mkdir(parents=True, exist_ok=True)

        (facet_dir / "facet.json").write_text(
            json.dumps({"title": f"Test {facet}", "description": "Test facet"}) + "\n",
            encoding="utf-8",
        )

        config_entries = activity_config or [{"id": "coding"}, {"id": "meeting"}]
        (activities_dir / "activities.jsonl").write_text(
            "".join(
                json.dumps(entry, ensure_ascii=False) + "\n" for entry in config_entries
            ),
            encoding="utf-8",
        )

        day_path = activities_dir / f"{day}.jsonl"
        if entries is not None:
            day_path.write_text(
                "".join(
                    json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries
                ),
                encoding="utf-8",
            )

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
        monkeypatch.setenv("SOL_DAY", day)
        monkeypatch.setenv("SOL_FACET", facet)
        monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
        return tmp_path, facet, day, day_path

    return _create
