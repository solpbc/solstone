# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for todos app tests.

These fixtures are fully standalone and only depend on pytest builtins.
No shared dependencies from the root conftest.py are required.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest


@pytest.fixture(autouse=True)
def _skip_supervisor_check(monkeypatch):
    """Allow app CLI tests to run without a live solstone supervisor."""
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")


@pytest.fixture
def todo_env(tmp_path, monkeypatch):
    """Create a temporary journal facet with optional todo entries.

    This fixture is self-contained and provides a complete test environment
    for todo operations without any external dependencies.

    Usage:
        def test_example(todo_env):
            day, facet, todo_path = todo_env([
                {"text": "First item"},
                {"text": "Second item", "completed": True}
            ])
            # Now _SOLSTONE_JOURNAL_OVERRIDE is set and todo file exists
    """

    def _create(
        entries: list[dict] | None = None,
        day: str | None = None,
        facet: str = "personal",
    ):
        if day is None:
            day = datetime.now().strftime("%Y%m%d")
        todos_dir = tmp_path / "facets" / facet / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        todo_path = todos_dir / f"{day}.jsonl"
        if entries is not None:
            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        return day, facet, todo_path

    return _create


@pytest.fixture
def facet_env(tmp_path, monkeypatch):
    """Create a temporary facet with full structure for testing.

    Includes facet.json configuration, todos directory, and logs directory.
    Suitable for testing action logging and other facet-scoped operations.
    """
    journal = tmp_path / "journal"
    journal.mkdir()

    def _create(facet: str = "test_facet"):
        facet_path = journal / "facets" / facet
        facet_path.mkdir(parents=True)

        # Create facet.json
        facet_json = facet_path / "facet.json"
        facet_json.write_text(
            json.dumps({"title": f"Test {facet}", "description": "Test facet"}),
            encoding="utf-8",
        )

        # Create todos directory
        (facet_path / "todos").mkdir()

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
        return journal, facet

    return _create


@pytest.fixture
def move_env(tmp_path, monkeypatch):
    """Create a two-facet environment for move tests."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

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

        todos_dir = tmp_path / "facets" / src_facet / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        todo_path = todos_dir / f"{day}.jsonl"
        if entries:
            now_ms = int(datetime.now().timestamp() * 1000)
            lines = []
            for entry in entries:
                data = {
                    "text": entry["text"],
                    "created_at": entry.get("created_at", now_ms),
                    "updated_at": entry.get("updated_at", now_ms),
                }
                if entry.get("cancelled"):
                    data["cancelled"] = True
                if entry.get("completed"):
                    data["completed"] = True
                if entry.get("nudge"):
                    data["nudge"] = entry["nudge"]
                lines.append(json.dumps(data, ensure_ascii=False))
            todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return tmp_path, src_facet, dst_facet

    return _create
