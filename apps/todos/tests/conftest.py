"""Self-contained fixtures for todos app tests.

These fixtures are fully standalone and only depend on pytest builtins.
No shared dependencies from the root conftest.py are required.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest


@pytest.fixture
def todo_env(tmp_path, monkeypatch):
    """Create a temporary journal facet with optional todo entries.

    This fixture is self-contained and provides a complete test environment
    for todo operations without any external dependencies.

    Usage:
        def test_example(todo_env):
            day, facet, todo_path = todo_env(["- [ ] First item"])
            # Now JOURNAL_PATH is set and todo file exists
    """

    def _create(
        entries: list[str] | None = None,
        day: str | None = None,
        facet: str = "personal",
    ):
        if day is None:
            day = datetime.now().strftime("%Y%m%d")
        todos_dir = tmp_path / "facets" / facet / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        todo_path = todos_dir / f"{day}.md"
        if entries is not None:
            todo_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
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

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        return journal, facet

    return _create
