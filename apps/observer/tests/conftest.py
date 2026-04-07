# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained fixtures for observer app tests.

These fixtures are fully standalone and only depend on pytest builtins.
No shared dependencies from the root conftest.py are required.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def observer_env(tmp_path, monkeypatch):
    """Create a temporary journal for observer app testing.

    Returns a factory function that sets up the environment and returns
    the Flask test client along with the journal path.
    """

    def _create():
        journal = tmp_path / "journal"
        journal.mkdir()

        # Set environment
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

        # Create Flask test client
        from convey import create_app

        app = create_app(journal=str(journal))
        client = app.test_client()

        class Env:
            def __init__(self):
                self.journal = journal
                self.client = client
                self.app = app

        return Env()

    return _create
