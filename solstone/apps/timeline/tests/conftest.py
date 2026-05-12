# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _skip_supervisor_check(monkeypatch):
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")


@pytest.fixture(autouse=True)
def _reset_caches():
    from solstone.apps.timeline import routes

    routes._master_cache = None
    routes._master_key = None
    routes._seg_cache.clear()
    yield
    routes._master_cache = None
    routes._master_key = None
    routes._seg_cache.clear()


@pytest.fixture
def timeline_env(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    journal.mkdir()
    shutil.copytree(FIXTURES / "chronicle_layout", journal / "chronicle")
    shutil.copy2(FIXTURES / "master_small.json", journal / "timeline.json")

    facet_dir = journal / "facets" / "work"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Work", "description": "Test facet"}) + "\n",
        encoding="utf-8",
    )
    (journal / "config").mkdir()
    (journal / "config" / "journal.json").write_text(
        json.dumps(
            {
                "convey": {"secret": "test-secret", "trust_localhost": True},
                "setup": {"completed_at": 1700000000000},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    return journal


@pytest.fixture
def client(timeline_env):
    from solstone.convey import create_app

    app = create_app(str(timeline_env))
    app.config.update(TESTING=True)
    return app.test_client()
