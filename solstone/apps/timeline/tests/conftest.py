# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Fixtures for timeline app tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

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


@pytest.fixture
def timeline_journal(tmp_path, monkeypatch) -> Path:
    journal = tmp_path / "journal"
    (journal / "chronicle").mkdir(parents=True)
    (journal / "config").mkdir(parents=True)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

    import solstone.think.utils as think_utils

    think_utils._journal_path_cache = None
    return journal


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def mock_agenerate(monkeypatch):
    def _install(*payloads: dict | Exception):
        responses = list(payloads)

        async def _fake_agenerate(**kwargs):
            if not responses:
                return json.dumps({"picks": [0], "rationale": "default"})
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return json.dumps(item)

        mock = AsyncMock(side_effect=_fake_agenerate)
        monkeypatch.setattr("solstone.think.batch.agenerate", mock)
        return mock

    return _install
