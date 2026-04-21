# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Fixtures for transcripts app tests."""

import os
import sys
from pathlib import Path

import pytest

from convey import create_app

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._baseline_harness import copytree_tracked


@pytest.fixture(autouse=True)
def _journal_env(request, monkeypatch):
    """Point tests at a copied journal when needed, otherwise the tracked fixture."""
    if "journal_copy" in request.fixturenames:
        journal_copy = request.getfixturevalue("journal_copy")
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_copy))
        return

    monkeypatch.setenv(
        "_SOLSTONE_JOURNAL_OVERRIDE",
        os.path.join(os.getcwd(), "tests", "fixtures", "journal"),
    )


@pytest.fixture
def client(journal_copy, monkeypatch):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_copy))
    app = create_app(str(journal_copy))
    return app.test_client()


@pytest.fixture
def journal_copy(tmp_path, monkeypatch):
    src = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "journal"
    dst = tmp_path / "journal"
    copytree_tracked(src, dst)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst.resolve()))
    return dst
