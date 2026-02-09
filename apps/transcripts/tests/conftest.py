# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Fixtures for transcripts app tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _journal_env(monkeypatch):
    """Point JOURNAL_PATH at the test fixtures."""
    monkeypatch.setenv(
        "JOURNAL_PATH", os.path.join(os.getcwd(), "tests", "fixtures", "journal")
    )
