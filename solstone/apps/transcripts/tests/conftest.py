# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Fixtures for transcripts app tests."""

import os
import sys
from pathlib import Path

import pytest

from solstone.think.utils import get_project_root

ROOT = Path(get_project_root())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for name, module in list(sys.modules.items()):
    if name.split(".", 1)[0] != "solstone":
        continue
    module_file = getattr(module, "__file__", None)
    if module_file and not Path(module_file).resolve().is_relative_to(ROOT):
        sys.modules.pop(name, None)

from solstone.convey import create_app
from tests._baseline_harness import copytree_tracked


@pytest.fixture(autouse=True)
def _journal_env(request, monkeypatch):
    """Point tests at a copied journal when needed, otherwise the tracked fixture."""
    if "journal_copy" in request.fixturenames:
        journal_copy = request.getfixturevalue("journal_copy")
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_copy))
        return

    monkeypatch.setenv(
        "SOLSTONE_JOURNAL",
        os.path.join(os.getcwd(), "tests", "fixtures", "journal"),
    )


@pytest.fixture
def client(journal_copy, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_copy))
    app = create_app(str(journal_copy))
    return app.test_client()


@pytest.fixture
def journal_copy(tmp_path, monkeypatch):
    src = Path(get_project_root()) / "tests" / "fixtures" / "journal"
    dst = tmp_path / "journal"
    copytree_tracked(src, dst)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(dst.resolve()))
    return dst
