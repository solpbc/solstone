# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def journal_path(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "journal.json").write_text(json.dumps({}))
    return tmp_path


def _run_identity_hydrate(journal_path):
    env = os.environ.copy()
    env.update(
        {
            "_SOLSTONE_JOURNAL_OVERRIDE": str(journal_path),
            "SOL_SKIP_SUPERVISOR_CHECK": "1",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", "from think.tools.sol import app; app()"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_identity_hydrate_reads_all_sections(journal_path):
    sol_dir = journal_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "self.md").write_text("self body")
    (sol_dir / "partner.md").write_text("partner body")
    (sol_dir / "agency.md").write_text("agency body")
    (sol_dir / "awareness.md").write_text("awareness body")

    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    expected = ["# self", "# partner", "# agency", "# awareness"]
    positions = [result.stdout.index(marker) for marker in expected]
    assert positions == sorted(positions)
    assert "self body" in result.stdout
    assert "partner body" in result.stdout
    assert "agency body" in result.stdout
    assert "awareness body" in result.stdout


def test_identity_hydrate_marks_missing_sections(journal_path):
    sol_dir = journal_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "self.md").write_text("self body")
    (sol_dir / "partner.md").write_text("partner body")
    (sol_dir / "awareness.md").write_text("awareness body")

    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    assert "# agency\n\n(not present)\n" in result.stdout


def test_identity_hydrate_handles_empty_sol_directory(journal_path):
    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    for stem in ("self", "partner", "agency", "awareness"):
        assert f"# {stem}\n\n(not present)\n" in result.stdout
