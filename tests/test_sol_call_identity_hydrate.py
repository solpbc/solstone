# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import os
import subprocess
import sys

import pytest

from think.tools.sol import _SPECIES_PREAMBLE


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


def test_identity_hydrate_starts_with_species_preamble(journal_path):
    sol_dir = journal_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "self.md").write_text("self body")
    (sol_dir / "partner.md").write_text("partner body")
    (sol_dir / "agency.md").write_text("agency body")
    (sol_dir / "awareness.md").write_text("awareness body")

    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    assert result.stdout.startswith("# species\n\n")
    assert _SPECIES_PREAMBLE in result.stdout
    expected = ["# species", "# self", "# partner", "# agency", "# awareness"]
    positions = [result.stdout.index(marker) for marker in expected]
    assert positions == sorted(positions)


def test_identity_hydrate_strips_duplicate_section_heading(journal_path):
    sol_dir = journal_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "self.md").write_text("# self\n\nself body\n")
    (sol_dir / "partner.md").write_text("partner body")
    (sol_dir / "agency.md").write_text("agency body")
    (sol_dir / "awareness.md").write_text("awareness body")

    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    assert result.stdout.splitlines().count("# self") == 1
    assert "# self\n\nself body" in result.stdout


def test_identity_hydrate_preserves_non_matching_heading(journal_path):
    sol_dir = journal_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "self.md").write_text("# My Custom Heading\n\nself body\n")
    (sol_dir / "partner.md").write_text("partner body")
    (sol_dir / "agency.md").write_text("agency body")
    (sol_dir / "awareness.md").write_text("awareness body")

    result = _run_identity_hydrate(journal_path)

    assert result.returncode == 0
    assert "# My Custom Heading" in result.stdout
    assert result.stdout.splitlines().count("# self") == 1
    assert "self body" in result.stdout
