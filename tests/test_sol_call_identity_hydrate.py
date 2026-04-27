# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

import pytest

from think.tools.sol import _SPECIES_PREAMBLE, _hydrate


@pytest.fixture
def journal_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "journal.json").write_text(json.dumps({}))
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    return tmp_path


def test_identity_hydrate_reads_all_sections(journal_path):
    identity_dir = journal_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "self.md").write_text("self body")
    (identity_dir / "partner.md").write_text("partner body")
    (identity_dir / "agency.md").write_text("agency body")
    (identity_dir / "awareness.md").write_text("awareness body")

    output = _hydrate()

    expected = ["# self", "# partner", "# agency", "# awareness"]
    positions = [output.index(marker) for marker in expected]
    assert positions == sorted(positions)
    assert "self body" in output
    assert "partner body" in output
    assert "agency body" in output
    assert "awareness body" in output


def test_identity_hydrate_marks_missing_sections(journal_path):
    identity_dir = journal_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "self.md").write_text("self body")
    (identity_dir / "partner.md").write_text("partner body")
    (identity_dir / "awareness.md").write_text("awareness body")

    output = _hydrate()

    assert "# agency\n\n(not present)\n" in output


def test_identity_hydrate_handles_empty_identity_directory(journal_path):
    output = _hydrate()

    for stem in ("self", "partner", "agency", "awareness"):
        assert f"# {stem}\n\n(not present)\n" in output


def test_identity_hydrate_starts_with_species_preamble(journal_path):
    identity_dir = journal_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "self.md").write_text("self body")
    (identity_dir / "partner.md").write_text("partner body")
    (identity_dir / "agency.md").write_text("agency body")
    (identity_dir / "awareness.md").write_text("awareness body")

    output = _hydrate()

    assert output.startswith("# species\n\n")
    assert _SPECIES_PREAMBLE in output
    expected = ["# species", "# self", "# partner", "# agency", "# awareness"]
    positions = [output.index(marker) for marker in expected]
    assert positions == sorted(positions)


def test_identity_hydrate_strips_duplicate_section_heading(journal_path):
    identity_dir = journal_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "self.md").write_text("# self\n\nself body\n")
    (identity_dir / "partner.md").write_text("partner body")
    (identity_dir / "agency.md").write_text("agency body")
    (identity_dir / "awareness.md").write_text("awareness body")

    output = _hydrate()

    assert output.splitlines().count("# self") == 1
    assert "# self\n\nself body" in output


def test_identity_hydrate_preserves_non_matching_heading(journal_path):
    identity_dir = journal_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "self.md").write_text("# My Custom Heading\n\nself body\n")
    (identity_dir / "partner.md").write_text("partner body")
    (identity_dir / "agency.md").write_text("agency body")
    (identity_dir / "awareness.md").write_text("awareness body")

    output = _hydrate()

    assert "# My Custom Heading" in output
    assert output.splitlines().count("# self") == 1
    assert "self body" in output
