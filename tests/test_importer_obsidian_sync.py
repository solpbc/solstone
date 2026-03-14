# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import glob
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest


def _write_note(
    vault_dir: Path,
    rel_path: str,
    content: str,
    mtime: float | None = None,
) -> Path:
    """Write a note file to the vault."""
    path = vault_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(path, (mtime, mtime))
    return path


SAMPLE_NOTE = dedent("""\
    ---
    tags: [project, alpha]
    ---
    # Alpha Project

    This is a note about the [[Alpha Project]].
    See also [[Bob Smith]] and [[Design Doc]].
""")

SAMPLE_NOTE_2 = dedent("""\
    # Daily Note

    Today I worked on [[Beta Launch]].
""")

UPDATED_NOTE = dedent("""\
    ---
    tags: [project, alpha]
    ---
    # Alpha Project

    This is an updated note about the [[Alpha Project]].
    See also [[Bob Smith]], [[Design Doc]], and [[Launch Plan]].
""")


def test_obsidian_sync_protocol_conformance():
    """ObsidianSyncBackend satisfies SyncableBackend protocol."""
    from think.importers.obsidian import ObsidianSyncBackend
    from think.importers.sync import SyncableBackend

    assert isinstance(ObsidianSyncBackend(), SyncableBackend)


def test_obsidian_sync_registry_discovery():
    """Registry discovery includes obsidian."""
    from think.importers.sync import get_syncable_backends

    backends = get_syncable_backends()
    assert "obsidian" in [backend.name for backend in backends]


def test_obsidian_sync_dry_run(tmp_path):
    """Dry-run catalogs notes and saves state."""
    from think.importers.obsidian import ObsidianSyncBackend
    from think.importers.sync import load_sync_state

    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)
    _write_note(vault, "Daily/2026-03-14.md", SAMPLE_NOTE_2, mtime=1_700_000_600)

    result = ObsidianSyncBackend().sync(tmp_path, source_path=vault, dry_run=True)

    assert result["total"] >= 2
    assert result["available"] == 2
    assert result["imported"] == 0
    assert result["downloaded"] == 0

    state = load_sync_state(tmp_path, "obsidian")
    assert state is not None
    assert state["files"]["Projects/Alpha.md"]["status"] == "available"
    assert state["files"]["Daily/2026-03-14.md"]["status"] == "available"


def test_obsidian_sync_import(tmp_path, monkeypatch):
    """Import mode writes note segments and updates state."""
    from think.importers.obsidian import ObsidianSyncBackend
    from think.importers.sync import load_sync_state

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    result = ObsidianSyncBackend().sync(tmp_path, source_path=vault, dry_run=False)

    assert result["downloaded"] >= 1
    assert result["imported"] >= 1

    segments = glob.glob(str(tmp_path / "*/import.obsidian/*/note_transcript.md"))
    assert len(segments) >= 1

    state = load_sync_state(tmp_path, "obsidian")
    assert state is not None
    assert state["files"]["Projects/Alpha.md"]["status"] == "imported"


def test_obsidian_sync_edit_creates_new_segments(tmp_path, monkeypatch):
    """Editing a note creates new segments and preserves old ones."""
    from think.importers.obsidian import ObsidianSyncBackend
    from think.importers.sync import load_sync_state

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    backend = ObsidianSyncBackend()
    first = backend.sync(tmp_path, source_path=vault, dry_run=False)
    assert first["downloaded"] == 1
    first_segments = sorted(
        glob.glob(str(tmp_path / "*/import.obsidian/*/note_transcript.md"))
    )
    assert len(first_segments) == 1

    _write_note(vault, "Projects/Alpha.md", UPDATED_NOTE, mtime=1_700_000_900)
    second = backend.sync(tmp_path, source_path=vault, dry_run=False)
    assert second["downloaded"] == 1

    all_segments = sorted(
        glob.glob(str(tmp_path / "*/import.obsidian/*/note_transcript.md"))
    )
    assert len(all_segments) == 2
    assert first_segments[0] in all_segments

    state = load_sync_state(tmp_path, "obsidian")
    assert state is not None
    assert state["files"]["Projects/Alpha.md"]["edit_count"] >= 2


def test_obsidian_sync_unchanged_skip(tmp_path, monkeypatch):
    """Mtime-only changes are skipped when content hash matches."""
    from think.importers.obsidian import ObsidianSyncBackend

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    backend = ObsidianSyncBackend()
    backend.sync(tmp_path, source_path=vault, dry_run=False)
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_300)

    result = backend.sync(tmp_path, source_path=vault, dry_run=True)
    assert result["available"] == 0


def test_obsidian_sync_deleted_note(tmp_path, monkeypatch):
    """Deleted notes are marked removed in state."""
    from think.importers.obsidian import ObsidianSyncBackend
    from think.importers.sync import load_sync_state

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    note = _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    backend = ObsidianSyncBackend()
    backend.sync(tmp_path, source_path=vault, dry_run=False)
    note.unlink()
    backend.sync(tmp_path, source_path=vault, dry_run=True)

    state = load_sync_state(tmp_path, "obsidian")
    assert state is not None
    assert state["files"]["Projects/Alpha.md"]["status"] == "removed"


def test_obsidian_sync_force(tmp_path, monkeypatch):
    """Force re-detects notes by clearing state."""
    from think.importers.obsidian import ObsidianSyncBackend

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    backend = ObsidianSyncBackend()
    backend.sync(tmp_path, source_path=vault, dry_run=False)
    result = backend.sync(tmp_path, source_path=vault, dry_run=True, force=True)

    assert result["available"] >= 1


def test_obsidian_sync_vault_auto_detection(tmp_path, monkeypatch):
    """Raises when no vault can be auto-detected."""
    from think.importers.obsidian import ObsidianSyncBackend

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("think.importers.obsidian.Path.home", lambda: home)

    with pytest.raises(
        ValueError,
        match="No Obsidian vault found. Use --path to specify your vault location.",
    ):
        ObsidianSyncBackend().sync(tmp_path)


def test_obsidian_sync_entity_seeding(tmp_path, monkeypatch):
    """Wikilinks are converted into Topic entities on import."""
    from think.importers.obsidian import ObsidianSyncBackend

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    captured: list[tuple[str, str, list[dict[str, str]]]] = []

    def _fake_seed_entities(
        facet: str,
        day: str,
        entities: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        captured.append((facet, day, entities))
        return entities

    with patch(
        "think.importers.obsidian.seed_entities", side_effect=_fake_seed_entities
    ):
        ObsidianSyncBackend().sync(tmp_path, source_path=vault, dry_run=False)

    assert len(captured) == 1
    facet, _day, entities = captured[0]
    assert facet == "import.obsidian"
    assert entities == [
        {"name": "Alpha Project", "type": "Topic"},
        {"name": "Bob Smith", "type": "Topic"},
        {"name": "Design Doc", "type": "Topic"},
    ]


def test_obsidian_sync_incremental(tmp_path, monkeypatch):
    """Incremental sync imports only newly added notes."""
    from think.importers.obsidian import ObsidianSyncBackend

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    vault = tmp_path / "vault"
    _write_note(vault, "Projects/Alpha.md", SAMPLE_NOTE, mtime=1_700_000_000)

    backend = ObsidianSyncBackend()
    first = backend.sync(tmp_path, source_path=vault, dry_run=False)
    assert first["downloaded"] == 1

    _write_note(vault, "Daily/2026-03-14.md", SAMPLE_NOTE_2, mtime=1_700_000_600)
    second = backend.sync(tmp_path, source_path=vault, dry_run=False)

    assert second["downloaded"] == 1
    assert second["available"] == 0
    assert second["imported"] >= 1


def test_obsidian_backends_cli_flag(capsys, monkeypatch):
    """sol import --backends lists obsidian."""
    import sys

    from think.importers.cli import main

    monkeypatch.setattr(sys, "argv", ["sol import", "--backends"])
    monkeypatch.setenv("JOURNAL_PATH", "/tmp/test-journal")

    main()
    captured = capsys.readouterr()
    assert "obsidian" in captured.out
