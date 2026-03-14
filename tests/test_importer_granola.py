# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the Granola meeting transcript importer (via muesli)."""

import json
from pathlib import Path
from textwrap import dedent

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = dedent("""\
    ---
    doc_id: doc_test_001
    source: granola
    created_at: "2025-10-28T15:04:05Z"
    remote_updated_at: "2025-10-29T01:23:45Z"
    title: Q1 Planning
    participants:
      - Alice
      - Bob
    duration_seconds: 3170
    labels:
      - Planning
    creator: "[[Alice Smith]]"
    attendees:
      - "[[Alice Smith]]"
      - "[[Bob Jones]]"
    summary_text: "Discussed Q1 priorities and resource allocation."
    generator: muesli 1.0
    ---

    # Q1 Planning

    _Date: 2025-10-28 · Duration: 52m · Participants: Alice Smith, Bob Jones_

    ## Participants

    - **Alice Smith**, Engineering Manager, Acme Corp, (alice@acme.com)
    - **Bob Jones**, (bob@jones.io)

    ---

    **Alice Smith (15:04:12):** Welcome to the planning session.
    **Bob Jones (15:04:19):** Thanks, let's get started.
    **Alice Smith (15:04:35):** First, let's review our Q4 results.
    **Bob Jones (15:05:01):** Revenue was up 15% quarter over quarter.
    **Alice Smith (15:09:22):** Now let's talk about Q1 priorities.
    **Bob Jones (15:09:45):** I think we should focus on the mobile launch.
""")


STUB_TRANSCRIPT = dedent("""\
    ---
    doc_id: doc_stub_002
    source: granola
    created_at: "2025-11-01T10:00:00Z"
    remote_updated_at: "2025-11-01T10:00:00Z"
    title: Quick Sync
    duration_seconds: 60
    generator: muesli 1.0
    ---

    # Quick Sync

    _No transcript available._
""")


NO_DOCID_TRANSCRIPT = dedent("""\
    ---
    source: granola
    title: Orphan
    ---

    # Orphan

    **Alice (10:00:00):** Hello.
""")


def _write_transcript(muesli_dir: Path, filename: str, content: str) -> Path:
    """Write a transcript file to the muesli directory."""
    muesli_dir.mkdir(parents=True, exist_ok=True)
    path = muesli_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def test_parse_muesli_file(tmp_path):
    """Parse frontmatter and body from a muesli markdown file."""
    from think.importers.granola import _parse_muesli_file

    path = _write_transcript(tmp_path, "test.md", SAMPLE_TRANSCRIPT)
    fm, body = _parse_muesli_file(path)

    assert fm["doc_id"] == "doc_test_001"
    assert fm["title"] == "Q1 Planning"
    assert fm["duration_seconds"] == 3170
    assert fm["summary_text"] == "Discussed Q1 priorities and resource allocation."
    assert "Alice Smith (15:04:12)" in body


def test_parse_muesli_file_missing_fields(tmp_path):
    """Handles files with minimal frontmatter."""
    from think.importers.granola import _parse_muesli_file

    content = dedent("""\
        ---
        doc_id: minimal
        ---

        **Speaker (10:00:00):** Hello.
    """)
    path = _write_transcript(tmp_path, "minimal.md", content)
    fm, body = _parse_muesli_file(path)

    assert fm["doc_id"] == "minimal"
    assert "title" not in fm
    assert "Speaker (10:00:00)" in body


# ---------------------------------------------------------------------------
# Participant parsing
# ---------------------------------------------------------------------------


def test_parse_participants():
    """Extract participant info from ## Participants section."""
    from think.importers.granola import _parse_participants

    participants = _parse_participants(SAMPLE_TRANSCRIPT)
    assert len(participants) == 2

    alice = participants[0]
    assert alice["name"] == "Alice Smith"
    assert alice["email"] == "alice@acme.com"
    assert alice["title"] == "Engineering Manager"
    assert alice["company"] == "Acme Corp"

    bob = participants[1]
    assert bob["name"] == "Bob Jones"
    assert bob["email"] == "bob@jones.io"


def test_parse_participants_with_linkedin():
    """Extract LinkedIn handle from participant line."""
    from think.importers.granola import _parse_participants

    body = dedent("""\
        ## Participants

        - **Jane Doe**, CTO, StartupCo, linkedin.com/in/janedoe, (jane@startup.co)
    """)
    participants = _parse_participants(body)
    assert len(participants) == 1
    assert participants[0]["name"] == "Jane Doe"
    assert participants[0]["email"] == "jane@startup.co"
    assert participants[0]["linkedin"] == "janedoe"
    assert participants[0]["title"] == "CTO"
    assert participants[0]["company"] == "StartupCo"


def test_parse_participants_no_section():
    """Returns empty list when no ## Participants section exists."""
    from think.importers.granola import _parse_participants

    assert _parse_participants("# Meeting\n\nJust some text.") == []


def test_parse_participants_wikilinks():
    """Strips [[wikilink]] brackets from participant names."""
    from think.importers.granola import _parse_participants

    body = dedent("""\
        ## Participants

        - **[[Alice Smith]]**, (alice@acme.com)
    """)
    participants = _parse_participants(body)
    assert participants[0]["name"] == "Alice Smith"


# ---------------------------------------------------------------------------
# Transcript entry parsing
# ---------------------------------------------------------------------------


def test_parse_transcript_entries():
    """Parse speaker-labeled entries into timestamped messages."""
    import datetime as dt

    from think.importers.granola import _parse_transcript_entries

    base_dt = dt.datetime(2025, 10, 28, 0, 0, 0, tzinfo=dt.timezone.utc)
    messages = _parse_transcript_entries(SAMPLE_TRANSCRIPT, base_dt, dt.timezone.utc)

    assert len(messages) == 6
    assert messages[0]["speaker"] == "Alice Smith"
    assert messages[0]["text"] == "Welcome to the planning session."
    assert messages[0]["model_slug"] is None

    assert messages[1]["speaker"] == "Bob Jones"
    assert messages[1]["text"] == "Thanks, let's get started."

    # Timestamps should be in order
    for i in range(1, len(messages)):
        assert messages[i]["create_time"] >= messages[i - 1]["create_time"]


def test_parse_transcript_entries_empty_body():
    """Returns empty list when no transcript entries found."""
    import datetime as dt

    from think.importers.granola import _parse_transcript_entries

    base_dt = dt.datetime(2025, 10, 28, 0, 0, 0, tzinfo=dt.timezone.utc)
    messages = _parse_transcript_entries("No transcript here.", base_dt, dt.timezone.utc)
    assert messages == []


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def test_date_from_filename():
    """Extract date from muesli filename."""
    from think.importers.granola import _date_from_filename

    import datetime as dt

    assert _date_from_filename("2025-10-28_q1-planning.md") == dt.date(2025, 10, 28)
    assert _date_from_filename("random-file.md") is None
    assert _date_from_filename("2025-01-01.md") == dt.date(2025, 1, 1)


def test_parse_created_at():
    """Parse created_at from frontmatter with timezone."""
    import datetime as dt

    from think.importers.granola import _parse_created_at

    fm = {"created_at": "2025-10-28T15:04:05Z"}
    created_dt, tz = _parse_created_at(fm, "test.md")
    assert created_dt is not None
    assert tz == dt.timezone.utc


def test_parse_created_at_fallback_to_filename():
    """Falls back to filename date when created_at is missing."""
    from think.importers.granola import _parse_created_at

    fm = {}
    created_dt, tz = _parse_created_at(fm, "2025-10-28_meeting.md")
    assert created_dt is not None
    assert tz is None  # naive/local


def test_parse_created_at_no_date():
    """Returns None when no date source available."""
    from think.importers.granola import _parse_created_at

    fm = {}
    created_dt, tz = _parse_created_at(fm, "random.md")
    assert created_dt is None


# ---------------------------------------------------------------------------
# GranolaBackend protocol
# ---------------------------------------------------------------------------


def test_granola_protocol_conformance():
    """GranolaBackend satisfies SyncableBackend protocol."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import SyncableBackend

    assert isinstance(GranolaBackend(), SyncableBackend)


def test_granola_in_registry():
    """Granola is registered in the syncable backend registry."""
    from think.importers.sync import get_syncable_backends

    backends = get_syncable_backends()
    names = [b.name for b in backends]
    assert "granola" in names


# ---------------------------------------------------------------------------
# GranolaBackend.sync() — detection
# ---------------------------------------------------------------------------


def test_granola_sync_no_muesli(tmp_path):
    """Raises ValueError when muesli is not installed."""
    from think.importers.granola import GranolaBackend

    nonexistent = tmp_path / "nonexistent" / "transcripts"
    with pytest.raises(ValueError, match="muesli to extract"):
        GranolaBackend().sync(tmp_path, source_path=nonexistent)


def test_granola_sync_no_transcripts(tmp_path):
    """Raises ValueError when muesli dir exists but no transcripts."""
    from think.importers.granola import GranolaBackend

    muesli_dir = tmp_path / "muesli" / "transcripts"
    # Create parent but not transcripts dir
    (tmp_path / "muesli").mkdir()
    with pytest.raises(ValueError, match="no transcripts found"):
        GranolaBackend().sync(tmp_path, source_path=muesli_dir)


# ---------------------------------------------------------------------------
# GranolaBackend.sync() — catalog mode
# ---------------------------------------------------------------------------


def test_granola_sync_dry_run(tmp_path):
    """Dry-run catalogs transcripts and saves state."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import load_sync_state

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)
    _write_transcript(muesli_dir, "stub.md", STUB_TRANSCRIPT)

    result = GranolaBackend().sync(tmp_path, source_path=muesli_dir)

    assert result["total"] == 2
    assert result["available"] == 1  # only the one with transcript content
    assert result["skipped"] == 1  # stub has no transcript
    assert result["imported"] == 0
    assert result["downloaded"] == 0

    # State was saved
    state = load_sync_state(tmp_path, "granola")
    assert state is not None
    assert len(state["files"]) == 2
    assert state["files"]["doc_test_001"]["status"] == "available"
    assert state["files"]["doc_stub_002"]["status"] == "skipped"
    assert state["files"]["doc_stub_002"]["skip_reason"] == "no_transcript"


def test_granola_sync_skips_no_docid(tmp_path):
    """Transcripts without doc_id in frontmatter are skipped."""
    from think.importers.granola import GranolaBackend

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "orphan.md", NO_DOCID_TRANSCRIPT)

    result = GranolaBackend().sync(tmp_path, source_path=muesli_dir)
    assert result["total"] == 0
    assert result["available"] == 0


# ---------------------------------------------------------------------------
# GranolaBackend.sync() — incremental
# ---------------------------------------------------------------------------


def test_granola_sync_incremental(tmp_path):
    """Second sync skips already-imported transcripts."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import load_sync_state, save_sync_state

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)

    # Pre-seed state: doc_test_001 already imported
    save_sync_state(
        tmp_path,
        "granola",
        {
            "backend": "granola",
            "files": {
                "doc_test_001": {
                    "filename": "2025-10-28_q1.md",
                    "remote_updated_at": "2025-10-29T01:23:45Z",
                    "status": "imported",
                    "imported_at": "2026-03-01T00:00:00",
                },
            },
        },
    )

    result = GranolaBackend().sync(tmp_path, source_path=muesli_dir)

    assert result["imported"] == 1
    assert result["available"] == 0  # nothing new


def test_granola_sync_detects_updated(tmp_path):
    """Re-imports when remote_updated_at is newer than last sync."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import save_sync_state

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)

    # Pre-seed with older remote_updated_at
    save_sync_state(
        tmp_path,
        "granola",
        {
            "backend": "granola",
            "files": {
                "doc_test_001": {
                    "filename": "2025-10-28_q1.md",
                    "remote_updated_at": "2025-10-28T00:00:00Z",  # older
                    "status": "imported",
                },
            },
        },
    )

    result = GranolaBackend().sync(tmp_path, source_path=muesli_dir)
    assert result["available"] == 1  # updated, needs re-import


def test_granola_sync_detects_removed(tmp_path):
    """Marks files as removed when they disappear from muesli dir."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import load_sync_state, save_sync_state

    muesli_dir = tmp_path / "muesli"
    muesli_dir.mkdir(parents=True)
    # No files in dir, but state has one

    save_sync_state(
        tmp_path,
        "granola",
        {
            "backend": "granola",
            "files": {
                "doc_gone": {
                    "filename": "old.md",
                    "status": "imported",
                },
            },
        },
    )

    GranolaBackend().sync(tmp_path, source_path=muesli_dir)
    state = load_sync_state(tmp_path, "granola")
    assert state["files"]["doc_gone"]["status"] == "removed"


def test_granola_sync_force(tmp_path):
    """Force flag clears state and re-imports everything."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import save_sync_state

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)

    # Pre-seed as already imported
    save_sync_state(
        tmp_path,
        "granola",
        {
            "backend": "granola",
            "files": {
                "doc_test_001": {
                    "filename": "2025-10-28_q1.md",
                    "remote_updated_at": "2025-10-29T01:23:45Z",
                    "status": "imported",
                },
            },
        },
    )

    result = GranolaBackend().sync(tmp_path, source_path=muesli_dir, force=True)
    assert result["available"] == 1  # force causes re-import


# ---------------------------------------------------------------------------
# GranolaBackend.sync() — import mode
# ---------------------------------------------------------------------------


def test_granola_sync_import(tmp_path, monkeypatch):
    """Import mode writes segments and updates state."""
    from think.importers.granola import GranolaBackend
    from think.importers.sync import load_sync_state

    # Point journal to tmp_path
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)

    result = GranolaBackend().sync(
        tmp_path, source_path=muesli_dir, dry_run=False
    )

    assert result["downloaded"] == 1
    assert result["imported"] == 1
    assert result["available"] == 0
    assert result["errors"] == []

    # State updated
    state = load_sync_state(tmp_path, "granola")
    doc = state["files"]["doc_test_001"]
    assert doc["status"] == "imported"
    assert "imported_at" in doc
    assert doc["segments"] > 0

    # Verify segments were written
    # The transcript spans 15:04:12 to 15:09:45 (local time from UTC),
    # so we check for import.granola directories
    import glob

    segments = glob.glob(str(tmp_path / "*/import.granola/*/conversation_transcript.jsonl"))
    assert len(segments) >= 1

    # Check JSONL content of first segment
    with open(segments[0], "r") as f:
        lines = f.readlines()

    # First line is metadata
    meta = json.loads(lines[0])
    assert "imported" in meta
    assert meta["imported"]["id"] == "doc_test_001"
    assert meta["topics"] == "Q1 Planning"

    # Subsequent lines are entries
    entry = json.loads(lines[1])
    assert "speaker" in entry
    assert "text" in entry
    assert "start" in entry
    assert entry["source"] == "import"

    # Check source.md was copied
    source_files = glob.glob(str(tmp_path / "*/import.granola/*/source.md"))
    assert len(source_files) == 1  # only in first segment


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_granola_backends_cli_flag(capsys, monkeypatch):
    """sol import --backends lists granola."""
    import sys

    from think.importers.cli import main

    monkeypatch.setattr(sys, "argv", ["sol import", "--backends"])
    monkeypatch.setenv("JOURNAL_PATH", "/tmp/test-journal")
    main()
    captured = capsys.readouterr()
    assert "granola" in captured.out


def test_granola_sync_cli(capsys, monkeypatch, tmp_path):
    """sol import --sync granola --path <dir> runs catalog."""
    import sys

    from think.importers.cli import main

    muesli_dir = tmp_path / "muesli"
    _write_transcript(muesli_dir, "2025-10-28_q1.md", SAMPLE_TRANSCRIPT)

    monkeypatch.setattr(
        sys,
        "argv",
        ["sol import", "--sync", "granola", "--path", str(muesli_dir)],
    )
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    main()
    captured = capsys.readouterr()
    assert "Total:" in captured.out
    assert "Available to import:" in captured.out
    assert "Q1 Planning" in captured.out  # title shown in available list
