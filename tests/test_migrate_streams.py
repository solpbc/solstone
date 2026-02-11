# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for scratch/migrate_streams.py stream migration script."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

# Set JOURNAL_PATH before importing anything that calls get_journal()
os.environ["JOURNAL_PATH"] = "tests/fixtures/journal"

# scratch/ is not an installed package — load the module from file path
_script_path = Path(__file__).resolve().parent.parent / "scratch" / "migrate_streams.py"
_spec = importlib.util.spec_from_file_location("migrate_streams", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SIGNAL_AUDIO_HOST = _mod.SIGNAL_AUDIO_HOST
SIGNAL_AUDIO_IMPORTED = _mod.SIGNAL_AUDIO_IMPORTED
SIGNAL_AUDIO_REMOTE = _mod.SIGNAL_AUDIO_REMOTE
SIGNAL_AUDIO_STREAM = _mod.SIGNAL_AUDIO_STREAM
SIGNAL_EXISTING = _mod.SIGNAL_EXISTING
SIGNAL_HOSTNAME_FALLBACK = _mod.SIGNAL_HOSTNAME_FALLBACK
SIGNAL_IMPORT_INDEX = _mod.SIGNAL_IMPORT_INDEX
SIGNAL_IMPORTED_JSONL = _mod.SIGNAL_IMPORTED_JSONL
SIGNAL_TMUX_ONLY = _mod.SIGNAL_TMUX_ONLY
build_import_reverse_index = _mod.build_import_reverse_index
classify_segment = _mod.classify_segment
migrate_main = _mod.main

from think.streams import read_segment_stream


@pytest.fixture
def tmp_journal(tmp_path):
    """Create a temporary journal with test segments."""
    journal = tmp_path / "journal"
    journal.mkdir()

    day1 = journal / "20240501"
    day1.mkdir()

    # Basic segment with audio.jsonl
    seg1 = day1 / "100000_300"
    seg1.mkdir()
    header = {"raw": "audio.flac", "model": "whisper-1"}
    (seg1 / "audio.jsonl").write_text(
        json.dumps(header)
        + "\n"
        + json.dumps({"start": "00:00:01", "text": "hello"})
        + "\n"
    )

    # Segment with remote field
    seg2 = day1 / "100500_300"
    seg2.mkdir()
    header = {"raw": "audio.flac", "remote": "laptop"}
    (seg2 / "audio.jsonl").write_text(json.dumps(header) + "\n")

    # Segment with imported field
    seg3 = day1 / "110000_300"
    seg3.mkdir()
    header = {
        "raw": "../../imports/20240501_110000/meeting.m4a",
        "imported": {"id": "20240501_110000"},
    }
    (seg3 / "audio.jsonl").write_text(json.dumps(header) + "\n")

    # Segment with stream field in audio.jsonl
    seg4 = day1 / "120000_300"
    seg4.mkdir()
    header = {"raw": "audio.flac", "stream": "workstation"}
    (seg4 / "audio.jsonl").write_text(json.dumps(header) + "\n")

    # Segment with existing stream.json
    seg5 = day1 / "130000_300"
    seg5.mkdir()
    (seg5 / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")
    (seg5 / "stream.json").write_text(
        json.dumps(
            {"stream": "myhost", "prev_day": None, "prev_segment": None, "seq": 1}
        )
    )

    # Segment with host field
    seg6 = day1 / "140000_300"
    seg6.mkdir()
    header = {"raw": "audio.flac", "host": "serverbox"}
    (seg6 / "audio.jsonl").write_text(json.dumps(header) + "\n")

    # Tmux-only segment
    seg7 = day1 / "150000_300"
    seg7.mkdir()
    (seg7 / "tmux_main_screen.jsonl").write_text(
        json.dumps({"frame_id": 1, "content": "$ ls"}) + "\n"
    )

    # imported_audio.jsonl segment
    seg8 = day1 / "160000_300"
    seg8.mkdir()
    header = {"raw": "../../imports/test/recording.m4a", "imported": {"id": "test"}}
    (seg8 / "imported_audio.jsonl").write_text(json.dumps(header) + "\n")

    # Bare segment (screen only)
    seg9 = day1 / "170000_300"
    seg9.mkdir()
    (seg9 / "screen.jsonl").write_text(
        json.dumps({"frame_id": 1, "analysis": {"primary": "code"}}) + "\n"
    )

    return journal


class TestClassifySegment:
    """Test the signal cascade for segment classification."""

    def test_existing_marker(self, tmp_journal):
        seg = tmp_journal / "20240501" / "130000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "myhost"
        assert signal == SIGNAL_EXISTING

    def test_audio_stream_field(self, tmp_journal):
        seg = tmp_journal / "20240501" / "120000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "workstation"
        assert signal == SIGNAL_AUDIO_STREAM

    def test_audio_remote_field(self, tmp_journal):
        seg = tmp_journal / "20240501" / "100500_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "laptop"
        assert signal == SIGNAL_AUDIO_REMOTE

    def test_audio_imported_field(self, tmp_journal):
        seg = tmp_journal / "20240501" / "110000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "import.apple"
        assert signal == SIGNAL_AUDIO_IMPORTED

    def test_imported_audio_jsonl(self, tmp_journal):
        seg = tmp_journal / "20240501" / "160000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "import.apple"
        assert signal == SIGNAL_IMPORTED_JSONL

    def test_audio_host_field(self, tmp_journal):
        seg = tmp_journal / "20240501" / "140000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "serverbox"
        assert signal == SIGNAL_AUDIO_HOST

    def test_tmux_only(self, tmp_journal):
        seg = tmp_journal / "20240501" / "150000_300"
        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "fallback.tmux"
        assert signal == SIGNAL_TMUX_ONLY

    def test_hostname_fallback(self, tmp_journal):
        seg = tmp_journal / "20240501" / "170000_300"
        name, signal = classify_segment(seg, "20240501", {}, "mypc")
        assert name == "mypc"
        assert signal == SIGNAL_HOSTNAME_FALLBACK

    def test_import_reverse_index(self, tmp_journal):
        """Import reverse index takes priority over hostname fallback."""
        imports_dir = tmp_journal / "imports" / "20240501_170000"
        imports_dir.mkdir(parents=True)
        (imports_dir / "segments.json").write_text(
            json.dumps({"segments": ["170000_300"], "day": "20240501"})
        )
        (imports_dir / "import.json").write_text(
            json.dumps({"original_filename": "notes.txt", "mime_type": "text/plain"})
        )

        index = build_import_reverse_index(tmp_journal)
        seg = tmp_journal / "20240501" / "170000_300"
        name, signal = classify_segment(seg, "20240501", index, "fallback")
        assert name == "import.text"
        assert signal == SIGNAL_IMPORT_INDEX


class TestBuildImportReverseIndex:
    def test_empty_journal(self, tmp_path):
        journal = tmp_path / "journal"
        journal.mkdir()
        assert build_import_reverse_index(journal) == {}

    def test_no_imports_dir(self, tmp_path):
        journal = tmp_path / "journal"
        journal.mkdir()
        assert build_import_reverse_index(journal) == {}

    def test_with_imports(self, tmp_path):
        journal = tmp_path / "journal"
        imports = journal / "imports" / "20240501_120000"
        imports.mkdir(parents=True)
        (imports / "segments.json").write_text(
            json.dumps({"segments": ["120000_300", "120500_300"], "day": "20240501"})
        )
        (imports / "import.json").write_text(
            json.dumps(
                {"original_filename": "recording.m4a", "mime_type": "audio/mp4"}
            )
        )

        index = build_import_reverse_index(journal)
        assert ("20240501", "120000_300") in index
        assert ("20240501", "120500_300") in index
        assert index[("20240501", "120000_300")]["source"] == "apple"

    def test_text_import(self, tmp_path):
        journal = tmp_path / "journal"
        imports = journal / "imports" / "20240501_130000"
        imports.mkdir(parents=True)
        (imports / "segments.json").write_text(
            json.dumps({"segments": ["130000_300"], "day": "20240501"})
        )
        (imports / "import.json").write_text(
            json.dumps({"original_filename": "notes.md"})
        )

        index = build_import_reverse_index(journal)
        assert index[("20240501", "130000_300")]["source"] == "text"


class TestSignalPriority:
    def test_existing_marker_beats_audio_stream(self, tmp_path):
        journal = tmp_path / "journal"
        day = journal / "20240501"
        seg = day / "100000_300"
        seg.mkdir(parents=True)
        (seg / "audio.jsonl").write_text(
            json.dumps({"raw": "audio.flac", "stream": "other"}) + "\n"
        )
        (seg / "stream.json").write_text(
            json.dumps(
                {
                    "stream": "correct",
                    "prev_day": None,
                    "prev_segment": None,
                    "seq": 1,
                }
            )
        )

        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "correct"
        assert signal == SIGNAL_EXISTING

    def test_remote_beats_host(self, tmp_path):
        journal = tmp_path / "journal"
        day = journal / "20240501"
        seg = day / "100000_300"
        seg.mkdir(parents=True)
        (seg / "audio.jsonl").write_text(
            json.dumps({"raw": "audio.flac", "remote": "phone", "host": "server"})
            + "\n"
        )

        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "phone"
        assert signal == SIGNAL_AUDIO_REMOTE

    def test_imported_beats_host(self, tmp_path):
        journal = tmp_path / "journal"
        day = journal / "20240501"
        seg = day / "100000_300"
        seg.mkdir(parents=True)
        (seg / "audio.jsonl").write_text(
            json.dumps(
                {
                    "raw": "recording.m4a",
                    "imported": {"id": "test"},
                    "host": "server",
                }
            )
            + "\n"
        )

        name, signal = classify_segment(seg, "20240501", {}, "fallback")
        assert name == "import.apple"
        assert signal == SIGNAL_AUDIO_IMPORTED


class TestEndToEnd:
    def test_dry_run(self, tmp_path, monkeypatch):
        """Dry-run should classify but not write markers."""
        journal = tmp_path / "journal"
        day = journal / "20240601"
        day.mkdir(parents=True)

        seg1 = day / "090000_300"
        seg1.mkdir()
        (seg1 / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")
        seg2 = day / "090500_300"
        seg2.mkdir()
        (seg2 / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        import think.utils

        think.utils._journal_path_cache = None

        import sys

        monkeypatch.setattr(sys, "argv", ["migrate_streams", "--host", "testbox"])
        migrate_main()

        assert not (seg1 / "stream.json").exists()
        assert not (seg2 / "stream.json").exists()

    def test_apply(self, tmp_path, monkeypatch):
        """Apply mode should write markers with correct linkage."""
        journal = tmp_path / "journal"
        day1 = journal / "20240601"
        day1.mkdir(parents=True)
        day2 = journal / "20240602"
        day2.mkdir(parents=True)

        seg1 = day1 / "090000_300"
        seg1.mkdir()
        (seg1 / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")
        seg2 = day1 / "100000_300"
        seg2.mkdir()
        (seg2 / "audio.jsonl").write_text(
            json.dumps({"raw": "audio.flac", "remote": "phone"}) + "\n"
        )
        seg3 = day2 / "080000_300"
        seg3.mkdir()
        (seg3 / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        import think.utils

        think.utils._journal_path_cache = None

        import sys

        monkeypatch.setattr(
            sys, "argv", ["migrate_streams", "--apply", "--host", "testbox"]
        )
        migrate_main()

        m1 = read_segment_stream(seg1)
        assert m1 is not None
        assert m1["stream"] == "testbox"
        assert m1["seq"] == 1
        assert m1["prev_day"] is None

        m2 = read_segment_stream(seg2)
        assert m2 is not None
        assert m2["stream"] == "phone"
        assert m2["seq"] == 1

        m3 = read_segment_stream(seg3)
        assert m3 is not None
        assert m3["stream"] == "testbox"
        assert m3["seq"] == 2
        assert m3["prev_day"] == "20240601"
        assert m3["prev_segment"] == "090000_300"

        # Stream state files created
        streams_dir = journal / "streams"
        assert (streams_dir / "testbox.json").exists()
        assert (streams_dir / "phone.json").exists()

        testbox_state = json.loads((streams_dir / "testbox.json").read_text())
        assert testbox_state["last_day"] == "20240602"
        assert testbox_state["seq"] == 2

    def test_idempotent(self, tmp_path, monkeypatch):
        """Running apply twice should produce identical results."""
        journal = tmp_path / "journal"
        day = journal / "20240601"
        day.mkdir(parents=True)

        seg = day / "090000_300"
        seg.mkdir()
        (seg / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        import think.utils

        think.utils._journal_path_cache = None

        import sys

        monkeypatch.setattr(
            sys, "argv", ["migrate_streams", "--apply", "--host", "testbox"]
        )
        migrate_main()

        m1 = read_segment_stream(seg)
        assert m1["stream"] == "testbox"

        # Second run — should skip (already correct)
        migrate_main()

        m2 = read_segment_stream(seg)
        assert m1 == m2

    def test_hostname_domain_stripped(self, tmp_path, monkeypatch):
        """Fallback host with domain suffix should be stripped."""
        journal = tmp_path / "journal"
        day = journal / "20240601"
        day.mkdir(parents=True)

        seg = day / "090000_300"
        seg.mkdir()
        (seg / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        import think.utils

        think.utils._journal_path_cache = None

        import sys

        monkeypatch.setattr(
            sys, "argv", ["migrate_streams", "--apply", "--host", "ja1r.local"]
        )
        migrate_main()

        m = read_segment_stream(seg)
        assert m["stream"] == "ja1r"
