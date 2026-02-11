# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.streams module."""

import threading

import pytest

from think.streams import (
    get_stream_state,
    list_streams,
    read_segment_stream,
    rebuild_stream_state,
    stream_name,
    update_stream,
    write_segment_stream,
)

# --- stream_name tests ---


def test_stream_name_observer():
    """Host only -> hostname."""
    assert stream_name(host="archon") == "archon"


def test_stream_name_observer_tmux():
    """Host + qualifier -> host.tmux."""
    assert stream_name(host="archon", qualifier="tmux") == "archon.tmux"


def test_stream_name_remote():
    """Remote name -> remote name."""
    assert stream_name(remote="laptop") == "laptop"


def test_stream_name_import_apple():
    """import_source='apple' -> import.apple."""
    assert stream_name(import_source="apple") == "import.apple"


def test_stream_name_import_text():
    """import_source='text' -> import.text."""
    assert stream_name(import_source="text") == "import.text"


def test_stream_name_sanitization():
    """Spaces, slashes, uppercase are normalized."""
    assert stream_name(host="My Host") == "my-host"
    assert stream_name(host="FOO/BAR") == "foo-bar"
    assert stream_name(host="  ARCHON  ") == "archon"
    assert stream_name(remote="My Laptop") == "my-laptop"


def test_stream_name_hostname_stripping():
    """Domain suffixes are stripped from hostnames and remote names."""
    # .local, .home, .lan etc — keep only first label
    assert stream_name(host="ja1r.local") == "ja1r"
    assert stream_name(host="archon.home") == "archon"
    assert stream_name(host="server.corp.example.com") == "server"
    assert stream_name(remote="phone.local") == "phone"

    # With qualifier — dot is for qualifier only
    assert stream_name(host="ja1r.local", qualifier="tmux") == "ja1r.tmux"

    # Simple hostnames unchanged
    assert stream_name(host="archon") == "archon"
    assert stream_name(remote="laptop") == "laptop"

    # IP addresses become dash-separated
    assert stream_name(host="192.168.1.1") == "192-168-1-1"
    assert stream_name(host="10.0.0.1") == "10-0-0-1"


def test_stream_name_validation():
    """Empty/invalid raises ValueError."""
    with pytest.raises(ValueError):
        stream_name()  # No source

    with pytest.raises(ValueError):
        stream_name(host="")  # Empty after strip

    with pytest.raises(ValueError):
        stream_name(host="  ")  # Whitespace only


# --- update_stream tests ---


def test_update_stream_first_segment(tmp_path, monkeypatch):
    """First segment creates state, prev=None, seq=1."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    result = update_stream("archon", "20250119", "142500_300", type="observer")

    assert result["prev_day"] is None
    assert result["prev_segment"] is None
    assert result["seq"] == 1

    # State file should exist
    state = get_stream_state("archon")
    assert state is not None
    assert state["name"] == "archon"
    assert state["type"] == "observer"
    assert state["last_day"] == "20250119"
    assert state["last_segment"] == "142500_300"
    assert state["seq"] == 1


def test_update_stream_subsequent(tmp_path, monkeypatch):
    """Subsequent segments increment seq and return correct prev."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    update_stream("archon", "20250119", "142500_300", type="observer")
    result = update_stream("archon", "20250119", "143000_300")

    assert result["prev_day"] == "20250119"
    assert result["prev_segment"] == "142500_300"
    assert result["seq"] == 2

    state = get_stream_state("archon")
    assert state["seq"] == 2
    assert state["last_segment"] == "143000_300"


def test_update_stream_cross_day(tmp_path, monkeypatch):
    """Prev points to different day when crossing midnight."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    update_stream("archon", "20250119", "235500_300")
    result = update_stream("archon", "20250120", "000000_300")

    assert result["prev_day"] == "20250119"
    assert result["prev_segment"] == "235500_300"
    assert result["seq"] == 2


# --- write/read segment stream tests ---


def test_write_read_segment_stream(tmp_path):
    """Round-trip write/read stream.json."""
    seg_dir = tmp_path / "20250119" / "142500_300"
    seg_dir.mkdir(parents=True)

    write_segment_stream(seg_dir, "archon", "20250119", "142000_300", 5)

    marker = read_segment_stream(seg_dir)
    assert marker is not None
    assert marker["stream"] == "archon"
    assert marker["prev_day"] == "20250119"
    assert marker["prev_segment"] == "142000_300"
    assert marker["seq"] == 5


def test_write_segment_stream_first(tmp_path):
    """First segment has None prev values."""
    seg_dir = tmp_path / "20250119" / "142500_300"
    seg_dir.mkdir(parents=True)

    write_segment_stream(seg_dir, "archon", None, None, 1)

    marker = read_segment_stream(seg_dir)
    assert marker["prev_day"] is None
    assert marker["prev_segment"] is None
    assert marker["seq"] == 1


def test_read_segment_stream_missing(tmp_path):
    """Returns None for pre-stream segments."""
    seg_dir = tmp_path / "20250119" / "142500_300"
    seg_dir.mkdir(parents=True)

    assert read_segment_stream(seg_dir) is None


# --- list_streams tests ---


def test_list_streams(tmp_path, monkeypatch):
    """Discovers all stream state files."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    update_stream("archon", "20250119", "142500_300", type="observer")
    update_stream("laptop", "20250119", "142500_300", type="observer")
    update_stream("import.apple", "20250119", "100000_300", type="import")

    streams = list_streams()
    names = [s["name"] for s in streams]
    assert "archon" in names
    assert "laptop" in names
    assert "import.apple" in names
    assert len(streams) == 3


# --- rebuild_stream_state tests ---


def test_rebuild_stream_state(tmp_path, monkeypatch):
    """Reconstructs state from segment markers."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create segment dirs with stream markers
    day_dir = tmp_path / "20250119"
    seg1 = day_dir / "142500_300"
    seg2 = day_dir / "143000_300"
    seg1.mkdir(parents=True)
    seg2.mkdir(parents=True)

    write_segment_stream(seg1, "archon", None, None, 1)
    write_segment_stream(seg2, "archon", "20250119", "142500_300", 2)

    # Delete stream state files to simulate corruption
    streams_dir = tmp_path / "streams"
    if streams_dir.exists():
        for f in streams_dir.glob("*.json"):
            f.unlink()

    # Rebuild
    summary = rebuild_stream_state()
    assert "archon" in summary["rebuilt"]
    assert summary["segments_scanned"] == 2

    # Verify rebuilt state
    state = get_stream_state("archon")
    assert state is not None
    assert state["seq"] == 2
    assert state["last_segment"] == "143000_300"


# --- atomicity test ---


def test_update_stream_atomicity(tmp_path, monkeypatch):
    """Concurrent writes don't corrupt state file."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    errors = []

    def writer(stream_id):
        try:
            for i in range(10):
                update_stream("archon", "20250119", f"{140000 + i}_300")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent writes: {errors}"

    # State file should be valid JSON
    state = get_stream_state("archon")
    assert state is not None
    assert state["seq"] > 0
    assert state["name"] == "archon"
