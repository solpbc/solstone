# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.segment."""

import argparse
import json

import pytest

from think.segment import cmd_inspect, cmd_list, cmd_move, cmd_verify


def _make_segment(
    base,
    day,
    stream,
    segment,
    *,
    stream_json=None,
    audio=True,
    screen=True,
    talents=None,
):
    """Create a minimal segment fixture directory."""
    seg_dir = base / "chronicle" / day / stream / segment
    seg_dir.mkdir(parents=True)
    if stream_json is not None:
        (seg_dir / "stream.json").write_text(json.dumps(stream_json))
    if audio:
        (seg_dir / "audio.jsonl").write_text('{"t":0}\n')
    if screen:
        (seg_dir / "screen.jsonl").write_text('{"t":0}\n')
    if talents:
        talents_dir = seg_dir / "talents"
        talents_dir.mkdir()
        for name in talents:
            (talents_dir / name).write_text("# talent output\n")
    return seg_dir


def test_list_basic(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        talents=["audio.md"],
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
        talents=["audio.md", "screen.md"],
    )

    args = argparse.Namespace(
        day="20240101", stream=None, json_output=False, subcommand="list"
    )
    cmd_list(args)
    out = capsys.readouterr().out

    assert "STREAM" in out
    assert "SEGMENT" in out
    assert "default" in out
    assert "090000_300" in out
    assert "140000_300" in out


def test_list_stream_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "custom",
        "100000_300",
        stream_json={
            "stream": "custom",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )

    args = argparse.Namespace(
        day="20240101", stream="default", json_output=False, subcommand="list"
    )
    cmd_list(args)
    out = capsys.readouterr().out

    assert "090000_300" in out
    assert "100000_300" not in out


def test_list_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        talents=["audio.md"],
    )

    args = argparse.Namespace(
        day="20240101", stream=None, json_output=True, subcommand="list"
    )
    cmd_list(args)
    data = json.loads(capsys.readouterr().out)

    assert isinstance(data, list)
    assert data[0]["stream"] == "default"
    assert data[0]["segment"] == "090000_300"
    assert data[0]["talents"] == 1


def test_list_empty_day(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    args = argparse.Namespace(
        day="20240101", stream=None, json_output=False, subcommand="list"
    )
    cmd_list(args)
    out = capsys.readouterr().out

    assert "No segments found for 20240101" in out


def test_inspect_basic(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        talents=["audio.md"],
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300", json_output=False, subcommand="inspect"
    )
    cmd_inspect(args)
    out = capsys.readouterr().out

    assert "Segment: 20240101/default/090000_300" in out
    assert "Stream:  default (seq 1)" in out
    assert "09:00:00 - 09:05:00" in out
    assert "stream.json" in out
    assert "audio.md" in out


def test_inspect_bad_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    args = argparse.Namespace(path="bad/path", json_output=False, subcommand="inspect")
    with pytest.raises(SystemExit) as excinfo:
        cmd_inspect(args)

    err = capsys.readouterr().err
    assert excinfo.value.code == 1
    assert "Segment path must be day/stream/segment" in err


def test_inspect_missing_segment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    args = argparse.Namespace(
        path="20240101/default/090000_300", json_output=False, subcommand="inspect"
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_inspect(args)

    err = capsys.readouterr().err
    assert excinfo.value.code == 1
    assert "Segment not found" in err


def test_inspect_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        talents=["audio.md"],
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300", json_output=True, subcommand="inspect"
    )
    cmd_inspect(args)
    data = json.loads(capsys.readouterr().out)

    assert data["stream"] == "default"
    assert data["segment"] == "090000_300"
    assert data["chain"]["prev"] == "(none)"
    assert "stats" in data


def test_inspect_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "180000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "140000_300",
            "seq": 3,
        },
    )

    args = argparse.Namespace(
        path="20240101/default/140000_300", json_output=False, subcommand="inspect"
    )
    cmd_inspect(args)
    out = capsys.readouterr().out

    assert "prev: 20240101/default/090000_300" in out
    assert "next: 20240101/default/180000_300" in out


def test_verify_all_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        screen=False,
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        day=None,
        json_output=False,
        subcommand="verify",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "PASS  directory exists" in out
    assert "PASS  forward chain" in out
    assert "7/7 checks passed" in out


def test_verify_missing_stream_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(tmp_path, "20240101", "default", "090000_300", stream_json=None)
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        day=None,
        json_output=False,
        subcommand="verify",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "FAIL  stream.json exists: stream.json missing" in out


def test_verify_missing_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        audio=False,
        screen=False,
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        day=None,
        json_output=False,
        subcommand="verify",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "FAIL  content files present: no audio.jsonl or screen.jsonl" in out


def test_verify_broken_backward_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "140000_300", "seq": 2})
    )

    args = argparse.Namespace(
        path="20240101/default/140000_300",
        day=None,
        json_output=False,
        subcommand="verify",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert (
        "FAIL  backward chain: missing previous segment 20240101/default/090000_300"
        in out
    )


def test_verify_day_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "140000_300", "seq": 2})
    )

    args = argparse.Namespace(
        path=None, day="20240101", json_output=False, subcommand="verify"
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "--- 20240101/default/090000_300 ---" in out
    assert "--- 20240101/default/140000_300 ---" in out
    assert "Summary: 14/14 checks passed" in out


def test_verify_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        day=None,
        json_output=True,
        subcommand="verify",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    data = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 0
    assert isinstance(data, list)
    assert any(item["check"] == "directory exists" for item in data)


def test_verify_no_args(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    args = argparse.Namespace(
        path=None, day=None, json_output=False, subcommand="verify"
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_verify(args)

    err = capsys.readouterr().err
    assert excinfo.value.code == 1
    assert "verify requires a segment path or --day" in err


def test_main_no_subcommand(monkeypatch, capsys):
    from think.segment import main

    monkeypatch.setattr("sys.argv", ["sol"])
    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1


# --- move tests ---


def test_move_basic(tmp_path, monkeypatch, capsys):
    """Basic move from one day to another."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)

    assert not (tmp_path / "chronicle" / "20240101" / "default" / "090000_300").exists()
    assert (tmp_path / "chronicle" / "20240115" / "default" / "090000_300").is_dir()
    import think.streams

    marker = think.streams.read_segment_stream(
        tmp_path / "chronicle" / "20240115" / "default" / "090000_300"
    )
    assert marker["stream"] == "default"
    assert marker["seq"] == 1


def test_move_with_to_time(tmp_path, monkeypatch, capsys):
    """Move with --to-time changes the segment key, preserving duration."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240101",
        to_time="140000",
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)

    assert not (tmp_path / "chronicle" / "20240101" / "default" / "090000_300").exists()
    assert (tmp_path / "chronicle" / "20240101" / "default" / "140000_300").is_dir()


def test_move_dry_run(tmp_path, monkeypatch, capsys):
    """--dry-run prints plan without moving."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=True,
        subcommand="move",
    )
    cmd_move(args)
    out = capsys.readouterr().out

    assert "[dry run]" in out
    assert "090000_300" in out
    assert (tmp_path / "chronicle" / "20240101" / "default" / "090000_300").is_dir()


def test_move_collision_no_to_time(tmp_path, monkeypatch, capsys):
    """Collision without --to-time is an error."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240115",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240115", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_move(args)

    err = capsys.readouterr().err
    assert excinfo.value.code == 1
    assert "already exists" in err


def test_move_no_events_jsonl(tmp_path, monkeypatch, capsys):
    """Segment with no events.jsonl moves cleanly."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        audio=False,
        screen=False,
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)
    out = capsys.readouterr().out

    assert (tmp_path / "chronicle" / "20240115" / "default" / "090000_300").is_dir()
    assert "events.jsonl lines: 0" in out


def test_move_patches_successor(tmp_path, monkeypatch, capsys):
    """Moving a segment patches the successor's prev_day/prev_segment."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "140000_300", "seq": 2})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)

    import think.streams

    succ_marker = think.streams.read_segment_stream(
        tmp_path / "chronicle" / "20240101" / "default" / "140000_300"
    )
    assert succ_marker["prev_day"] == "20240115"
    assert succ_marker["prev_segment"] == "090000_300"


def test_move_stream_tail(tmp_path, monkeypatch, capsys):
    """Moving the stream tail (no successor) works cleanly."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "140000_300",
        stream_json={
            "stream": "default",
            "prev_day": "20240101",
            "prev_segment": "090000_300",
            "seq": 2,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "140000_300", "seq": 2})
    )

    args = argparse.Namespace(
        path="20240101/default/140000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)
    out = capsys.readouterr().out

    assert "successor to patch: (none - stream tail)" in out
    assert (tmp_path / "chronicle" / "20240115" / "default" / "140000_300").is_dir()


def test_move_rewrites_events_jsonl(tmp_path, monkeypatch, capsys):
    """events.jsonl day and segment fields are updated after move."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    seg_dir = _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
        audio=False,
        screen=False,
    )
    events = [
        {
            "tract": "observe",
            "event": "start",
            "day": "20240101",
            "segment": "090000_300",
        },
        {"tract": "dream", "event": "done", "day": "20240101", "segment": "090000_300"},
    ]
    (seg_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time="140000",
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)

    new_events_path = (
        tmp_path / "chronicle" / "20240115" / "default" / "140000_300" / "events.jsonl"
    )
    assert new_events_path.exists()
    lines = [
        json.loads(line)
        for line in new_events_path.read_text().splitlines()
        if line.strip()
    ]
    assert all(event["day"] == "20240115" for event in lines)
    assert all(event["segment"] == "140000_300" for event in lines)


def test_move_touches_health_markers(tmp_path, monkeypatch, capsys):
    """Health markers are touched on both source and destination days."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "default.json").write_text(
        json.dumps({"last_day": "20240101", "last_segment": "090000_300", "seq": 1})
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    cmd_move(args)

    assert (tmp_path / "chronicle" / "20240101" / "health" / "stream.updated").exists()
    assert (tmp_path / "chronicle" / "20240115" / "health" / "stream.updated").exists()


def test_move_same_location_refused(tmp_path, monkeypatch, capsys):
    """Moving to the same location is refused."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240101",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_move(args)
    assert excinfo.value.code == 1
    assert "same" in capsys.readouterr().err


def test_move_invalid_to_time(tmp_path, monkeypatch, capsys):
    """Invalid --to-time format is rejected."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="20240115",
        to_time="bad",
        dry_run=False,
        subcommand="move",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_move(args)
    assert excinfo.value.code == 1
    assert "Invalid --to-time" in capsys.readouterr().err


def test_move_invalid_to_day(tmp_path, monkeypatch, capsys):
    """Invalid --to-day format is rejected."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={
            "stream": "default",
            "prev_day": None,
            "prev_segment": None,
            "seq": 1,
        },
    )

    args = argparse.Namespace(
        path="20240101/default/090000_300",
        to_day="not-a-day",
        to_time=None,
        dry_run=False,
        subcommand="move",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_move(args)
    assert excinfo.value.code == 1
    assert "Invalid --to-day" in capsys.readouterr().err
