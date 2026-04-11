# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.segment."""

import argparse
import json

import pytest

from think.segment import cmd_inspect, cmd_list, cmd_verify


def _make_segment(
    base,
    day,
    stream,
    segment,
    *,
    stream_json=None,
    audio=True,
    screen=True,
    agents=None,
):
    """Create a minimal segment fixture directory."""
    seg_dir = base / day / stream / segment
    seg_dir.mkdir(parents=True)
    if stream_json is not None:
        (seg_dir / "stream.json").write_text(json.dumps(stream_json))
    if audio:
        (seg_dir / "audio.jsonl").write_text('{"t":0}\n')
    if screen:
        (seg_dir / "screen.jsonl").write_text('{"t":0}\n')
    if agents:
        agents_dir = seg_dir / "agents"
        agents_dir.mkdir()
        for name in agents:
            (agents_dir / name).write_text("# agent output\n")
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
        agents=["audio.md"],
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
        agents=["audio.md", "screen.md"],
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
    )
    _make_segment(
        tmp_path,
        "20240101",
        "custom",
        "100000_300",
        stream_json={"stream": "custom", "prev_day": None, "prev_segment": None, "seq": 1},
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
        agents=["audio.md"],
    )

    args = argparse.Namespace(
        day="20240101", stream=None, json_output=True, subcommand="list"
    )
    cmd_list(args)
    data = json.loads(capsys.readouterr().out)

    assert isinstance(data, list)
    assert data[0]["stream"] == "default"
    assert data[0]["segment"] == "090000_300"
    assert data[0]["agents"] == 1


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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
        agents=["audio.md"],
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
        agents=["audio.md"],
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
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
    assert "FAIL  backward chain: missing previous segment 20240101/default/090000_300" in out


def test_verify_day_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _make_segment(
        tmp_path,
        "20240101",
        "default",
        "090000_300",
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
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
        stream_json={"stream": "default", "prev_day": None, "prev_segment": None, "seq": 1},
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
