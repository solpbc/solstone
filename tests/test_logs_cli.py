# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
from datetime import datetime, timedelta

import pytest


def make_journal(tmp_path, day, services, supervisor_lines=None):
    """Create a synthetic journal with health logs."""
    health_dir = tmp_path / day / "health"
    health_dir.mkdir(parents=True)

    for name, lines in services.items():
        actual = health_dir / f"ref_{name}.log"
        actual.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

        symlink = health_dir / f"{name}.log"
        symlink.symlink_to(f"ref_{name}.log")

    journal_health = tmp_path / "health"
    journal_health.mkdir(exist_ok=True)

    for name in services:
        journal_sym = journal_health / f"{name}.log"
        journal_sym.symlink_to(f"../{day}/health/ref_{name}.log")

    if supervisor_lines is not None:
        sup = journal_health / "supervisor.log"
        sup.write_text(
            "\n".join(supervisor_lines) + "\n" if supervisor_lines else "",
            encoding="utf-8",
        )

    return tmp_path


def _args(
    *,
    c=5,
    since=None,
    service=None,
    grep=None,
):
    return argparse.Namespace(c=c, f=False, since=since, service=service, grep=grep)


def test_parse_log_line_runner():
    from think.logs_cli import parse_log_line

    result = parse_log_line("2026-02-09T10:00:00 [echo:stdout] hello world")
    assert result is not None
    assert result.service == "echo"
    assert result.stream == "stdout"
    assert result.message == "hello world"
    assert result.timestamp == datetime(2026, 2, 9, 10, 0, 0)


def test_parse_log_line_supervisor():
    from think.logs_cli import parse_log_line

    result = parse_log_line("2026-02-09T10:00:00 [supervisor:log] INFO Starting service")
    assert result is not None
    assert result.service == "supervisor"
    assert result.stream == "log"
    assert result.message == "INFO Starting service"


def test_parse_log_line_malformed():
    from think.logs_cli import parse_log_line

    assert parse_log_line("not a log line") is None
    assert parse_log_line("") is None
    assert parse_log_line("2026-02-09T10:00:00 no brackets") is None


def test_parse_since_relative():
    from think.logs_cli import parse_since

    result = parse_since("30m")
    delta = datetime.now() - result
    assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)


def test_parse_since_absolute():
    from think.logs_cli import parse_since

    result = parse_since("16:00")
    assert result.hour == 16
    assert result.minute == 0
    assert result.date() == datetime.now().date()


def test_parse_since_absolute_ampm():
    from think.logs_cli import parse_since

    result = parse_since("4pm")
    assert result.hour == 16
    result2 = parse_since("4:30pm")
    assert result2.hour == 16
    assert result2.minute == 30


def test_parse_since_invalid():
    from think.logs_cli import parse_since

    with pytest.raises(argparse.ArgumentTypeError):
        parse_since("xyz")


def test_tail_lines_large(tmp_path):
    from think.logs_cli import tail_lines_large

    path = tmp_path / "big.log"
    lines = [f"line {i}" for i in range(1000)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = tail_lines_large(path, 5)
    assert result == [f"line {i}" for i in range(995, 1000)]


def test_get_day_log_files_filters_non_symlinks(tmp_path):
    from think.logs_cli import get_day_log_files

    health = tmp_path / "health"
    health.mkdir()
    (health / "ref_echo.log").write_text("data", encoding="utf-8")
    (health / "echo.log").symlink_to("ref_echo.log")
    (health / "something.port").write_text("8080", encoding="utf-8")
    result = get_day_log_files(health)
    assert len(result) == 1
    assert result[0].name == "echo.log"


def test_collect_default(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    lines = [f"2026-02-09T10:{i:02d}:00 [echo:stdout] line {i}" for i in range(10)]
    make_journal(tmp_path, day, {"echo": lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(c=5))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 5
    assert "line 5" in output[0]
    assert "line 9" in output[4]


def test_collect_count(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    lines = [f"2026-02-09T11:{i:02d}:00 [echo:stdout] line {i}" for i in range(10)]
    make_journal(tmp_path, day, {"echo": lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(c=2))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 2
    assert "line 8" in output[0]
    assert "line 9" in output[1]


def test_filter_service(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    echo_lines = [
        "2026-02-09T10:00:00 [echo:stdout] alpha",
        "2026-02-09T10:01:00 [echo:stdout] beta",
    ]
    observer_lines = [
        "2026-02-09T10:00:30 [observer:stdout] gamma",
        "2026-02-09T10:01:30 [observer:stdout] delta",
    ]
    make_journal(tmp_path, day, {"echo": echo_lines, "observer": observer_lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(service="echo"))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 2
    assert all("[echo:stdout]" in line for line in output)


def test_filter_grep(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    lines = [
        "2026-02-09T10:00:00 [echo:stdout] normal line",
        "2026-02-09T10:01:00 [echo:stdout] special event",
    ]
    make_journal(tmp_path, day, {"echo": lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(grep="special"))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert "special event" in output[0]


def test_filter_since(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    lines = [
        "2026-02-09T10:00:00 [echo:stdout] old",
        "2026-02-09T10:20:00 [echo:stdout] new",
    ]
    make_journal(tmp_path, day, {"echo": lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(since=datetime(2026, 2, 9, 10, 10, 0)))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert "new" in output[0]


def test_filters_compose(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    echo_lines = [
        "2026-02-09T10:00:00 [echo:stdout] keep this special",
        "2026-02-09T10:01:00 [echo:stdout] ignore this",
    ]
    observer_lines = [
        "2026-02-09T10:00:30 [observer:stdout] special but wrong service",
    ]
    make_journal(tmp_path, day, {"echo": echo_lines, "observer": observer_lines})
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(service="echo", grep="special"))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert "[echo:stdout]" in output[0]
    assert "keep this special" in output[0]


def test_supervisor_included_default(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    echo_lines = [
        "2026-02-09T10:00:00 [echo:stdout] line 0",
        "2026-02-09T10:01:00 [echo:stdout] line 1",
        "2026-02-09T10:02:00 [echo:stdout] line 2",
    ]
    supervisor_lines = [
        "2026-02-09T10:03:00 [supervisor:log] INFO a",
        "2026-02-09T10:04:00 [supervisor:log] INFO b",
        "2026-02-09T10:05:00 [supervisor:log] INFO c",
    ]
    make_journal(tmp_path, day, {"echo": echo_lines}, supervisor_lines=supervisor_lines)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(c=2))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 4
    assert any("[echo:stdout]" in line for line in output)
    assert any("[supervisor:log]" in line for line in output)


def test_supervisor_excluded_with_filters(tmp_path, monkeypatch, capsys):
    from think import logs_cli

    day = datetime.now().strftime("%Y%m%d")
    echo_lines = [
        "2026-02-09T10:00:00 [echo:stdout] special",
    ]
    supervisor_lines = [
        "2026-02-09T10:01:00 [supervisor:log] INFO special",
    ]
    make_journal(tmp_path, day, {"echo": echo_lines}, supervisor_lines=supervisor_lines)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    logs_cli.collect_and_print(_args(grep="special"))

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert "[echo:stdout]" in output[0]
    assert "[supervisor:log]" not in output[0]
