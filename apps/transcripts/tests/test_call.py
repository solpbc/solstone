# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for transcripts CLI commands (sol call transcripts ...)."""

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


class TestScan:
    def test_scan_day(self):
        result = runner.invoke(call_app, ["transcripts", "scan", "20240101"])
        assert result.exit_code == 0
        assert "Audio:" in result.output
        assert "Screen:" in result.output

    def test_scan_empty_day(self):
        result = runner.invoke(call_app, ["transcripts", "scan", "20990101"])
        assert result.exit_code == 0
        assert "(none)" in result.output


class TestSegments:
    def test_segments_day(self):
        result = runner.invoke(call_app, ["transcripts", "segments", "20240101"])
        assert result.exit_code == 0
        assert "123456_300" in result.output

    def test_segments_empty(self):
        result = runner.invoke(call_app, ["transcripts", "segments", "20990101"])
        assert result.exit_code == 0
        assert "No segments" in result.output


class TestRead:
    def test_read_default(self):
        result = runner.invoke(call_app, ["transcripts", "read", "20240101"])
        assert result.exit_code == 0
        assert "## " in result.output

    def test_read_full(self):
        result = runner.invoke(call_app, ["transcripts", "read", "20240101", "--full"])
        assert result.exit_code == 0

    def test_read_raw(self):
        result = runner.invoke(call_app, ["transcripts", "read", "20240101", "--raw"])
        assert result.exit_code == 0

    def test_read_segment(self):
        result = runner.invoke(
            call_app, ["transcripts", "read", "20240101", "--segment", "123456_300"]
        )
        assert result.exit_code == 0

    def test_read_range(self):
        result = runner.invoke(
            call_app,
            ["transcripts", "read", "20240101", "--start", "123456", "--length", "5"],
        )
        assert result.exit_code == 0

    def test_read_full_and_raw_error(self):
        result = runner.invoke(
            call_app, ["transcripts", "read", "20240101", "--full", "--raw"]
        )
        assert result.exit_code == 1
        assert "Cannot use --full and --raw" in result.output

    def test_read_start_without_length(self):
        result = runner.invoke(
            call_app, ["transcripts", "read", "20240101", "--start", "123456"]
        )
        assert result.exit_code == 1
        assert "--start and --length must be used together" in result.output

    def test_read_segment_with_start(self):
        result = runner.invoke(
            call_app,
            [
                "transcripts",
                "read",
                "20240101",
                "--segment",
                "123456_300",
                "--start",
                "123456",
            ],
        )
        assert result.exit_code == 1


class TestStats:
    def test_stats_month(self):
        result = runner.invoke(call_app, ["transcripts", "stats", "202401"])
        assert result.exit_code == 0
        assert "20240101" in result.output
        assert "Total: 1 days with data" in result.output

    def test_stats_empty(self):
        result = runner.invoke(call_app, ["transcripts", "stats", "209901"])
        assert result.exit_code == 0
        assert "No data" in result.output


class TestSolEnvResolution:
    """Tests for SOL_* env var resolution in transcripts commands."""

    def test_scan_from_sol_day(self, monkeypatch):
        """scan with SOL_DAY env and no arg works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["transcripts", "scan"])
        assert result.exit_code == 0
        assert "Audio:" in result.output

    def test_read_from_sol_day(self, monkeypatch):
        """read with SOL_DAY env and no arg works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["transcripts", "read"])
        assert result.exit_code == 0

    def test_read_from_sol_day_and_segment(self, monkeypatch):
        """read with SOL_DAY + SOL_SEGMENT env works."""
        monkeypatch.setenv("SOL_DAY", "20240101")
        monkeypatch.setenv("SOL_SEGMENT", "123456_300")
        result = runner.invoke(call_app, ["transcripts", "read"])
        assert result.exit_code == 0
