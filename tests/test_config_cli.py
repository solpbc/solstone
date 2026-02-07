# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol config CLI."""

import json

from think.config_cli import main


def test_config_prints_json(monkeypatch, capsys):
    """Default command prints full config JSON."""
    monkeypatch.setenv("JOURNAL_PATH", "fixtures/journal")
    monkeypatch.setattr("sys.argv", ["sol config"])

    main()

    output = capsys.readouterr().out
    config = json.loads(output)
    assert "identity" in config


def test_config_env_prints_path(monkeypatch, capsys):
    """env subcommand prints resolved JOURNAL_PATH."""
    monkeypatch.setenv("JOURNAL_PATH", "fixtures/journal")
    monkeypatch.setattr("sys.argv", ["sol config", "env"])

    main()

    output = capsys.readouterr().out.strip()
    assert output == "JOURNAL_PATH=fixtures/journal (from shell)"


def test_config_env_shows_source(monkeypatch, capsys):
    """env subcommand includes source for the resolved path."""
    monkeypatch.setenv("JOURNAL_PATH", "fixtures/journal")
    monkeypatch.setattr("sys.argv", ["sol config", "env"])

    main()

    output = capsys.readouterr().out.strip()
    assert output.endswith("(from shell)")
