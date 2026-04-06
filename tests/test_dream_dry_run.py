# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for dream --dry-run."""

import importlib


def test_dry_run_daily(journal_copy, capsys):
    """Dry-run daily mode prints prompts without spawning agents."""
    mod = importlib.import_module("think.dream")

    mod.dry_run("20240101")

    out = capsys.readouterr().out
    assert "2024-01-01" in out
    assert "Pre-phase" in out
    assert "Post-phase" in out
    assert "Priority" in out
    assert "Total:" in out


def test_dry_run_segment(journal_copy, capsys):
    """Dry-run segment mode skips pre/post phases."""
    mod = importlib.import_module("think.dream")

    mod.dry_run("20240101", segment="120000_300")

    out = capsys.readouterr().out
    assert "segment 120000_300" in out
    assert "Sense orchestrator" in out
    assert "Pre-phase" not in out
    assert "Post-phase" not in out


def test_dry_run_segments_lists_all(journal_copy, capsys):
    """Dry-run --segments lists discovered segments."""
    mod = importlib.import_module("think.dream")

    mod.dry_run("20240101", segments=True)

    out = capsys.readouterr().out
    assert "segments" in out.lower()


def test_dry_run_flush(journal_copy, capsys):
    """Dry-run --flush shows flush-eligible agents."""
    mod = importlib.import_module("think.dream")

    mod.dry_run("20240101", flush=True, segment="120000_300")

    out = capsys.readouterr().out
    assert "flush" in out.lower()


def test_dry_run_shows_refresh(journal_copy, capsys):
    """Dry-run indicates refresh mode in header."""
    mod = importlib.import_module("think.dream")

    mod.dry_run("20240101", refresh=True)

    out = capsys.readouterr().out
    assert "(refresh)" in out


def test_dry_run_no_callosum(journal_copy, monkeypatch, capsys):
    """Dry-run works without callosum connection."""
    mod = importlib.import_module("think.dream")

    # Save and clear _callosum to verify dry_run doesn't create one
    prev = mod._callosum
    monkeypatch.setattr(mod, "_callosum", None)
    mod.dry_run("20240101")
    assert mod._callosum is None
    monkeypatch.setattr(mod, "_callosum", prev)
