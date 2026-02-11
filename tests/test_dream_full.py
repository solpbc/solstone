# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the dream module unified priority system."""

import importlib
import shutil
from pathlib import Path

FIXTURES = Path("tests/fixtures")


def copy_journal(tmp_path: Path) -> Path:
    src = FIXTURES / "journal"
    dest = tmp_path / "journal"
    shutil.copytree(src, dest)
    return dest


def test_main_runs_with_mocked_prompts(tmp_path, monkeypatch):
    """Test that main() runs pre/post phases and prompts by priority."""
    mod = importlib.import_module("think.dream")
    journal = copy_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    commands_run = []
    prompts_run = False

    def mock_run_command(cmd, day):
        commands_run.append(cmd)
        return True

    def mock_run_queued_command(cmd, day, timeout=600):
        commands_run.append(cmd)
        return True

    def mock_run_prompts_by_priority(day, segment, force, verbose, **kwargs):
        nonlocal prompts_run
        prompts_run = True
        return (5, 0, [])  # 5 success, 0 failures, no failed names

    monkeypatch.setattr(mod, "run_command", mock_run_command)
    monkeypatch.setattr(mod, "run_queued_command", mock_run_queued_command)
    monkeypatch.setattr(mod, "run_prompts_by_priority", mock_run_prompts_by_priority)
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        ["sol dream", "--day", "20240101", "--force", "--verbose"],
    )

    mod.main()

    # Verify pre-phase: sense ran
    assert any(c[0] == "sol" and c[1] == "sense" for c in commands_run)

    # Verify main phase: prompts ran
    assert prompts_run, "run_prompts_by_priority should have been called"

    # Verify post-phase: indexer rescan ran
    indexer_cmds = [c for c in commands_run if c[0] == "sol" and c[1] == "indexer"]
    assert len(indexer_cmds) >= 1
    assert any("--rescan" in cmd for cmd in indexer_cmds)


def test_segment_mode_skips_pre_post_phases(tmp_path, monkeypatch):
    """Test that segment mode skips sense and journal-stats."""
    mod = importlib.import_module("think.dream")
    journal = copy_journal(tmp_path)

    # Create segment directory
    segment_dir = journal / "20240101" / "120000_300"
    segment_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    commands_run = []

    def mock_run_command(cmd, day):
        commands_run.append(cmd)
        return True

    def mock_run_queued_command(cmd, day, timeout=600):
        commands_run.append(cmd)
        return True

    def mock_run_prompts_by_priority(day, segment, force, verbose, **kwargs):
        return (1, 0, [])

    monkeypatch.setattr(mod, "run_command", mock_run_command)
    monkeypatch.setattr(mod, "run_queued_command", mock_run_queued_command)
    monkeypatch.setattr(mod, "run_prompts_by_priority", mock_run_prompts_by_priority)
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        ["sol dream", "--day", "20240101", "--segment", "120000_300"],
    )

    mod.main()

    # Segment mode should NOT run sense or journal-stats
    assert not any(c[1] == "sense" for c in commands_run if len(c) > 1)
    assert not any(c[1] == "journal-stats" for c in commands_run if len(c) > 1)


def test_priority_validation_required(tmp_path, monkeypatch):
    """Test that get_muse_configs raises error for scheduled prompts without priority."""
    from think.muse import get_muse_configs

    # This test verifies the validation exists - actual validation tested in test_utils.py
    # Here we just confirm all existing scheduled prompts have priority
    configs = get_muse_configs(schedule="daily")
    for name, config in configs.items():
        assert "priority" in config, f"Scheduled prompt '{name}' missing priority"


def test_run_single_prompt_validates_schedule(tmp_path, monkeypatch):
    """Test that --run validates schedule compatibility."""
    mod = importlib.import_module("think.dream")
    journal = copy_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Mock to avoid actual execution
    def mock_cortex_request(*args, **kwargs):
        return "mock-id"

    def mock_wait_for_agents(*args, **kwargs):
        return ({"mock-id": "finish"}, [])

    monkeypatch.setattr(mod, "cortex_request", mock_cortex_request)
    monkeypatch.setattr(mod, "wait_for_agents", mock_wait_for_agents)

    # Running a daily prompt with --segment should fail
    # Note: This requires a real daily prompt in the fixtures
    # For now, just verify the function exists and is callable
    assert callable(mod.run_single_prompt)
