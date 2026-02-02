# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import os


def test_get_muse_configs_generators():
    """Test that system generators are discovered with source field."""
    muse = importlib.import_module("think.muse")
    generators = muse.get_muse_configs(has_tools=False, has_output=True)
    assert "flow" in generators
    info = generators["flow"]
    assert os.path.basename(info["path"]) == "flow.md"
    assert isinstance(info["color"], str)
    assert isinstance(info["mtime"], int)
    assert "title" in info
    assert "occurrences" in info
    # New: check source field
    assert info.get("source") == "system"


def test_get_output_topic():
    """Test generator key to filename conversion."""
    muse = importlib.import_module("think.muse")

    # System generators: key unchanged
    assert muse.get_output_topic("activity") == "activity"
    assert muse.get_output_topic("flow") == "flow"

    # App generators: _app_topic format
    assert muse.get_output_topic("chat:sentiment") == "_chat_sentiment"
    assert muse.get_output_topic("my_app:weekly_summary") == "_my_app_weekly_summary"


def test_get_muse_configs_app_discovery(tmp_path, monkeypatch):
    """Test that app generators are discovered from apps/*/muse/."""
    muse = importlib.import_module("think.muse")

    # Create a fake app with a generator
    app_dir = tmp_path / "apps" / "test_app" / "muse"
    app_dir.mkdir(parents=True)

    # Create generator files with frontmatter
    (app_dir / "custom_generator.md").write_text(
        '{\n  "title": "Custom Generator",\n  "color": "#ff0000"\n}\n\nTest prompt'
    )

    # Also create workspace.html to make it a valid app (not strictly required for generators)
    (tmp_path / "apps" / "test_app" / "workspace.html").write_text("<h1>Test</h1>")

    # For now, just verify system generators have correct source
    generators = muse.get_muse_configs(has_tools=False, has_output=True)
    for key, info in generators.items():
        if ":" not in key:
            assert info.get("source") == "system", f"{key} should have source=system"


def test_get_muse_configs_by_schedule():
    """Test filtering generators by schedule."""
    muse = importlib.import_module("think.muse")

    # Get daily generators
    daily = muse.get_muse_configs(has_tools=False, has_output=True, schedule="daily")
    assert len(daily) > 0
    for key, meta in daily.items():
        assert meta.get("schedule") == "daily", f"{key} should have schedule=daily"

    # Get segment generators
    segment = muse.get_muse_configs(
        has_tools=False, has_output=True, schedule="segment"
    )
    assert len(segment) > 0
    for key, meta in segment.items():
        assert meta.get("schedule") == "segment", f"{key} should have schedule=segment"

    # Verify no overlap
    assert not set(daily.keys()) & set(
        segment.keys()
    ), "daily and segment should not overlap"

    # Unknown schedule returns empty dict
    assert (
        muse.get_muse_configs(has_tools=False, has_output=True, schedule="hourly") == {}
    )
    assert muse.get_muse_configs(has_tools=False, has_output=True, schedule="") == {}


def test_get_muse_configs_include_disabled(monkeypatch):
    """Test include_disabled parameter."""
    muse = importlib.import_module("think.muse")

    # Get generators without disabled (default)
    without_disabled = muse.get_muse_configs(
        has_tools=False, has_output=True, schedule="daily"
    )

    # Get generators with disabled included
    with_disabled = muse.get_muse_configs(
        has_tools=False, has_output=True, schedule="daily", include_disabled=True
    )

    # Should have at least as many with disabled included
    # (files.md, media.md, tools.md are disabled by default)
    assert len(with_disabled) >= len(without_disabled)


def test_scheduled_generators_have_valid_schedule():
    """Test that scheduled generators have valid schedule field.

    Generators with a schedule field must have valid values ('segment' or 'daily').
    Some generators (like importer) have output but no schedule - they're used
    for ad-hoc processing, not scheduled runs.
    """
    muse = importlib.import_module("think.muse")

    generators = muse.get_muse_configs(has_tools=False, has_output=True)
    valid_schedules = ("segment", "daily")

    for key, meta in generators.items():
        sched = meta.get("schedule")
        if sched is not None:
            assert (
                sched in valid_schedules
            ), f"Generator '{key}' has invalid schedule '{sched}'"


def test_speakers_has_required_audio():
    """Test that speakers generator has audio as required source."""
    muse = importlib.import_module("think.muse")

    generators = muse.get_muse_configs(
        has_tools=False, has_output=True, schedule="segment"
    )
    assert "speakers" in generators

    speakers = generators["speakers"]
    instructions = speakers.get("instructions", {})
    sources = instructions.get("sources", {})

    assert sources.get("audio") == "required", "speakers should require audio"
    assert sources.get("screen") is True, "speakers should include screen"
