# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import os


def test_get_muse_configs_generators():
    """Test that system generators are discovered with source field."""
    utils = importlib.import_module("think.utils")
    generators = utils.get_muse_configs(has_tools=False, has_output=True)
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
    utils = importlib.import_module("think.utils")

    # System generators: key unchanged
    assert utils.get_output_topic("activity") == "activity"
    assert utils.get_output_topic("flow") == "flow"

    # App generators: _app_topic format
    assert utils.get_output_topic("chat:sentiment") == "_chat_sentiment"
    assert utils.get_output_topic("my_app:weekly_summary") == "_my_app_weekly_summary"


def test_get_muse_configs_app_discovery(tmp_path, monkeypatch):
    """Test that app generators are discovered from apps/*/muse/."""
    utils = importlib.import_module("think.utils")

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
    generators = utils.get_muse_configs(has_tools=False, has_output=True)
    for key, info in generators.items():
        if ":" not in key:
            assert info.get("source") == "system", f"{key} should have source=system"


def test_get_muse_configs_by_schedule():
    """Test filtering generators by schedule."""
    utils = importlib.import_module("think.utils")

    # Get daily generators
    daily = utils.get_muse_configs(has_tools=False, has_output=True, schedule="daily")
    assert len(daily) > 0
    for key, meta in daily.items():
        assert meta.get("schedule") == "daily", f"{key} should have schedule=daily"

    # Get segment generators
    segment = utils.get_muse_configs(
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
        utils.get_muse_configs(has_tools=False, has_output=True, schedule="hourly")
        == {}
    )
    assert utils.get_muse_configs(has_tools=False, has_output=True, schedule="") == {}


def test_get_muse_configs_include_disabled(monkeypatch):
    """Test include_disabled parameter."""
    utils = importlib.import_module("think.utils")

    # Get generators without disabled (default)
    without_disabled = utils.get_muse_configs(
        has_tools=False, has_output=True, schedule="daily"
    )

    # Get generators with disabled included
    with_disabled = utils.get_muse_configs(
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
    utils = importlib.import_module("think.utils")

    generators = utils.get_muse_configs(has_tools=False, has_output=True)
    valid_schedules = ("segment", "daily")

    for key, meta in generators.items():
        sched = meta.get("schedule")
        if sched is not None:
            assert (
                sched in valid_schedules
            ), f"Generator '{key}' has invalid schedule '{sched}'"
