# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the activities module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Set up test environment before importing the module
os.environ["JOURNAL_PATH"] = "fixtures/journal"


def test_get_default_activities():
    """Test that default activities are returned correctly."""
    from think.activities import get_default_activities

    defaults = get_default_activities()

    # Should return a list
    assert isinstance(defaults, list)
    assert len(defaults) > 0

    # Each activity should have required fields
    for activity in defaults:
        assert "id" in activity
        assert "name" in activity
        assert "description" in activity

    # Check some known activities exist
    ids = [a["id"] for a in defaults]
    assert "meeting" in ids
    assert "coding" in ids
    assert "browsing" in ids


def test_get_default_activities_returns_copy():
    """Test that get_default_activities returns a copy, not the original."""
    from think.activities import get_default_activities

    defaults1 = get_default_activities()
    defaults2 = get_default_activities()

    # Should be equal but not the same object
    assert defaults1 == defaults2
    assert defaults1 is not defaults2

    # Modifying one should not affect the other
    defaults1[0]["id"] = "modified"
    assert defaults2[0]["id"] != "modified"


def test_generate_activity_id():
    """Test activity ID generation from names."""
    from think.activities import generate_activity_id

    assert generate_activity_id("My Activity") == "my_activity"
    assert generate_activity_id("Research & Development") == "research_development"
    assert generate_activity_id("  Spaces  ") == "spaces"
    assert generate_activity_id("123-Numbers!") == "123_numbers"
    assert generate_activity_id("") == "activity"


def test_facet_activities_empty():
    """Test loading activities from a facet with no activities file."""
    from think.activities import get_facet_activities

    # The test journal may not have activities set up
    activities = get_facet_activities("personal")

    # Should return empty list if no file
    assert isinstance(activities, list)


def test_facet_activities_roundtrip():
    """Test saving and loading activities."""
    from think.activities import (
        _get_activities_path,
        get_facet_activities,
        save_facet_activities,
    )

    # Create a temp journal
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override JOURNAL_PATH
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create facet directory
        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Save some activities
            activities = [
                {"id": "meeting", "priority": "high"},
                {"id": "coding", "description": "Custom coding description"},
                {
                    "id": "custom_activity",
                    "name": "Custom",
                    "description": "A custom activity",
                    "custom": True,
                },
            ]
            save_facet_activities("test_facet", activities)

            # Verify file was created
            path = _get_activities_path("test_facet")
            assert path.exists()

            # Load and verify
            loaded = get_facet_activities("test_facet")
            assert len(loaded) == 3

            # Check meeting (predefined with priority override)
            meeting = next(a for a in loaded if a["id"] == "meeting")
            assert meeting["priority"] == "high"
            assert meeting["custom"] is False
            assert "name" in meeting  # Should have default name

            # Check coding (predefined with description override)
            coding = next(a for a in loaded if a["id"] == "coding")
            assert coding["description"] == "Custom coding description"

            # Check custom activity
            custom = next(a for a in loaded if a["id"] == "custom_activity")
            assert custom["custom"] is True
            assert custom["name"] == "Custom"

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path


def test_add_activity_to_facet():
    """Test adding an activity to a facet."""
    from think.activities import (
        add_activity_to_facet,
        get_facet_activities,
        remove_activity_from_facet,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Add a predefined activity
            result = add_activity_to_facet("test_facet", "meeting", priority="high")
            assert result["id"] == "meeting"

            # Verify it was added
            activities = get_facet_activities("test_facet")
            assert len(activities) == 1

            # Adding same activity again should not duplicate
            result2 = add_activity_to_facet("test_facet", "meeting")
            activities = get_facet_activities("test_facet")
            assert len(activities) == 1

            # Remove it
            removed = remove_activity_from_facet("test_facet", "meeting")
            assert removed is True
            activities = get_facet_activities("test_facet")
            assert len(activities) == 0

            # Removing non-existent should return False
            removed = remove_activity_from_facet("test_facet", "meeting")
            assert removed is False

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path


def test_update_activity_in_facet():
    """Test updating an activity in a facet."""
    from think.activities import (
        add_activity_to_facet,
        get_activity_by_id,
        update_activity_in_facet,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Add an activity
            add_activity_to_facet("test_facet", "meeting")

            # Update it
            updated = update_activity_in_facet(
                "test_facet", "meeting", priority="low", description="Updated desc"
            )
            assert updated is not None
            assert updated["priority"] == "low"
            assert updated["description"] == "Updated desc"

            # Verify via lookup
            activity = get_activity_by_id("test_facet", "meeting")
            assert activity["priority"] == "low"

            # Update non-existent should return None
            result = update_activity_in_facet(
                "test_facet", "nonexistent", priority="high"
            )
            assert result is None

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path
