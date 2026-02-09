# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for output path generation with facet support."""

from pathlib import Path

from think.muse import get_output_path, get_output_topic


class TestGetOutputTopic:
    """Tests for get_output_topic."""

    def test_simple_key(self):
        assert get_output_topic("activity") == "activity"

    def test_app_key(self):
        assert get_output_topic("chat:sentiment") == "_chat_sentiment"

    def test_entities_app_key(self):
        assert get_output_topic("entities:observer") == "_entities_observer"


class TestGetOutputPath:
    """Tests for get_output_path."""

    def test_daily_output_md(self):
        path = get_output_path("/journal/20250101", "activity", output_format="md")
        assert path == Path("/journal/20250101/agents/activity.md")

    def test_daily_output_json(self):
        path = get_output_path("/journal/20250101", "facets", output_format="json")
        assert path == Path("/journal/20250101/agents/facets.json")

    def test_segment_output(self):
        path = get_output_path(
            "/journal/20250101", "activity", segment="120000_300", output_format="md"
        )
        assert path == Path("/journal/20250101/120000_300/agents/activity.md")

    def test_app_key_output(self):
        path = get_output_path(
            "/journal/20250101", "entities:observer", output_format="md"
        )
        assert path == Path("/journal/20250101/agents/_entities_observer.md")

    def test_facet_daily_output(self):
        """Multi-facet agent output uses a facet subdirectory."""
        path = get_output_path(
            "/journal/20250101", "newsletter", output_format="md", facet="work"
        )
        assert path == Path("/journal/20250101/agents/work/newsletter.md")

    def test_facet_segment_output(self):
        """Multi-facet segment output uses a facet subdirectory."""
        path = get_output_path(
            "/journal/20250101",
            "summary",
            segment="120000_300",
            output_format="json",
            facet="personal",
        )
        assert path == Path("/journal/20250101/120000_300/agents/personal/summary.json")

    def test_facet_with_app_key(self):
        """App-qualified key with facet uses both prefixes."""
        path = get_output_path(
            "/journal/20250101", "entities:observer", output_format="md", facet="work"
        )
        assert path == Path("/journal/20250101/agents/work/_entities_observer.md")

    def test_facet_none_same_as_omitted(self):
        """Explicit facet=None produces same path as omitting facet."""
        path_none = get_output_path(
            "/journal/20250101", "activity", output_format="md", facet=None
        )
        path_omit = get_output_path("/journal/20250101", "activity", output_format="md")
        assert path_none == path_omit
