# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for segment facet functionality in dream module."""

import json

import pytest


@pytest.fixture
def segment_dir(tmp_path, monkeypatch):
    """Create a temporary journal with segment directory."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"
    segment_path = day_dir / "120000_300"
    segment_path.mkdir(parents=True)

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    return segment_path


class TestLoadSegmentFacets:
    """Tests for load_segment_facets helper function."""

    def test_missing_file_returns_empty(self, segment_dir, monkeypatch):
        """Missing facets.json returns empty list."""
        from think.dream import load_segment_facets

        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_empty_file_returns_empty(self, segment_dir, monkeypatch):
        """Empty facets.json returns empty list."""
        from think.dream import load_segment_facets

        (segment_dir / "facets.json").write_text("")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_empty_array_returns_empty(self, segment_dir, monkeypatch):
        """Empty JSON array returns empty list."""
        from think.dream import load_segment_facets

        (segment_dir / "facets.json").write_text("[]")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_valid_facets_extracted(self, segment_dir, monkeypatch):
        """Valid facets.json extracts facet IDs."""
        from think.dream import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Code review", "level": "high"},
            {"facet": "personal", "activity": "Email check", "level": "low"},
        ]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        result = load_segment_facets("20240115", "120000_300")
        assert result == ["work", "personal"]

    def test_malformed_json_returns_empty(self, segment_dir, monkeypatch, caplog):
        """Malformed JSON returns empty list with error logged."""
        from think.dream import load_segment_facets

        (segment_dir / "facets.json").write_text("{ invalid json")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []
        assert "Failed to parse facets.json" in caplog.text

    def test_non_array_returns_empty(self, segment_dir, monkeypatch, caplog):
        """Non-array JSON returns empty list with warning."""
        from think.dream import load_segment_facets

        (segment_dir / "facets.json").write_text('{"facet": "work"}')
        result = load_segment_facets("20240115", "120000_300")
        assert result == []
        assert "not an array" in caplog.text

    def test_missing_facet_field_skipped(self, segment_dir, monkeypatch):
        """Items without facet field are skipped."""
        from think.dream import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Coding"},
            {"activity": "Unknown"},  # Missing facet field
            {"facet": "personal", "activity": "Email"},
        ]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        result = load_segment_facets("20240115", "120000_300")
        assert result == ["work", "personal"]


class TestRunSegmentAgentsMultiFacet:
    """Tests for multi-facet segment agent spawning."""

    def test_multi_facet_agent_spawns_per_facet(self, segment_dir, monkeypatch):
        """Multi-facet segment agent spawns once per detected facet."""
        from think import dream

        # Set up facets.json
        facets_data = [
            {"facet": "work", "activity": "Coding", "level": "high"},
            {"facet": "personal", "activity": "Email", "level": "low"},
        ]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        # Track cortex_request calls
        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"prompt": prompt, "name": name, "config": config})
            return "mock-agent-id"

        # Mock get_muse_configs to return a multi-facet segment agent
        def mock_get_muse_configs(has_tools=None, **kwargs):
            return {
                "test_agent": {
                    "schedule": "segment",
                    "multi_facet": True,
                    "tools": "journal",
                }
            }

        # Mock get_enabled_facets to return both facets as enabled
        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}, "personal": {"title": "Personal"}}

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)

        count = dream.run_segment_agents("20240115", "120000_300")

        assert count == 2  # One per facet
        assert len(spawned) == 2

        # Verify facet-specific config
        facets_spawned = [s["config"]["facet"] for s in spawned]
        assert "work" in facets_spawned
        assert "personal" in facets_spawned

        # Verify segment is included
        for s in spawned:
            assert s["config"]["segment"] == "120000_300"
            assert s["config"]["env"]["SEGMENT_KEY"] == "120000_300"

    def test_non_multi_facet_agent_spawns_once(self, segment_dir, monkeypatch):
        """Regular segment agent spawns once regardless of facets."""
        from think import dream

        # Set up facets.json with multiple facets
        facets_data = [
            {"facet": "work", "activity": "Coding", "level": "high"},
            {"facet": "personal", "activity": "Email", "level": "low"},
        ]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"prompt": prompt, "name": name, "config": config})
            return "mock-agent-id"

        # Regular agent (no multi_facet)
        def mock_get_muse_configs(has_tools=None, **kwargs):
            return {
                "regular_agent": {
                    "schedule": "segment",
                    "tools": "journal",
                }
            }

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)

        count = dream.run_segment_agents("20240115", "120000_300")

        assert count == 1
        assert len(spawned) == 1
        assert "facet" not in spawned[0]["config"]

    def test_multi_facet_no_facets_detected(self, segment_dir, monkeypatch):
        """Multi-facet agent with no facets detected spawns nothing."""
        from think import dream

        # Empty facets.json
        (segment_dir / "facets.json").write_text("[]")

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"prompt": prompt, "name": name, "config": config})
            return "mock-agent-id"

        def mock_get_muse_configs(has_tools=None, **kwargs):
            return {
                "test_agent": {
                    "schedule": "segment",
                    "multi_facet": True,
                    "tools": "journal",
                }
            }

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)

        count = dream.run_segment_agents("20240115", "120000_300")

        assert count == 0
        assert len(spawned) == 0

    def test_mixed_agents_spawn_correctly(self, segment_dir, monkeypatch):
        """Mix of multi-facet and regular agents spawn correctly."""
        from think import dream

        facets_data = [{"facet": "work", "activity": "Coding", "level": "high"}]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"prompt": prompt, "name": name, "config": config})
            return "mock-agent-id"

        def mock_get_muse_configs(has_tools=None, **kwargs):
            return {
                "faceted_agent": {
                    "schedule": "segment",
                    "multi_facet": True,
                    "tools": "journal",
                },
                "regular_agent": {
                    "schedule": "segment",
                    "tools": "journal",
                },
            }

        # Mock get_enabled_facets to return work as enabled
        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}}

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)

        count = dream.run_segment_agents("20240115", "120000_300")

        assert count == 2  # 1 faceted (1 facet) + 1 regular
        assert len(spawned) == 2

        faceted = [s for s in spawned if s["name"] == "faceted_agent"]
        regular = [s for s in spawned if s["name"] == "regular_agent"]

        assert len(faceted) == 1
        assert faceted[0]["config"]["facet"] == "work"

        assert len(regular) == 1
        assert "facet" not in regular[0]["config"]

    def test_muted_facets_filtered(self, segment_dir, monkeypatch):
        """Muted facets are filtered out from segment multi-facet agents."""
        from think import dream

        # facets.json contains both work and personal
        facets_data = [
            {"facet": "work", "activity": "Coding", "level": "high"},
            {"facet": "personal", "activity": "Email", "level": "low"},
        ]
        (segment_dir / "facets.json").write_text(json.dumps(facets_data))

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"prompt": prompt, "name": name, "config": config})
            return "mock-agent-id"

        def mock_get_muse_configs(has_tools=None, **kwargs):
            return {
                "test_agent": {
                    "schedule": "segment",
                    "multi_facet": True,
                    "tools": "journal",
                }
            }

        # Mock get_enabled_facets to only return "work" (personal is muted)
        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}}

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)

        count = dream.run_segment_agents("20240115", "120000_300")

        # Only work facet should be spawned, personal is muted
        assert count == 1
        assert len(spawned) == 1
        assert spawned[0]["config"]["facet"] == "work"
