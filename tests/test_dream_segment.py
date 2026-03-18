# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for segment facet functionality in dream module."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def segment_dir(tmp_path, monkeypatch):
    """Create a temporary journal with segment directory (stream layout)."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"
    segment_path = day_dir / "default" / "120000_300"
    segment_path.mkdir(parents=True)
    (segment_path / "agents").mkdir(parents=True)

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    return segment_path


class TestLoadSegmentFacets:
    """Tests for load_segment_facets helper function."""

    def test_missing_file_returns_empty(self, segment_dir, monkeypatch):
        """Missing facets.json returns empty list."""
        from think.facets import load_segment_facets

        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_empty_file_returns_empty(self, segment_dir, monkeypatch):
        """Empty facets.json returns empty list."""
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_empty_array_returns_empty(self, segment_dir, monkeypatch):
        """Empty JSON array returns empty list."""
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("[]")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []

    def test_valid_facets_extracted(self, segment_dir, monkeypatch):
        """Valid facets.json extracts facet IDs."""
        from think.facets import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Code review", "level": "high"},
            {"facet": "personal", "activity": "Email check", "level": "low"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        result = load_segment_facets("20240115", "120000_300")
        assert result == ["work", "personal"]

    def test_malformed_json_returns_empty(self, segment_dir, monkeypatch, caplog):
        """Malformed JSON returns empty list with error logged."""
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("{ invalid json")
        result = load_segment_facets("20240115", "120000_300")
        assert result == []
        assert "Failed to parse facets.json" in caplog.text

    def test_non_array_returns_empty(self, segment_dir, monkeypatch, caplog):
        """Non-array JSON returns empty list with warning."""
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text('{"facet": "work"}')
        result = load_segment_facets("20240115", "120000_300")
        assert result == []
        assert "not an array" in caplog.text

    def test_missing_facet_field_skipped(self, segment_dir, monkeypatch):
        """Items without facet field are skipped."""
        from think.facets import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Coding"},
            {"activity": "Unknown"},  # Missing facet field
            {"facet": "personal", "activity": "Email"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        result = load_segment_facets("20240115", "120000_300")
        assert result == ["work", "personal"]


class TestRunPromptsByPriority:
    """Tests for unified priority execution."""

    def test_prompts_grouped_by_priority(self, segment_dir, monkeypatch):
        """Prompts are grouped and executed by priority order."""
        from think import dream

        spawned = []
        wait_calls = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"name": name, "config": config})
            return f"agent-{name}"

        def mock_wait_for_agents(agent_ids, timeout=600):
            wait_calls.append(agent_ids)
            return ({aid: "finish" for aid in agent_ids}, [])  # All complete

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "low_priority": {
                    "priority": 10,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
                "high_priority": {
                    "priority": 90,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
                "mid_priority": {
                    "priority": 50,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
            }

        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}}

        def mock_get_active_facets(day):
            return set()

        def mock_run_queued_command(cmd, day, timeout=60):
            return True

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)
        monkeypatch.setattr(dream, "get_active_facets", mock_get_active_facets)
        monkeypatch.setattr(dream, "run_queued_command", mock_run_queued_command)

        success, failed, failed_names = dream.run_prompts_by_priority(
            "20240115", "120000_300", refresh=False, verbose=False
        )

        assert success == 3
        assert failed == 0
        assert failed_names == []

        # Verify wait was called 3 times (once per priority group)
        assert len(wait_calls) == 3

        # Verify order: priority 10 first, then 50, then 90
        spawn_order = [s["name"] for s in spawned]
        assert spawn_order.index("low_priority") < spawn_order.index("mid_priority")
        assert spawn_order.index("mid_priority") < spawn_order.index("high_priority")

    def test_multi_facet_prompt_spawns_per_facet(self, segment_dir, monkeypatch):
        """Multi-facet prompts spawn once per active facet."""
        from think import dream

        # Set up facets.json
        facets_data = [
            {"facet": "work", "activity": "Coding", "level": "high"},
            {"facet": "personal", "activity": "Email", "level": "low"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"name": name, "config": config})
            return f"agent-{len(spawned)}"

        def mock_wait_for_agents(agent_ids, timeout=600):
            return ({aid: "finish" for aid in agent_ids}, [])

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "multi_facet_prompt": {
                    "priority": 10,
                    "type": "cogitate",
                    "tools": "journal",
                    "multi_facet": True,
                    "schedule": "segment",
                },
            }

        def mock_get_enabled_facets():
            return {
                "work": {"title": "Work"},
                "personal": {"title": "Personal"},
            }

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)

        success, failed, failed_names = dream.run_prompts_by_priority(
            "20240115", "120000_300", refresh=False, verbose=False
        )

        assert success == 2  # One per facet
        assert len(spawned) == 2

        facets_spawned = [s["config"]["facet"] for s in spawned]
        assert "work" in facets_spawned
        assert "personal" in facets_spawned

    def test_muted_facets_filtered(self, segment_dir, monkeypatch):
        """Muted facets are filtered out from multi-facet prompts."""
        from think import dream

        # facets.json contains both work and personal
        facets_data = [
            {"facet": "work", "activity": "Coding", "level": "high"},
            {"facet": "personal", "activity": "Email", "level": "low"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        spawned = []

        def mock_cortex_request(prompt, name, config=None):
            spawned.append({"name": name, "config": config})
            return f"agent-{len(spawned)}"

        def mock_wait_for_agents(agent_ids, timeout=600):
            return ({aid: "finish" for aid in agent_ids}, [])

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "test_prompt": {
                    "priority": 10,
                    "type": "cogitate",
                    "tools": "journal",
                    "multi_facet": True,
                    "schedule": "segment",
                },
            }

        # Only work is enabled (personal is muted)
        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}}

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)

        success, failed, failed_names = dream.run_prompts_by_priority(
            "20240115", "120000_300", refresh=False, verbose=False
        )

        # Only work facet should be spawned, personal is muted
        assert success == 1
        assert len(spawned) == 1
        assert spawned[0]["config"]["facet"] == "work"

    def test_generator_triggers_incremental_indexing(self, segment_dir, monkeypatch):
        """Generators trigger incremental indexing after completion."""
        from think import dream

        indexer_calls = []

        def mock_cortex_request(prompt, name, config=None):
            return f"agent-{name}"

        def mock_wait_for_agents(agent_ids, timeout=600):
            return ({aid: "finish" for aid in agent_ids}, [])

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "test_generator": {
                    "priority": 10,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
            }

        def mock_get_enabled_facets():
            return {"work": {"title": "Work"}}

        def mock_get_active_facets(day):
            return set()

        def mock_run_queued_command(cmd, day, timeout=60):
            indexer_calls.append(cmd)
            return True

        # Create the output file so indexer is triggered
        output_file = segment_dir / "agents" / "test_generator.md"
        output_file.write_text("test output")

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)
        monkeypatch.setattr(dream, "get_active_facets", mock_get_active_facets)
        monkeypatch.setattr(dream, "run_queued_command", mock_run_queued_command)

        dream.run_prompts_by_priority(
            "20240115", "120000_300", refresh=False, verbose=False, stream="default"
        )

        # Verify indexer was called with --rescan-file
        assert len(indexer_calls) == 1
        assert indexer_calls[0][0] == "sol"
        assert indexer_calls[0][1] == "indexer"
        assert "--rescan-file" in indexer_calls[0]


class TestCortexRequestRetry:
    """Tests for _cortex_request_with_retry."""

    def test_succeeds_on_first_try(self, monkeypatch):
        """No retries when first call succeeds."""
        from think import dream

        calls = []

        def mock_cortex_request(**kwargs):
            calls.append(kwargs)
            return "agent-1"

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)

        result = dream._cortex_request_with_retry(prompt="hi", name="test")

        assert result == "agent-1"
        assert len(calls) == 1

    def test_succeeds_on_retry(self, monkeypatch):
        """Returns agent_id when a retry succeeds after initial failure."""
        from think import dream

        calls = []

        def mock_cortex_request(**kwargs):
            calls.append(kwargs)
            return None if len(calls) <= 1 else "agent-2"

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "_SEND_RETRY_DELAYS", (0.0, 0.0))

        result = dream._cortex_request_with_retry(prompt="hi", name="test")

        assert result == "agent-2"
        assert len(calls) == 2

    def test_returns_none_after_all_retries(self, monkeypatch):
        """Returns None when all attempts fail."""
        from think import dream

        calls = []

        def mock_cortex_request(**kwargs):
            calls.append(kwargs)
            return None

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "_SEND_RETRY_DELAYS", (0.0, 0.0))

        result = dream._cortex_request_with_retry(prompt="hi", name="test")

        assert result is None
        assert len(calls) == 3  # 1 initial + 2 retries

    def test_send_failure_counted_as_spawn_failure(self, segment_dir, monkeypatch):
        """When cortex_request returns None, dream counts it as a spawn failure."""
        from think import dream

        def mock_cortex_request(**kwargs):
            return None

        def mock_wait_for_agents(agent_ids, timeout=600):
            return ({}, [])

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "broken_agent": {
                    "priority": 10,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
            }

        def mock_get_enabled_facets():
            return {}

        def mock_get_active_facets(day):
            return set()

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "_SEND_RETRY_DELAYS", (0.0, 0.0))
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", mock_get_enabled_facets)
        monkeypatch.setattr(dream, "get_active_facets", mock_get_active_facets)

        success, failed, failed_names = dream.run_prompts_by_priority(
            "20240115", "120000_300", refresh=False, verbose=False
        )

        assert success == 0
        assert failed == 1
        assert any("send" in n for n in failed_names)


class TestStreamAutoResolution:
    """Tests for stream resolution in segment mode."""

    def test_auto_resolves_stream_from_filesystem(self, segment_dir, monkeypatch):
        """main() resolves stream from iter_segments when omitted."""
        mod = importlib.import_module("think.dream")
        calls: list[dict] = []

        class MockCallosumConnection:
            def __init__(self, *args, **kwargs):
                pass

            def start(self, callback=None):
                return None

            def stop(self):
                return None

        def mock_run_prompts_by_priority(day, segment, refresh, verbose, **kwargs):
            calls.append(
                {
                    "day": day,
                    "segment": segment,
                    "refresh": refresh,
                    "verbose": verbose,
                    **kwargs,
                }
            )
            return (1, 0, [])

        monkeypatch.setattr(
            mod,
            "iter_segments",
            lambda day: [("mystream", "120000_300", Path("/tmp/segment"))],
        )
        monkeypatch.setattr(
            mod, "run_prompts_by_priority", mock_run_prompts_by_priority
        )
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(
            mod, "run_queued_command", lambda cmd, day, timeout=600: True
        )
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
        monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
        monkeypatch.setattr(
            "sys.argv",
            ["sol dream", "--day", "20240115", "--segment", "120000_300"],
        )

        mod.main()

        assert len(calls) == 1
        assert calls[0]["stream"] == "mystream"

    def test_segment_not_found_exits(self, segment_dir, monkeypatch):
        """main() errors when segment cannot be resolved to a stream."""
        mod = importlib.import_module("think.dream")

        class MockCallosumConnection:
            def __init__(self, *args, **kwargs):
                pass

            def start(self, callback=None):
                return None

            def stop(self):
                return None

        monkeypatch.setattr(mod, "iter_segments", lambda day: [])
        monkeypatch.setattr(
            mod, "run_prompts_by_priority", lambda *args, **kwargs: (1, 0, [])
        )
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
        monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
        monkeypatch.setattr(
            "sys.argv",
            ["sol dream", "--day", "20240115", "--segment", "999999_300"],
        )

        with pytest.raises(SystemExit) as excinfo:
            mod.main()

        assert excinfo.value.code != 0

    def test_explicit_stream_skips_filesystem_lookup(self, segment_dir, monkeypatch):
        """Explicit --stream bypasses iter_segments lookup."""
        mod = importlib.import_module("think.dream")
        iter_calls = 0
        calls: list[dict] = []

        class MockCallosumConnection:
            def __init__(self, *args, **kwargs):
                pass

            def start(self, callback=None):
                return None

            def stop(self):
                return None

        def mock_iter_segments(day):
            nonlocal iter_calls
            iter_calls += 1
            return [("mystream", "120000_300", Path("/tmp/segment"))]

        def mock_run_prompts_by_priority(day, segment, refresh, verbose, **kwargs):
            calls.append(kwargs)
            return (1, 0, [])

        monkeypatch.setattr(mod, "iter_segments", mock_iter_segments)
        monkeypatch.setattr(
            mod, "run_prompts_by_priority", mock_run_prompts_by_priority
        )
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(
            mod, "run_queued_command", lambda cmd, day, timeout=600: True
        )
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
        monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
        monkeypatch.setattr(
            "sys.argv",
            [
                "sol dream",
                "--day",
                "20240115",
                "--segment",
                "120000_300",
                "--stream",
                "explicit_stream",
            ],
        )

        mod.main()

        assert iter_calls == 0
        assert len(calls) == 1
        assert calls[0]["stream"] == "explicit_stream"

    def test_no_timeout_passes_none_to_wait(self, segment_dir, monkeypatch):
        """run_prompts_by_priority forwards timeout=None to wait_for_agents."""
        from think import dream

        wait_calls = []

        def mock_cortex_request(prompt, name, config=None):
            return f"agent-{name}"

        def mock_wait_for_agents(agent_ids, timeout=600):
            wait_calls.append(timeout)
            return ({aid: "finish" for aid in agent_ids}, [])

        def mock_get_muse_configs(schedule=None, **kwargs):
            return {
                "test_generator": {
                    "priority": 10,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
            }

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "get_muse_configs", mock_get_muse_configs)
        monkeypatch.setattr(dream, "get_enabled_facets", lambda: {})
        monkeypatch.setattr(dream, "get_active_facets", lambda day: set())
        monkeypatch.setattr(dream, "_callosum", None)
        monkeypatch.setattr(
            dream, "run_queued_command", lambda cmd, day, timeout=60: True
        )

        dream.run_prompts_by_priority(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            timeout=None,
        )

        assert wait_calls == [None]
