# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for segment orchestration in dream."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def segment_dir(tmp_path, monkeypatch):
    """Create a temporary journal with a segment directory."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"
    segment_path = day_dir / "default" / "120000_300"
    segment_path.mkdir(parents=True)
    (segment_path / "agents").mkdir(parents=True)

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return segment_path


def _segment_configs(*names: str) -> dict[str, dict]:
    configs = {
        "sense": {
            "priority": 10,
            "type": "generate",
            "output": "json",
            "schedule": "segment",
        },
        "entities": {
            "priority": 20,
            "type": "cogitate",
            "schedule": "segment",
        },
        "screen": {
            "priority": 20,
            "type": "generate",
            "output": "md",
            "schedule": "segment",
        },
        "speaker_attribution": {
            "priority": 20,
            "type": "cogitate",
            "schedule": "segment",
        },
        "pulse": {
            "priority": 30,
            "type": "cogitate",
            "schedule": "segment",
        },
    }
    return {name: dict(configs[name]) for name in names}


def _write_sense_output(segment_dir: Path, sense_json: dict) -> None:
    (segment_dir / "agents" / "sense.json").write_text(
        json.dumps(sense_json),
        encoding="utf-8",
    )


class TestLoadSegmentFacets:
    """Tests for load_segment_facets helper function."""

    def test_missing_file_returns_empty(self, segment_dir):
        from think.facets import load_segment_facets

        assert load_segment_facets("20240115", "120000_300") == []

    def test_empty_file_returns_empty(self, segment_dir):
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("")
        assert load_segment_facets("20240115", "120000_300") == []

    def test_empty_array_returns_empty(self, segment_dir):
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("[]")
        assert load_segment_facets("20240115", "120000_300") == []

    def test_valid_facets_extracted(self, segment_dir):
        from think.facets import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Code review", "level": "high"},
            {"facet": "personal", "activity": "Email check", "level": "low"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        assert load_segment_facets("20240115", "120000_300") == ["work", "personal"]

    def test_malformed_json_returns_empty(self, segment_dir, caplog):
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text("{ invalid json")
        assert load_segment_facets("20240115", "120000_300") == []
        assert "Failed to parse facets.json" in caplog.text

    def test_non_array_returns_empty(self, segment_dir, caplog):
        from think.facets import load_segment_facets

        (segment_dir / "agents" / "facets.json").write_text('{"facet": "work"}')
        assert load_segment_facets("20240115", "120000_300") == []
        assert "not an array" in caplog.text

    def test_missing_facet_field_skipped(self, segment_dir):
        from think.facets import load_segment_facets

        facets_data = [
            {"facet": "work", "activity": "Coding"},
            {"activity": "Unknown"},
            {"facet": "personal", "activity": "Email"},
        ]
        (segment_dir / "agents" / "facets.json").write_text(json.dumps(facets_data))

        assert load_segment_facets("20240115", "120000_300") == ["work", "personal"]


class TestRunSegmentSense:
    def test_sense_runs_first(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs("sense", "entities"),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        success, failed, failed_names = dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert spawned == ["sense", "entities"]
        assert success == 2
        assert failed == 0
        assert failed_names == []

    def test_idle_segment_returns_early(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        updates = []

        class StubStateMachine:
            def update(self, sense_output, segment, day):
                updates.append((sense_output, segment, day))
                return []

            def get_current_state(self):
                return []

            def get_completed_activities(self):
                return []

        _write_sense_output(
            segment_dir,
            {"density": "idle", "recommend": {"screen_record": True}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(
                "sense", "entities", "screen"
            ),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        success, failed, _ = dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
            state_machine=StubStateMachine(),
        )

        assert spawned == ["sense"]
        assert success == 1
        assert failed == 0
        assert updates == [
            (
                {"density": "idle", "recommend": {"screen_record": True}, "facets": []},
                "120000_300",
                "20240115",
            )
        ]
        density = json.loads((segment_dir / "agents" / "density.json").read_text())
        assert density["classification"] == "idle"

        # Verify activity state persisted even on idle path
        activity_state_path = (
            segment_dir.parent.parent.parent / "awareness" / "activity_state.json"
        )
        assert activity_state_path.exists()
        state_data = json.loads(activity_state_path.read_text())
        assert state_data == []

    def test_conditional_screen_dispatch(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {"screen_record": True}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(
                "sense", "entities", "screen"
            ),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert spawned == ["sense", "entities", "screen"]

    @pytest.mark.parametrize(
        ("has_embeddings", "expected"),
        [
            (False, ["sense", "entities"]),
            (True, ["sense", "entities", "speaker_attribution"]),
        ],
    )
    def test_conditional_speaker_attribution(
        self,
        segment_dir,
        monkeypatch,
        has_embeddings,
        expected,
    ):
        from think import dream

        spawned = []
        if has_embeddings:
            (segment_dir / "audio.npz").write_bytes(b"npz")

        _write_sense_output(
            segment_dir,
            {
                "density": "active",
                "recommend": {"speaker_attribution": True},
                "facets": [],
            },
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(
                "sense",
                "entities",
                "speaker_attribution",
            ),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert spawned == expected

    def test_refresh_bypasses_idle(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "idle", "recommend": {}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs("sense", "entities"),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        success, failed, failed_names = dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=True,
            verbose=False,
            stream="default",
        )

        assert spawned == ["sense", "entities"]
        assert success == 2
        assert failed == 0
        assert failed_names == []

    def test_entities_always_runs(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {"screen_record": False}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(
                "sense", "entities", "screen"
            ),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert "entities" in spawned
        assert "screen" not in spawned

    def test_pulse_dispatch(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {"pulse_update": True}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(
                "sense", "entities", "pulse"
            ),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert spawned == ["sense", "entities", "pulse"]

    def test_sense_failure_stops_orchestrator(self, segment_dir, monkeypatch):
        from think import dream

        spawned = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs("sense", "entities"),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: spawned.append(name) or f"agent-{name}",
        )

        def mock_wait_for_agents(agent_ids, timeout=600):
            return ({agent_ids[0]: "error"}, [])

        monkeypatch.setattr(dream, "wait_for_agents", mock_wait_for_agents)
        monkeypatch.setattr(dream, "_callosum", None)

        success, failed, failed_names = dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert spawned == ["sense"]
        assert success == 0
        assert failed == 1
        assert failed_names == ["sense (error)"]

    def test_activity_state_machine_updated(self, segment_dir, monkeypatch):
        from think import dream

        updates = []
        activity_calls = []

        class StubStateMachine:
            def update(self, sense_output, segment, day):
                updates.append((sense_output, segment, day))
                return [{"state": "ended", "id": "coding_120000_300", "_facet": "work"}]

            def get_current_state(self):
                return [{"facet": "work", "state": "active", "id": "coding_120000_300"}]

            def get_completed_activities(self):
                return [{"id": "coding_120000_300", "activity": "coding", "segments": ["120000_300"], "level_avg": 0.5, "description": "coding", "active_entities": [], "created_at": 1713200000000}]

        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {}, "facets": []},
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs("sense", "entities"),
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(
            dream,
            "run_activity_prompts",
            lambda **kwargs: activity_calls.append(kwargs) or True,
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
            state_machine=StubStateMachine(),
        )

        assert updates == [
            (
                {"density": "active", "recommend": {}, "facets": []},
                "120000_300",
                "20240115",
            )
        ]
        assert activity_calls == [
            {
                "day": "20240115",
                "activity_id": "coding_120000_300",
                "facet": "work",
                "refresh": False,
                "verbose": False,
                "max_concurrency": 2,
            }
        ]
        activity_state_path = (
            segment_dir.parent.parent.parent / "awareness" / "activity_state.json"
        )
        assert activity_state_path.exists()
        state_data = json.loads(activity_state_path.read_text())
        assert state_data == [
            {"facet": "work", "state": "active", "id": "coding_120000_300"}
        ]

    def test_generator_triggers_incremental_indexing(self, segment_dir, monkeypatch):
        from think import dream

        indexer_calls = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {}, "facets": []},
        )
        (segment_dir / "agents" / "entities.md").write_text(
            "entities", encoding="utf-8"
        )

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: {
                **_segment_configs("sense"),
                "entities": {
                    "priority": 20,
                    "type": "generate",
                    "output": "md",
                    "schedule": "segment",
                },
            },
        )
        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(
            dream,
            "run_queued_command",
            lambda cmd, day, timeout=60: indexer_calls.append(cmd) or True,
        )
        monkeypatch.setattr(dream, "_callosum", None)

        dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert len(indexer_calls) == 1
        assert indexer_calls[0][:2] == ["sol", "indexer"]
        assert "--rescan-file" in indexer_calls[0]

    def test_send_failure_counted(self, segment_dir, monkeypatch):
        from think import dream

        calls = []
        _write_sense_output(
            segment_dir,
            {"density": "active", "recommend": {}, "facets": []},
        )

        def mock_cortex_request(prompt, name, config=None):
            calls.append(name)
            if name == "sense":
                return "agent-sense"
            return None

        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs("sense", "entities"),
        )
        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "_SEND_RETRY_DELAYS", (0.0, 0.0))
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)

        success, failed, failed_names = dream.run_segment_sense(
            "20240115",
            "120000_300",
            refresh=False,
            verbose=False,
            stream="default",
        )

        assert calls[0] == "sense"
        assert calls[1:] == ["entities", "entities", "entities"]
        assert success == 1
        assert failed == 1
        assert failed_names == ["entities (send)"]


class TestCortexRequestRetry:
    """Tests for _cortex_request_with_retry."""

    def test_succeeds_on_first_try(self, monkeypatch):
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
        from think import dream

        calls = []

        def mock_cortex_request(**kwargs):
            calls.append(kwargs)
            return None

        monkeypatch.setattr(dream, "cortex_request", mock_cortex_request)
        monkeypatch.setattr(dream, "_SEND_RETRY_DELAYS", (0.0, 0.0))

        result = dream._cortex_request_with_retry(prompt="hi", name="test")

        assert result is None
        assert len(calls) == 3


class TestStreamAutoResolution:
    """Tests for stream resolution in segment mode."""

    def test_auto_resolves_stream_from_filesystem(self, segment_dir, monkeypatch):
        mod = importlib.import_module("think.dream")
        calls: list[dict] = []

        class MockCallosumConnection:
            def __init__(self, *args, **kwargs):
                pass

            def start(self, callback=None):
                return None

            def stop(self):
                return None

        def mock_run_segment_sense(day, segment, refresh, verbose, **kwargs):
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
        monkeypatch.setattr(mod, "run_segment_sense", mock_run_segment_sense)
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(
            mod, "run_queued_command", lambda cmd, day, timeout=600: True
        )
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
        monkeypatch.setattr(
            "sys.argv",
            ["sol dream", "--day", "20240115", "--segment", "120000_300"],
        )

        mod.main()

        assert len(calls) == 1
        assert calls[0]["stream"] == "mystream"

    def test_segment_not_found_exits(self, segment_dir, monkeypatch):
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
            mod, "run_segment_sense", lambda *args, **kwargs: (1, 0, [])
        )
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
        monkeypatch.setattr(
            "sys.argv",
            ["sol dream", "--day", "20240115", "--segment", "999999_300"],
        )

        with pytest.raises(SystemExit) as excinfo:
            mod.main()

        assert excinfo.value.code != 0

    def test_explicit_stream_skips_filesystem_lookup(self, segment_dir, monkeypatch):
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

        def mock_run_segment_sense(day, segment, refresh, verbose, **kwargs):
            calls.append(kwargs)
            return (1, 0, [])

        monkeypatch.setattr(mod, "iter_segments", mock_iter_segments)
        monkeypatch.setattr(mod, "run_segment_sense", mock_run_segment_sense)
        monkeypatch.setattr(mod, "check_callosum_available", lambda: True)
        monkeypatch.setattr(mod, "run_command", lambda cmd, day: True)
        monkeypatch.setattr(
            mod, "run_queued_command", lambda cmd, day, timeout=600: True
        )
        monkeypatch.setattr(mod, "CallosumConnection", MockCallosumConnection)
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
