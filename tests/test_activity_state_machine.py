# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the deterministic activity state machine."""


def _sense(
    content_type="coding",
    density="active",
    facets=None,
    summary="Working on code.",
    entities=None,
    meeting=False,
    speakers=None,
):
    """Build a Sense output payload for testing."""
    if facets is None:
        facets = [{"facet": "work", "activity": content_type, "level": "high"}]
    return {
        "density": density,
        "content_type": content_type,
        "activity_summary": summary,
        "entities": entities or [],
        "facets": facets,
        "meeting_detected": meeting,
        "speakers": speakers or [],
        "recommend": {},
    }


class TestNewActivity:
    def test_first_segment_starts_new(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        changes = sm.update(_sense(), "090000_300", "20260304")

        assert len(changes) == 1
        assert changes[0]["_change"] == "new"
        assert changes[0]["activity"] == "coding"
        assert changes[0]["state"] == "active"
        assert changes[0]["since"] == "090000_300"
        assert changes[0]["facet"] == "work"


class TestContinuation:
    def test_same_type_continues(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")
        changes = sm.update(_sense(summary="Still coding."), "090500_300", "20260304")

        assert len(changes) == 1
        assert changes[0]["_change"] == "continuing"
        assert changes[0]["since"] == "090000_300"
        assert changes[0]["description"] == "Still coding."


class TestContentTypeChange:
    def test_type_change_ends_old_starts_new(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        pending = sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        changes = sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        assert all(c["state"] != "ended" for c in pending)
        assert pending[0]["_change"] == "type_change_pending"
        assert len(changes) == 2
        ended = [c for c in changes if c["state"] == "ended"]
        started = [c for c in changes if c["state"] == "active"]
        assert len(ended) == 1
        assert ended[0]["_change"] == "ended_type_change"
        assert ended[0]["activity"] == "coding"
        assert len(started) == 1
        assert started[0]["_change"] == "new"
        assert started[0]["activity"] == "meeting"


class TestIdleTransition:
    def test_idle_ends_all(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")
        changes = sm.update(_sense(density="idle"), "090500_300", "20260304")

        ended = [c for c in changes if c["state"] == "ended"]
        assert len(ended) == 1
        assert ended[0]["_change"] == "ended_idle"
        assert sm.state == {}


class TestTimeGap:
    def test_gap_over_600s_resets(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")
        changes = sm.update(_sense(), "100600_300", "20260304")

        ended_gap = [c for c in changes if c["_change"] == "ended_gap"]
        assert len(ended_gap) == 1
        new = [c for c in changes if c["_change"] == "new"]
        assert len(new) == 1

    def test_gap_equal_600s_no_reset(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")
        changes = sm.update(_sense(), "091500_300", "20260304")

        assert all(c["_change"] != "ended_gap" for c in changes)


class TestDayBoundary:
    def test_day_change_resets(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "230000_300", "20260304")
        changes = sm.update(_sense(), "000000_300", "20260305")

        ended_gap = [c for c in changes if c["_change"] == "ended_gap"]
        assert len(ended_gap) == 1


class TestMultiFacet:
    def test_independent_facet_tracking(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        facets = [
            {"facet": "work", "activity": "coding", "level": "high"},
            {"facet": "personal", "activity": "browsing", "level": "low"},
        ]
        changes = sm.update(_sense(facets=facets), "090000_300", "20260304")

        assert len(changes) == 2
        facet_names = {c["facet"] for c in changes}
        assert facet_names == {"work", "personal"}


class TestFacetDisappearing:
    def test_facet_gone_emits_ended(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        two_facets = [
            {"facet": "work", "activity": "coding", "level": "high"},
            {"facet": "personal", "activity": "browsing", "level": "low"},
        ]
        sm.update(_sense(facets=two_facets), "090000_300", "20260304")
        one_facet = [{"facet": "work", "activity": "coding", "level": "high"}]
        pending = sm.update(_sense(facets=one_facet), "090500_300", "20260304")
        changes = sm.update(_sense(facets=one_facet), "091000_300", "20260304")

        assert pending[0]["_change"] == "facet_gone_pending"
        ended = [c for c in changes if c["_change"] == "ended_facet_gone"]
        assert len(ended) == 1
        assert ended[0]["facet"] == "personal"


class TestStateShape:
    def test_active_entries_use_persisted_field_names(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")

        assert set(sm.state) == {"work"}
        entry = sm.state["work"]
        assert "id" in entry
        assert "activity" in entry
        assert "state" in entry and entry["state"] == "active"
        assert "since" in entry
        assert "level" in entry
        assert "active_entities" in entry
        assert entry["facet"] == "work"
        assert entry["segment"] == "090000_300"
        assert entry["segments"] == ["090000_300"]
        assert entry["_change"] == "new"
        assert "_facet" not in entry
        assert "_segment" not in entry
        assert "_segments" not in entry


class TestGetCompletedActivities:
    def test_completed_format(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")
        completed = sm.get_completed_activities()

        assert len(completed) == 1
        rec = completed[0]
        assert "id" in rec
        assert "activity" in rec and rec["activity"] == "coding"
        assert "segments" in rec and isinstance(rec["segments"], list)
        assert rec["segments"] == ["090000_300", "090500_300"]
        assert "level_avg" in rec and isinstance(rec["level_avg"], float)
        assert "description" in rec
        assert "active_entities" in rec
        assert "created_at" in rec
        assert isinstance(rec["created_at"], int)


class TestSegmentAccumulation:
    def test_continuing_accumulates_segments(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="coding"), "090500_300", "20260304")
        sm.update(_sense(content_type="coding"), "091000_300", "20260304")
        # End by type change
        sm.update(_sense(content_type="meeting"), "091500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "092000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        rec = completed[0]
        assert rec["segments"] == [
            "090000_300",
            "090500_300",
            "091000_300",
            "091500_300",
        ]

    def test_ten_segments_produces_ten_keys(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        for i in range(10):
            minutes = i * 5
            seg = f"09{minutes:02d}00_300"
            sm.update(_sense(content_type="coding"), seg, "20260304")
        # End with idle
        sm.update(_sense(density="idle"), "095000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert len(completed[0]["segments"]) == 10

    def test_segments_accumulate_in_state(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="coding"), "090500_300", "20260304")

        assert sm.state["work"]["segments"] == ["090000_300", "090500_300"]


class TestEntityTracking:
    def test_extracts_names(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        entities = [
            {"type": "Person", "name": "Alice", "context": "colleague"},
            {"type": "Tool", "name": "VSCode", "context": "editor"},
        ]
        changes = sm.update(_sense(entities=entities), "090000_300", "20260304")

        assert changes[0]["active_entities"] == ["Alice", "VSCode"]

    def test_skips_blank_names(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        entities = [
            {"type": "Person", "name": "", "context": "unknown"},
            {"type": "Tool", "context": "no name key"},
        ]
        changes = sm.update(_sense(entities=entities), "090000_300", "20260304")

        assert changes[0]["active_entities"] == []


class TestSegmentAccumulationEdgeCases:
    """Regression tests for segment accumulation across all ending paths."""

    def test_idle_ending_preserves_accumulated_segments(self):
        """Idle transition after multi-segment activity must carry all segments."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="coding"), "090500_300", "20260304")
        sm.update(_sense(content_type="coding"), "091000_300", "20260304")
        # End via idle
        sm.update(_sense(density="idle"), "091500_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert completed[0]["segments"] == [
            "090000_300",
            "090500_300",
            "091000_300",
        ]

    def test_facet_gone_preserves_accumulated_segments(self):
        """Facet disappearing after multi-segment activity must carry all segments.

        Note: the state machine uses the top-level content_type for all facets,
        not the per-facet activity field. So both facets have activity=="coding".
        """
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        two = [
            {"facet": "work", "activity": "coding", "level": "high"},
            {"facet": "personal", "activity": "browsing", "level": "low"},
        ]
        one = [{"facet": "work", "activity": "coding", "level": "high"}]
        sm.update(_sense(facets=two), "090000_300", "20260304")
        sm.update(_sense(facets=two), "090500_300", "20260304")
        sm.update(_sense(facets=two), "091000_300", "20260304")
        # personal disappears
        sm.update(_sense(facets=one), "091500_300", "20260304")
        sm.update(_sense(facets=one), "092000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        # Activity type comes from top-level content_type, not per-facet
        assert completed[0]["activity"] == "coding"
        assert completed[0]["segments"] == [
            "090000_300",
            "090500_300",
            "091000_300",
            "091500_300",
        ]

    def test_gap_ending_preserves_accumulated_segments(self):
        """Time gap after multi-segment activity must carry all segments."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="coding"), "090500_300", "20260304")
        # 20-minute gap triggers reset (> 600s threshold)
        sm.update(_sense(content_type="coding"), "093000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert completed[0]["segments"] == ["090000_300", "090500_300"]

    def test_type_change_preserves_accumulated_segments(self):
        """Type change after multi-segment activity must carry all segments."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="coding"), "090500_300", "20260304")
        sm.update(_sense(content_type="coding"), "091000_300", "20260304")
        sm.update(_sense(content_type="coding"), "091500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "092000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "092500_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert completed[0]["segments"] == [
            "090000_300",
            "090500_300",
            "091000_300",
            "091500_300",
            "092000_300",
        ]

    def test_single_segment_activity_has_one_segment(self):
        """Activity lasting exactly one segment has segments: [that_segment]."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(density="idle"), "090500_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert completed[0]["segments"] == ["090000_300"]

    def test_duplicate_segment_key_not_appended(self):
        """Same segment key processed twice doesn't duplicate in _segments."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        # Same segment key again (shouldn't happen in practice, but defensive)
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 1
        assert completed[0]["segments"] == ["090000_300", "090500_300"]

    def test_multi_facet_simultaneous_ending_all_have_segments(self):
        """Multiple facets ending simultaneously each have their own segments."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        two = [
            {"facet": "work", "activity": "coding", "level": "high"},
            {"facet": "personal", "activity": "browsing", "level": "low"},
        ]
        sm.update(_sense(facets=two), "090000_300", "20260304")
        sm.update(_sense(facets=two), "090500_300", "20260304")
        sm.update(_sense(facets=two), "091000_300", "20260304")
        # Both end via idle
        sm.update(_sense(density="idle"), "091500_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 2
        for rec in completed:
            assert rec["segments"] == [
                "090000_300",
                "090500_300",
                "091000_300",
            ]


class TestCompletedRecordFields:
    """Verify all fields in completed records have correct types and values."""

    def test_created_at_is_millisecond_int(self):
        """created_at must be an integer in milliseconds (not ISO string)."""
        import time

        from think.activity_state_machine import ActivityStateMachine

        before = int(time.time() * 1000)
        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")
        after = int(time.time() * 1000)

        rec = sm.get_completed_activities()[0]
        assert isinstance(rec["created_at"], int)
        assert before <= rec["created_at"] <= after

    def test_created_at_comparable_to_float_cutoff(self):
        """created_at must be comparable with float (routes.py cutoff_ts)."""
        from datetime import datetime, timedelta

        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        rec = sm.get_completed_activities()[0]
        # Simulate routes.py cutoff calculation
        cutoff_ts = (datetime.now() - timedelta(hours=4)).timestamp() * 1000
        # This must not raise TypeError
        assert isinstance(rec["created_at"] < cutoff_ts, bool)
        # Also must support division (routes.py line 283)
        dt = datetime.fromtimestamp(rec["created_at"] / 1000)
        assert isinstance(dt, datetime)

    def test_level_avg_is_float(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(
            _sense(
                content_type="coding",
                facets=[{"facet": "work", "activity": "coding", "level": "high"}],
            ),
            "090000_300",
            "20260304",
        )
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        rec = sm.get_completed_activities()[0]
        assert isinstance(rec["level_avg"], float)
        assert rec["level_avg"] == 1.0  # "high" maps to 1.0

    def test_all_required_fields_present(self):
        """Completed record must have every field run_activity_prompts expects."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        rec = sm.get_completed_activities()[0]
        required = {
            "id",
            "activity",
            "segments",
            "level_avg",
            "description",
            "active_entities",
            "created_at",
        }
        assert required.issubset(rec.keys())
        # No internal _fields should leak
        assert not any(k.startswith("_") for k in rec.keys())

    def test_active_entities_preserved_in_completed(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        entities = [
            {"type": "Person", "name": "Alice", "context": "dev"},
            {"type": "Tool", "name": "VSCode", "context": "editor"},
        ]
        sm.update(
            _sense(content_type="coding", entities=entities), "090000_300", "20260304"
        )
        sm.update(_sense(density="idle"), "090500_300", "20260304")

        rec = sm.get_completed_activities()[0]
        assert rec["active_entities"] == ["Alice", "VSCode"]


class TestCumulativeCompletedList:
    """Verify get_completed_activities() is cumulative and correct."""

    def test_multiple_endings_accumulate(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        # Activity 1: coding
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")
        # Activity 2: meeting
        sm.update(_sense(content_type="coding"), "091500_300", "20260304")
        sm.update(_sense(content_type="coding"), "092000_300", "20260304")

        completed = sm.get_completed_activities()
        assert len(completed) == 2
        assert completed[0]["activity"] == "coding"
        assert completed[1]["activity"] == "meeting"

    def test_completed_list_is_copy(self):
        """get_completed_activities returns a copy, not internal state."""
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        sm.update(_sense(content_type="meeting"), "091000_300", "20260304")

        list1 = sm.get_completed_activities()
        list2 = sm.get_completed_activities()
        assert list1 is not list2
        assert list1 == list2
