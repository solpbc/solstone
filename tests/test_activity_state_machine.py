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
        assert changes[0]["_facet"] == "work"


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
        changes = sm.update(
            _sense(content_type="meeting"), "090500_300", "20260304"
        )

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
        assert sm.get_current_state() == []


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
        facet_names = {c["_facet"] for c in changes}
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
        changes = sm.update(_sense(facets=one_facet), "090500_300", "20260304")

        ended = [c for c in changes if c["_change"] == "ended_facet_gone"]
        assert len(ended) == 1
        assert ended[0]["_facet"] == "personal"


class TestGetCurrentState:
    def test_returns_clean_entries(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(), "090000_300", "20260304")
        state = sm.get_current_state()

        assert len(state) == 1
        entry = state[0]
        assert "id" in entry
        assert "activity" in entry
        assert "state" in entry and entry["state"] == "active"
        assert "since" in entry
        assert "level" in entry
        assert "active_entities" in entry
        assert "_change" not in entry
        assert "_facet" not in entry
        assert "_segment" not in entry


class TestGetCompletedActivities:
    def test_completed_format(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        sm.update(_sense(content_type="coding"), "090000_300", "20260304")
        sm.update(_sense(content_type="meeting"), "090500_300", "20260304")
        completed = sm.get_completed_activities()

        assert len(completed) == 1
        rec = completed[0]
        assert "id" in rec
        assert "activity" in rec and rec["activity"] == "coding"
        assert "segments" in rec and isinstance(rec["segments"], list)
        assert "level_avg" in rec and isinstance(rec["level_avg"], float)
        assert "description" in rec
        assert "active_entities" in rec
        assert "created_at" in rec


class TestPseudoFacet:
    def test_no_facets_uses_underscore(self):
        from think.activity_state_machine import ActivityStateMachine

        sm = ActivityStateMachine()
        changes = sm.update(_sense(facets=[]), "090000_300", "20260304")

        assert len(changes) == 1
        assert changes[0]["_facet"] == "__"


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
