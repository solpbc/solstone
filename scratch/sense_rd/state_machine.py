"""
Activity state machine for replacing LLM-based activity_state agent.

Maintains per-facet activity state across segments, producing output
compatible with the existing activity_state.json format.
"""

import re


def _parse_segment_time(segment_key: str) -> int | None:
    """
    Parse a segment key like "091500_420" into absolute seconds from midnight.
    Format: HHMMSS_duration
    Returns start time in seconds, or None if unparseable.
    """
    match = re.match(r"(\d{2})(\d{2})(\d{2})_(\d+)", segment_key)
    if not match:
        return None
    h, m, s, _ = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s)


def _segment_end_time(segment_key: str) -> int | None:
    """Return end time in seconds from midnight."""
    match = re.match(r"(\d{2})(\d{2})(\d{2})_(\d+)", segment_key)
    if not match:
        return None
    h, m, s, dur = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(dur)


def _make_activity_id(content_type: str, segment_key: str) -> str:
    """Generate an activity ID like 'meeting_091500_420'."""
    return f"{content_type}_{segment_key}"


class ActivityStateMachine:
    """
    Tracks per-facet activity state across segments.

    State format per facet:
    {
        "id": "meeting_091500_420",
        "activity": "meeting",
        "state": "active" | "ended",
        "since": "091500_420",
        "description": "...",
        "level": "high" | "medium" | "low",
        "active_entities": [...]
    }
    """

    # Gap threshold: if more than 10 minutes between segments, end all active
    GAP_THRESHOLD_SECONDS = 600

    def __init__(self):
        # {facet_id: state_dict}
        self.state: dict[str, dict] = {}
        self.last_segment_key: str | None = None
        self.last_segment_day: str | None = None
        self.history: list[dict] = []  # All state changes

    def update(self, sense_output: dict, segment_key: str, day: str,
               previous_segment_key: str | None = None) -> list[dict]:
        """
        Given Sense output for a segment, update activity state.

        Args:
            sense_output: Parsed JSON from the Sense agent
            segment_key: e.g. "091500_420"
            day: e.g. "20260201"
            previous_segment_key: Previous segment key (for gap detection)

        Returns:
            List of state entries (new, continuing, ended) for this segment.
        """
        changes = []

        # Check for day change or time gap — end all active
        if self._should_reset(segment_key, day, previous_segment_key):
            for facet_id, state in self.state.items():
                if state["state"] == "active":
                    state["state"] = "ended"
                    changes.append({**state, "_change": "ended_gap"})
            self.state.clear()

        content_type = sense_output.get("content_type", "idle")
        density = sense_output.get("density", "idle")
        activity_summary = sense_output.get("activity_summary", "")
        entities = sense_output.get("entities", [])
        facets = sense_output.get("facets", [])
        entity_names = [e.get("name", "") for e in entities]

        # If idle density, end all active states
        if density == "idle":
            for facet_id, state in list(self.state.items()):
                if state["state"] == "active":
                    state["state"] = "ended"
                    changes.append({**state, "_change": "ended_idle"})
            self.state.clear()
            self.last_segment_key = segment_key
            self.last_segment_day = day
            self.history.extend(changes)
            return changes

        # Process each facet from Sense output
        active_facet_ids = set()

        for facet_data in facets:
            facet_id = facet_data.get("facet", "")
            facet_activity = facet_data.get("activity", "")
            facet_level = facet_data.get("level", "medium")
            active_facet_ids.add(facet_id)

            if facet_id in self.state and self.state[facet_id]["state"] == "active":
                # Same facet still active — check if content_type changed
                existing = self.state[facet_id]
                if existing["activity"] == content_type:
                    # Continuing: same content type + same facet
                    existing["description"] = facet_activity or activity_summary
                    existing["level"] = facet_level
                    existing["active_entities"] = entity_names
                    changes.append({**existing, "_change": "continuing"})
                else:
                    # Content type changed within same facet — end old, start new
                    existing["state"] = "ended"
                    changes.append({**existing, "_change": "ended_type_change"})

                    new_state = {
                        "id": _make_activity_id(content_type, segment_key),
                        "activity": content_type,
                        "state": "active",
                        "since": segment_key,
                        "description": facet_activity or activity_summary,
                        "level": facet_level,
                        "active_entities": entity_names,
                    }
                    self.state[facet_id] = new_state
                    changes.append({**new_state, "_change": "new"})
            else:
                # New facet or previously ended — start new
                new_state = {
                    "id": _make_activity_id(content_type, segment_key),
                    "activity": content_type,
                    "state": "active",
                    "since": segment_key,
                    "description": facet_activity or activity_summary,
                    "level": facet_level,
                    "active_entities": entity_names,
                }
                self.state[facet_id] = new_state
                changes.append({**new_state, "_change": "new"})

        # End facets that were active but not in current Sense output
        for facet_id in list(self.state.keys()):
            if facet_id not in active_facet_ids and self.state[facet_id]["state"] == "active":
                self.state[facet_id]["state"] = "ended"
                changes.append({**self.state[facet_id], "_change": "ended_facet_gone"})
                del self.state[facet_id]

        # If no facets from Sense but there is activity, use content_type as a pseudo-facet
        if not facets and density != "idle":
            pseudo_facet = f"__{content_type}"
            if pseudo_facet in self.state and self.state[pseudo_facet]["state"] == "active":
                existing = self.state[pseudo_facet]
                existing["description"] = activity_summary
                existing["active_entities"] = entity_names
                changes.append({**existing, "_change": "continuing"})
            else:
                new_state = {
                    "id": _make_activity_id(content_type, segment_key),
                    "activity": content_type,
                    "state": "active",
                    "since": segment_key,
                    "description": activity_summary,
                    "level": "medium",
                    "active_entities": entity_names,
                }
                self.state[pseudo_facet] = new_state
                changes.append({**new_state, "_change": "new"})

        self.last_segment_key = segment_key
        self.last_segment_day = day
        self.history.extend(changes)
        return changes

    def get_current_state(self) -> list[dict]:
        """Return current active states in activity_state.json format."""
        result = []
        for state in self.state.values():
            if state["state"] == "active":
                result.append({
                    "id": state["id"],
                    "activity": state["activity"],
                    "state": state["state"],
                    "since": state["since"],
                    "description": state["description"],
                    "level": state["level"],
                    "active_entities": state["active_entities"],
                })
        return result

    def _should_reset(self, segment_key: str, day: str,
                      previous_segment_key: str | None) -> bool:
        """Check if we should end all active states due to gap or day change."""
        # Day change
        if self.last_segment_day and day != self.last_segment_day:
            return True

        # Time gap
        prev_key = previous_segment_key or self.last_segment_key
        if prev_key:
            prev_end = _segment_end_time(prev_key)
            curr_start = _parse_segment_time(segment_key)
            if prev_end is not None and curr_start is not None:
                gap = curr_start - prev_end
                if gap > self.GAP_THRESHOLD_SECONDS:
                    return True

        return False


# ---------------------------------------------------------------------------
# Comparison against baseline activity_state.json
# ---------------------------------------------------------------------------

def compare_state_machine(sm_output: list[dict], baseline_state: list[dict]) -> dict:
    """
    Compare state machine output against existing LLM-generated activity_state.json.

    Both are lists of state entries with: id, activity, state, since, description,
    level, active_entities.
    """
    sm_activities = {s.get("activity", "") for s in sm_output}
    bl_activities = {s.get("activity", "") for s in baseline_state}

    # Activity type match
    activity_match = sm_activities == bl_activities

    # Count comparison
    count_match = len(sm_output) == len(baseline_state)

    # State comparison (active/ended)
    sm_states = {s.get("id", ""): s.get("state", "") for s in sm_output}
    bl_states = {s.get("id", ""): s.get("state", "") for s in baseline_state}

    # Level comparison for matched activities
    sm_by_activity = {s.get("activity", ""): s for s in sm_output}
    bl_by_activity = {s.get("activity", ""): s for s in baseline_state}

    common_activities = sm_activities & bl_activities
    level_matches = 0
    entity_overlaps = []

    for act in common_activities:
        sm_entry = sm_by_activity.get(act, {})
        bl_entry = bl_by_activity.get(act, {})

        if sm_entry.get("level") == bl_entry.get("level"):
            level_matches += 1

        # Entity overlap
        sm_ents = set(sm_entry.get("active_entities", []))
        bl_ents = set(bl_entry.get("active_entities", []))
        if sm_ents or bl_ents:
            intersection = len(sm_ents & bl_ents)
            union = len(sm_ents | bl_ents)
            entity_overlaps.append(intersection / union if union > 0 else 1.0)

    n_common = len(common_activities)

    return {
        "activity_match": activity_match,
        "count_match": count_match,
        "sm_count": len(sm_output),
        "baseline_count": len(baseline_state),
        "common_activities": list(common_activities),
        "sm_only_activities": list(sm_activities - bl_activities),
        "baseline_only_activities": list(bl_activities - sm_activities),
        "level_match_count": level_matches,
        "level_match_total": n_common,
        "avg_entity_overlap": (
            round(sum(entity_overlaps) / len(entity_overlaps), 4)
            if entity_overlaps else None
        ),
    }
