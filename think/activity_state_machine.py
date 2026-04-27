# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Deterministic activity state machine replacing LLM-based activity tracking."""

import json
import logging
import time
from pathlib import Path

from think.activities import LEVEL_VALUES, make_activity_id
from think.utils import segment_parse

# 10 min; R&D default. Existing LLM hook uses 3600.
GAP_THRESHOLD_SECONDS = 600


class ActivityStateMachine:
    def __init__(self, journal_root: Path | None = None) -> None:
        self.state: dict[str, dict] = {}
        self.last_segment_key: str | None = None
        self.last_segment_day: str | None = None
        self._completed: list[dict] = []
        self.journal_root = Path(journal_root) if journal_root else None

        if self.journal_root is None:
            return

        state_path = self.journal_root / "awareness" / "activity_state.json"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            logging.debug("Failed to hydrate activity state: %s", exc)
            return

        if isinstance(data, list):
            active = {
                str(entry.get("facet") or "__"): entry
                for entry in data
                if isinstance(entry, dict)
            }
        elif isinstance(data, dict):
            raw_active = data.get("active")
            active = raw_active if isinstance(raw_active, dict) else {}
            self.last_segment_key = data.get("last_segment_key")
            self.last_segment_day = data.get("last_segment_day")
        else:
            logging.debug("Ignoring unexpected activity state shape")
            return

        for facet, raw_entry in active.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            if not all(
                key in entry for key in ("id", "activity", "since", "description")
            ):
                logging.debug(
                    "Ignoring incomplete activity state entry for facet %s", facet
                )
                continue
            entry.setdefault("facet", facet)
            entry.setdefault("segment", entry.get("since"))
            entry.setdefault("segments", [entry["since"]])
            self.state[str(facet)] = entry

    def _parse_segment_seconds(self, segment_key: str) -> int | None:
        start_time, _end_time = segment_parse(segment_key)
        if start_time is None:
            return None
        return start_time.hour * 3600 + start_time.minute * 60 + start_time.second

    def _parse_segment_end_seconds(self, segment_key: str) -> int | None:
        _start_time, end_time = segment_parse(segment_key)
        if end_time is None:
            return None
        return end_time.hour * 3600 + end_time.minute * 60 + end_time.second

    def _should_reset(
        self, segment_key: str, day: str, previous_segment_key: str | None
    ) -> bool:
        if self.last_segment_day is None:
            return False
        if day != self.last_segment_day:
            return True

        prev_key = previous_segment_key or self.last_segment_key
        if prev_key is None:
            return False

        prev_end = self._parse_segment_end_seconds(prev_key)
        curr_start = self._parse_segment_seconds(segment_key)
        if prev_end is None or curr_start is None:
            return False

        return (curr_start - prev_end) > GAP_THRESHOLD_SECONDS

    def _end_all(self, segment_key: str, change: str) -> list[dict]:
        changes = []
        for facet in sorted(self.state):
            prior = self.state[facet]
            entry = {
                "id": prior["id"],
                "activity": prior["activity"],
                "state": "ended",
                "since": prior["since"],
                "description": prior["description"],
                "_change": change,
                "facet": facet,
                "segment": segment_key,
            }
            changes.append(entry)
            self._completed.append(self._make_completed_record(prior))

        self.state = {}
        return changes

    def _make_completed_record(self, entry: dict) -> dict:
        return {
            "id": entry["id"],
            "activity": entry["activity"],
            "segments": entry.get("segments", [entry["since"]]),
            "level_avg": LEVEL_VALUES.get(entry.get("level", "medium"), 0.5),
            "description": entry["description"],
            "active_entities": entry.get("active_entities", []),
            "created_at": int(time.time() * 1000),
        }

    def update(
        self,
        sense_output: dict,
        segment_key: str,
        day: str,
        previous_segment_key: str | None = None,
    ) -> list[dict]:
        changes = []

        if self._should_reset(segment_key, day, previous_segment_key):
            changes.extend(self._end_all(segment_key, "ended_gap"))

        density = sense_output.get("density") or "active"
        content_type = sense_output.get("content_type") or "idle"
        activity_summary = sense_output.get("activity_summary") or ""
        raw_entities = sense_output.get("entities") or []
        entity_names = [
            entry["name"]
            for entry in raw_entities
            if isinstance(entry, dict) and entry.get("name")
        ]
        raw_facets = sense_output.get("facets") or []

        if density == "idle":
            changes.extend(self._end_all(segment_key, "ended_idle"))
            self.last_segment_key = segment_key
            self.last_segment_day = day
            return changes

        facet_map = {}
        for facet in raw_facets:
            if isinstance(facet, dict) and facet.get("facet"):
                facet_map[facet["facet"]] = facet
        current_facets = set(facet_map.keys()) if facet_map else {"__"}

        for facet in sorted(set(self.state.keys()) - current_facets):
            prior = self.state.pop(facet)
            entry = {
                "id": prior["id"],
                "activity": prior["activity"],
                "state": "ended",
                "since": prior["since"],
                "description": prior["description"],
                "_change": "ended_facet_gone",
                "facet": facet,
                "segment": segment_key,
            }
            changes.append(entry)
            self._completed.append(self._make_completed_record(prior))

        for facet in sorted(current_facets):
            facet_data = facet_map.get(facet, {})
            level = facet_data.get("level", "medium")
            if level not in ("high", "medium", "low"):
                level = "medium"

            if facet in self.state:
                prior = self.state[facet]
                if prior["activity"] != content_type:
                    ended = {
                        "id": prior["id"],
                        "activity": prior["activity"],
                        "state": "ended",
                        "since": prior["since"],
                        "description": prior["description"],
                        "_change": "ended_type_change",
                        "facet": facet,
                        "segment": segment_key,
                    }
                    changes.append(ended)
                    self._completed.append(self._make_completed_record(prior))

                    new_entry = {
                        "id": make_activity_id(content_type, segment_key),
                        "activity": content_type,
                        "state": "active",
                        "since": segment_key,
                        "description": activity_summary,
                        "level": level,
                        "active_entities": entity_names,
                        "_change": "new",
                        "facet": facet,
                        "segment": segment_key,
                        "segments": [segment_key],
                    }
                    self.state[facet] = new_entry
                    changes.append(dict(new_entry))
                else:
                    prior["description"] = activity_summary
                    prior["level"] = level
                    prior["active_entities"] = entity_names
                    prior["_change"] = "continuing"
                    prior["segment"] = segment_key
                    prior.setdefault("segments", [prior["since"]])
                    if segment_key not in prior["segments"]:
                        prior["segments"].append(segment_key)
                    changes.append(dict(prior))
            else:
                new_entry = {
                    "id": make_activity_id(content_type, segment_key),
                    "activity": content_type,
                    "state": "active",
                    "since": segment_key,
                    "description": activity_summary,
                    "level": level,
                    "active_entities": entity_names,
                    "_change": "new",
                    "facet": facet,
                    "segment": segment_key,
                    "segments": [segment_key],
                }
                self.state[facet] = new_entry
                changes.append(dict(new_entry))

        self.last_segment_key = segment_key
        self.last_segment_day = day
        return changes

    def get_completed_activities(self) -> list[dict]:
        return list(self._completed)
