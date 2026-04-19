# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the Sense output splitter."""

import json
from pathlib import Path


def _make_sense_output(**overrides):
    """Build a complete Sense output dict with sensible defaults."""
    base = {
        "density": "active",
        "content_type": "coding",
        "activity_summary": "Writing unit tests for the API module.",
        "entities": [
            {
                "type": "Project",
                "name": "SolAPI",
                "role": "mentioned",
                "source": "screen",
                "context": "main project",
            },
        ],
        "facets": [
            {"facet": "work", "activity": "coding", "level": "high"},
        ],
        "meeting_detected": False,
        "speakers": [],
        "recommend": {
            "screen_record": False,
            "speaker_attribution": False,
            "pulse_update": False,
        },
        "emotional_register": "neutral",
    }
    base.update(overrides)
    return base


class TestWriteSenseOutputs:
    def test_writes_all_standard_files(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        sense_json = _make_sense_output()

        write_sense_outputs(sense_json, seg_dir)

        agents_dir = seg_dir / "talents"
        assert (agents_dir / "activity.md").exists()
        assert (agents_dir / "facets.json").exists()
        assert (agents_dir / "density.json").exists()
        assert (agents_dir / "sense.json").exists()
        assert not (agents_dir / "speakers.json").exists()

        assert (agents_dir / "activity.md").read_text(encoding="utf-8") == (
            "Writing unit tests for the API module."
        )
        assert json.loads((agents_dir / "facets.json").read_text(encoding="utf-8")) == [
            {"facet": "work", "activity": "coding", "level": "high"}
        ]

        density = json.loads((agents_dir / "density.json").read_text(encoding="utf-8"))
        assert set(density.keys()) == {
            "classification",
            "transcript_lines",
            "screen_frames",
            "timestamp",
        }
        assert density["classification"] == "active"
        assert density["transcript_lines"] == 0
        assert density["screen_frames"] == 0

        assert json.loads((agents_dir / "sense.json").read_text(encoding="utf-8")) == (
            sense_json
        )

    def test_preserves_raw_payload_with_extra_keys_for_defensive_replay(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        # Advisory validation means unexpected keys can still reach the splitter.
        sense_json = _make_sense_output(foo="bar")

        write_sense_outputs(sense_json, seg_dir)

        stored = json.loads((seg_dir / "talents" / "sense.json").read_text("utf-8"))
        assert stored["foo"] == "bar"
        assert stored == sense_json

    def test_writes_sense_markdown_when_entities_exist(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        sense_json = _make_sense_output(
            entities=[
                {
                    "type": "Project",
                    "name": "SolAPI",
                    "role": "mentioned",
                    "source": "screen",
                    "context": "main project",
                },
                {
                    "type": "Person",
                    "name": "John Borthwick",
                    "role": "attendee",
                    "source": "voice",
                    "context": "active meeting participant",
                },
            ]
        )

        write_sense_outputs(sense_json, seg_dir)

        sense_md = (seg_dir / "talents" / "sense.md").read_text(encoding="utf-8")
        assert sense_md == (
            "# Sense Entities\n\n"
            "- Project — SolAPI (role=mentioned, source=screen) — main project\n"
            "- Person — John Borthwick (role=attendee, source=voice) "
            "— active meeting participant"
        )

    def test_skips_sense_markdown_when_entities_empty(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"

        write_sense_outputs(_make_sense_output(entities=[]), seg_dir)

        assert not (seg_dir / "talents" / "sense.md").exists()


class TestMeetingDetection:
    def test_writes_speakers_when_meeting_detected(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        sense_json = _make_sense_output(
            meeting_detected=True,
            speakers=["Alice", "Bob"],
        )

        write_sense_outputs(sense_json, seg_dir)

        speakers_path = seg_dir / "talents" / "speakers.json"
        assert speakers_path.exists()
        assert json.loads(speakers_path.read_text(encoding="utf-8")) == ["Alice", "Bob"]

    def test_no_speakers_when_not_meeting(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"

        write_sense_outputs(_make_sense_output(meeting_detected=False), seg_dir)

        assert not (seg_dir / "talents" / "speakers.json").exists()

    def test_meeting_with_no_speakers_writes_empty_array(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        sense_json = _make_sense_output(meeting_detected=True, speakers=None)

        write_sense_outputs(sense_json, seg_dir)

        speakers_path = seg_dir / "talents" / "speakers.json"
        assert speakers_path.exists()
        assert json.loads(speakers_path.read_text(encoding="utf-8")) == []


class TestEdgeCases:
    def test_missing_optional_fields(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"

        # Advisory schema validation means the splitter must still tolerate degraded input.
        write_sense_outputs({}, seg_dir)

        agents_dir = seg_dir / "talents"
        assert (agents_dir / "activity.md").exists()
        assert (agents_dir / "facets.json").exists()
        assert (agents_dir / "density.json").exists()
        assert (agents_dir / "sense.json").exists()
        assert (agents_dir / "activity.md").read_text(encoding="utf-8") == ""
        assert (
            json.loads((agents_dir / "facets.json").read_text(encoding="utf-8")) == []
        )
        density = json.loads((agents_dir / "density.json").read_text(encoding="utf-8"))
        assert density["classification"] == "active"

    def test_null_fields(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        sense_json = {
            "density": None,
            "content_type": None,
            "activity_summary": None,
            "entities": None,
            "facets": None,
            "meeting_detected": None,
            "speakers": None,
        }

        # Advisory schema validation means the splitter must still tolerate degraded input.
        write_sense_outputs(sense_json, seg_dir)

        agents_dir = seg_dir / "talents"
        assert (agents_dir / "activity.md").read_text(encoding="utf-8") == ""
        assert (
            json.loads((agents_dir / "facets.json").read_text(encoding="utf-8")) == []
        )
        density = json.loads((agents_dir / "density.json").read_text(encoding="utf-8"))
        assert density["classification"] == "active"
        assert not (agents_dir / "speakers.json").exists()

    def test_empty_activity_summary(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"

        write_sense_outputs(_make_sense_output(activity_summary=""), seg_dir)

        assert (seg_dir / "talents" / "activity.md").read_text(encoding="utf-8") == ""


class TestMultipleFacets:
    def test_multiple_facets_as_flat_array(self, tmp_path):
        from think.sense_splitter import write_sense_outputs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"
        facets = [
            {"facet": "work", "activity": "coding", "level": "high"},
            {"facet": "personal", "activity": "reading", "level": "low"},
        ]

        write_sense_outputs(_make_sense_output(facets=facets), seg_dir)

        assert json.loads((seg_dir / "talents" / "facets.json").read_text("utf-8")) == (
            facets
        )


class TestWriteIdleStubs:
    def test_writes_density_only(self, tmp_path):
        from think.sense_splitter import write_idle_stubs

        seg_dir = Path(tmp_path) / "20260304" / "default" / "090000_300"

        write_idle_stubs(seg_dir)

        agents_dir = seg_dir / "talents"
        assert (agents_dir / "density.json").exists()
        density = json.loads((agents_dir / "density.json").read_text(encoding="utf-8"))
        assert density["classification"] == "idle"
        assert density["transcript_lines"] == 0
        assert density["screen_frames"] == 0
        assert not (agents_dir / "activity.md").exists()
        assert not (agents_dir / "facets.json").exists()
        assert not (agents_dir / "speakers.json").exists()
        assert not (agents_dir / "sense.json").exists()
