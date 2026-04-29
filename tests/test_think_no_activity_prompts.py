# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think --no-activity-prompts behavior."""

import json
from pathlib import Path

import pytest

DAY = "20240115"
SEGMENT = "120000_300"
STREAM = "default"
FACET = "work"
ACTIVITY_ID = "coding_120000_300"


@pytest.fixture
def segment_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary journal with a segment directory."""
    journal = tmp_path / "journal"
    segment_path = journal / "chronicle" / DAY / STREAM / SEGMENT
    (segment_path / "talents").mkdir(parents=True)

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    return segment_path


def _segment_configs(*names: str) -> dict[str, dict]:
    configs = {
        "sense": {
            "priority": 10,
            "type": "generate",
            "output": "json",
            "schedule": "segment",
        },
    }
    return {name: dict(configs[name]) for name in names}


def _write_sense_output(segment_dir: Path, sense_json: dict) -> None:
    (segment_dir / "talents" / "sense.json").write_text(
        json.dumps(sense_json),
        encoding="utf-8",
    )


class EndedActivityStateMachine:
    def __init__(self, journal_root: Path) -> None:
        self.state: dict = {}
        self.last_segment_key: str | None = None
        self.last_segment_day: str | None = None
        self.journal_root = journal_root

    def update(self, sense_output: dict, segment: str, day: str) -> list[dict]:
        self.last_segment_key = segment
        self.last_segment_day = day
        self.state = {
            FACET: {
                "facet": FACET,
                "state": "active",
                "id": ACTIVITY_ID,
            }
        }
        return [{"state": "ended", "id": ACTIVITY_ID, "facet": FACET}]

    def get_completed_activities(self) -> list[dict]:
        return [
            {
                "id": ACTIVITY_ID,
                "activity": "coding",
                "segments": [SEGMENT],
                "level_avg": 0.5,
                "description": "coding",
                "active_entities": [],
                "created_at": 1713200000000,
            }
        ]


def _patch_segment_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    append_calls: list[tuple],
    activity_calls: list[dict],
) -> None:
    from think import thinking as think

    monkeypatch.setattr(
        think,
        "get_talent_configs",
        lambda schedule=None, **kwargs: _segment_configs("sense"),
    )
    monkeypatch.setattr(
        think,
        "cortex_request",
        lambda prompt, name, config=None: f"agent-{name}",
    )
    monkeypatch.setattr(
        think,
        "wait_for_uses",
        lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
    )
    monkeypatch.setattr(
        think,
        "append_activity_record",
        lambda *args: append_calls.append(args),
    )
    monkeypatch.setattr(
        think,
        "run_activity_prompts",
        lambda **kwargs: activity_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(think, "_callosum", None)


def test_flag_set_skips_prompts_but_writes_record(
    segment_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from think import thinking as think
    from think.thinking import ThinkingJSONLWriter

    append_calls: list[tuple] = []
    activity_calls: list[dict] = []
    jsonl_path = segment_dir.parent.parent / "health" / "test_no_prompts.jsonl"
    writer = ThinkingJSONLWriter(str(jsonl_path))

    _write_sense_output(
        segment_dir,
        {"density": "active", "recommend": {}, "facets": []},
    )
    _patch_segment_dependencies(monkeypatch, append_calls, activity_calls)
    monkeypatch.setattr(think, "_jsonl", writer)

    think.run_segment_sense(
        DAY,
        SEGMENT,
        refresh=False,
        verbose=False,
        stream=STREAM,
        state_machine=EndedActivityStateMachine(segment_dir.parents[3]),
        skip_activity_prompts=True,
    )
    writer.close()

    assert len(append_calls) >= 1
    assert activity_calls == []

    events = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event["event"] == "activity.prompts_skipped"
        and event["activity"] == ACTIVITY_ID
        and event["facet"] == FACET
        for event in events
    )


def test_flag_unset_runs_prompts_unchanged(
    segment_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from think import thinking as think

    append_calls: list[tuple] = []
    activity_calls: list[dict] = []

    _write_sense_output(
        segment_dir,
        {"density": "active", "recommend": {}, "facets": []},
    )
    _patch_segment_dependencies(monkeypatch, append_calls, activity_calls)

    think.run_segment_sense(
        DAY,
        SEGMENT,
        refresh=False,
        verbose=False,
        stream=STREAM,
        state_machine=EndedActivityStateMachine(segment_dir.parents[3]),
    )

    assert len(append_calls) >= 1
    assert activity_calls == [
        {
            "day": DAY,
            "activity_id": ACTIVITY_ID,
            "facet": FACET,
            "refresh": False,
            "verbose": False,
            "max_concurrency": 2,
        }
    ]


def test_flag_with_activity_mode_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from think import thinking as think

    journal = tmp_path / "journal"
    (journal / "chronicle" / DAY).mkdir(parents=True)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    monkeypatch.setattr(
        "sys.argv",
        [
            "sol think",
            "--activity",
            ACTIVITY_ID,
            "--facet",
            FACET,
            "--day",
            DAY,
            "--no-activity-prompts",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        think.main()

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "--no-activity-prompts" in stderr
    assert "--activity" in stderr
