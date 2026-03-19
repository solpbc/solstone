# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from apps.speakers.call import app
from apps.speakers.suggest import (
    _parse_meetings,
    format_suggestions,
    suggest_opportunities,
)


def create_meetings_md(env, day: str, content: str) -> Path:
    meetings_path = env.journal / day / "agents" / "meetings.md"
    meetings_path.parent.mkdir(parents=True, exist_ok=True)
    meetings_path.write_text(content, encoding="utf-8")
    return meetings_path


def _write_voiceprints(entity_dir: Path, embeddings: list[np.ndarray]) -> None:
    metadata = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": f"10000{i}_300",
                    "source": "mic_audio",
                    "sentence_id": i + 1,
                    "added_at": 1700000000000,
                }
            )
            for i in range(len(embeddings))
        ],
        dtype=str,
    )
    np.savez_compressed(
        entity_dir / "voiceprints.npz",
        embeddings=np.array(embeddings, dtype=np.float32),
        metadata=metadata,
    )


def test_suggest_empty_journal(speakers_env):
    speakers_env()

    assert suggest_opportunities() == []


def test_suggest_low_confidence_review(speakers_env):
    env = speakers_env()
    for idx in range(4):
        segment_key = f"1000{idx:02d}_300"
        env.create_segment("20240101", segment_key, ["mic_audio"])
        env.create_speaker_labels(
            "20240101",
            segment_key,
            [
                {
                    "sentence_id": 1,
                    "speaker": "alice_test",
                    "confidence": "medium",
                    "method": "voiceprint",
                },
                {
                    "sentence_id": 2,
                    "speaker": None,
                    "confidence": None,
                    "method": None,
                },
                {
                    "sentence_id": 3,
                    "speaker": "alice_test",
                    "confidence": "medium",
                    "method": "voiceprint",
                },
            ],
        )

    results = suggest_opportunities()

    suggestion = next(
        item for item in results if item["type"] == "low_confidence_review"
    )
    assert suggestion["day"] == "20240101"
    assert suggestion["medium_or_null_count"] == 12
    assert suggestion["total_labels"] == 12


def test_suggest_low_confidence_below_threshold(speakers_env):
    env = speakers_env()
    for idx in range(2):
        segment_key = f"1100{idx:02d}_300"
        env.create_segment("20240101", segment_key, ["mic_audio"])
        env.create_speaker_labels(
            "20240101",
            segment_key,
            [
                {
                    "sentence_id": 1,
                    "speaker": "alice_test",
                    "confidence": "medium",
                    "method": "voiceprint",
                },
                {
                    "sentence_id": 2,
                    "speaker": None,
                    "confidence": None,
                    "method": None,
                },
            ],
        )

    results = suggest_opportunities()

    assert all(item["type"] != "low_confidence_review" for item in results)


def test_suggest_name_variant(speakers_env):
    env = speakers_env()
    alice_dir = env.create_entity("Alice")
    alice_test_dir = env.create_entity("Alice Test")

    base = env.create_embedding([1.0, 0.0, 0.0])
    similar = env.create_embedding([1.0, 0.01, 0.0])
    _write_voiceprints(alice_dir, [base, similar])
    _write_voiceprints(alice_test_dir, [similar, base])

    results = suggest_opportunities()

    suggestion = next(item for item in results if item["type"] == "name_variant")
    assert suggestion["entity_a"]["id"] in {"alice", "alice_test"}
    assert suggestion["entity_b"]["id"] in {"alice", "alice_test"}
    assert suggestion["entity_a"]["id"] != suggestion["entity_b"]["id"]
    assert suggestion["similarity"] > 0.90


def test_suggest_import_linkable(speakers_env):
    env = speakers_env()
    env.create_entity("Romeo Montague")
    create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 10:00 Strategy Call with Romeo and Juliet\n",
    )

    results = suggest_opportunities()

    suggestion = next(item for item in results if item["type"] == "import_linkable")
    assert suggestion["entity_id"] == "romeo_montague"
    assert suggestion["name"] == "Romeo Montague"
    assert suggestion["meetings_mentioned"] == 1
    assert suggestion["meeting_days"] == ["20240101"]


def test_suggest_import_linkable_with_voiceprint_excluded(speakers_env):
    env = speakers_env()
    entity_dir = env.create_entity("Romeo Montague")
    _write_voiceprints(entity_dir, [env.create_embedding([1.0, 0.0, 0.0])])
    create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 10:00 Strategy Call with Romeo and Juliet\n",
    )

    results = suggest_opportunities()

    assert all(
        not (
            item["type"] == "import_linkable" and item["entity_id"] == "romeo_montague"
        )
        for item in results
    )


def test_suggest_limit(speakers_env):
    env = speakers_env()
    env.create_entity("Romeo Montague")
    alice_dir = env.create_entity("Alice")
    alice_test_dir = env.create_entity("Alice Test")
    _write_voiceprints(alice_dir, [env.create_embedding([1.0, 0.0, 0.0])])
    _write_voiceprints(alice_test_dir, [env.create_embedding([1.0, 0.01, 0.0])])
    create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 10:00 Strategy Call with Romeo and Juliet\n",
    )
    for idx in range(4):
        segment_key = f"1200{idx:02d}_300"
        env.create_segment("20240101", segment_key, ["mic_audio"])
        env.create_speaker_labels(
            "20240101",
            segment_key,
            [
                {
                    "sentence_id": sid,
                    "speaker": None,
                    "confidence": None,
                    "method": None,
                }
                for sid in range(1, 4)
            ],
        )

    results = suggest_opportunities(limit=1)

    assert len(results) == 1


def test_suggest_priority_order(speakers_env):
    env = speakers_env()
    env.create_entity("Romeo Montague")
    alice_dir = env.create_entity("Alice")
    alice_test_dir = env.create_entity("Alice Test")
    _write_voiceprints(alice_dir, [env.create_embedding([1.0, 0.0, 0.0])])
    _write_voiceprints(alice_test_dir, [env.create_embedding([1.0, 0.01, 0.0])])
    create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 10:00 Strategy Call with Romeo and Juliet\n",
    )
    for idx in range(4):
        segment_key = f"1300{idx:02d}_300"
        env.create_segment("20240101", segment_key, ["mic_audio"])
        env.create_speaker_labels(
            "20240101",
            segment_key,
            [
                {
                    "sentence_id": sid,
                    "speaker": None,
                    "confidence": None,
                    "method": None,
                }
                for sid in range(1, 4)
            ],
        )

    results = suggest_opportunities(limit=3)

    assert [item["type"] for item in results] == [
        "import_linkable",
        "name_variant",
        "low_confidence_review",
    ]


def test_parse_meetings_parenthesized(speakers_env):
    env = speakers_env()
    meetings_path = create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 08:30 Pre-Board Meeting Prep (Romeo, Juliet, Benvolio)\n",
    )

    meetings = _parse_meetings(str(meetings_path.parent.parent))

    assert meetings == [
        {
            "time": "08:30",
            "line": "- 08:30 Pre-Board Meeting Prep (Romeo, Juliet, Benvolio)",
            "participants": ["Romeo", "Juliet", "Benvolio"],
        }
    ]


def test_parse_meetings_with_keyword(speakers_env):
    env = speakers_env()
    meetings_path = create_meetings_md(
        env,
        "20240101",
        "# Meetings\n\n- 10:00 Strategy Call with Professor Lawrence, Romeo, and Juliet\n",
    )

    meetings = _parse_meetings(str(meetings_path.parent.parent))

    assert meetings == [
        {
            "time": "10:00",
            "line": "- 10:00 Strategy Call with Professor Lawrence, Romeo, and Juliet",
            "participants": ["Professor Lawrence", "Romeo", "Juliet"],
        }
    ]


def test_parse_meetings_missing_file(tmp_path):
    assert _parse_meetings(str(tmp_path)) == []


def test_format_suggestions_empty():
    assert format_suggestions([]) == "No speaker curation suggestions found."


def test_suggest_cli_json(speakers_env):
    speakers_env()
    runner = CliRunner()

    result = runner.invoke(app, ["suggest", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []
