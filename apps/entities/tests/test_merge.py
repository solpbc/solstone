# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for ``sol call entities merge``."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from apps.entities.call import app as entities_app
from think.entities.journal import load_journal_entity

runner = CliRunner()
STREAM = "test"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _entity_path(env, entity_id: str):
    return env.journal / "entities" / entity_id / "entity.json"


def _update_entity(env, entity_id: str, **fields) -> None:
    path = _entity_path(env, entity_id)
    payload = _read_json(path)
    payload.update(fields)
    _write_json(path, payload)


def _labels_path(env, day: str, segment_key: str):
    return env.journal / day / STREAM / segment_key / "talents" / "speaker_labels.json"


def _corrections_path(env, day: str, segment_key: str):
    return (
        env.journal
        / day
        / STREAM
        / segment_key
        / "talents"
        / "speaker_corrections.json"
    )


def _audit_log_path(env):
    return env.journal / "logs" / "entity-merges.jsonl"


def _voiceprint_count(env, entity_id: str) -> int:
    path = env.journal / "entities" / entity_id / "voiceprints.npz"
    with np.load(path, allow_pickle=False) as data:
        return len(data["embeddings"])


def test_merge_dry_run_plans_without_writing(speakers_env):
    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    env.create_entity(
        "Dry Alias",
        voiceprints=[
            ("20240101", "143022_300", "mic_audio", 1),
            ("20240101", "143022_300", "mic_audio", 2),
        ],
    )
    env.create_entity(
        "Dry Canon",
        voiceprints=[("20240101", "143022_300", "mic_audio", 3)],
    )
    env.create_facet_relationship(
        "work",
        "dry_alias",
        observations=["Likes coffee"],
    )
    env.create_facet_relationship(
        "work",
        "dry_canon",
        observations=["Senior role"],
    )
    env.create_facet_relationship("personal", "dry_alias", description="Runner")
    env.create_speaker_labels(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "speaker": "dry_alias",
                "confidence": "high",
                "method": "acoustic",
            }
        ],
    )
    env.create_speaker_corrections(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "original_speaker": "dry_alias",
                "corrected_speaker": "dry_alias",
                "timestamp": 1700000000000,
            }
        ],
    )
    cache_path = env.journal / "awareness" / "discovery_clusters.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"clusters": []}', encoding="utf-8")

    source_before = _entity_path(env, "dry_alias").read_text(encoding="utf-8")
    target_before = _entity_path(env, "dry_canon").read_text(encoding="utf-8")
    labels_before = _labels_path(env, "20240101", "143022_300").read_text(
        encoding="utf-8"
    )
    corrections_before = _corrections_path(env, "20240101", "143022_300").read_text(
        encoding="utf-8"
    )

    result = runner.invoke(entities_app, ["merge", "dry_alias", "dry_canon"])

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    data = json.loads(result.output)
    assert data["merged"] is False
    assert data["identity"]["akas_added"] == []
    assert data["voiceprints"]["added"] == 0
    assert data["facets"]["moved"] == []
    assert data["segments"]["files_scanned"] == 0
    assert "Dry Alias" in data["would_identity"]["akas_added"]
    assert data["would_voiceprints"]["added"] == 2
    assert data["would_facets"]["merged"] == ["work"]
    assert data["would_facets"]["moved"] == ["personal"]
    assert data["would_segments"]["labels_rewritten"] == 1
    assert data["would_segments"]["corrections_rewritten"] == 1
    assert data["audit_log_path"] is None
    assert data["caches_cleared"] == []

    assert _entity_path(env, "dry_alias").read_text(encoding="utf-8") == source_before
    assert _entity_path(env, "dry_canon").read_text(encoding="utf-8") == target_before
    assert (
        _labels_path(env, "20240101", "143022_300").read_text(encoding="utf-8")
        == labels_before
    )
    assert (
        _corrections_path(env, "20240101", "143022_300").read_text(encoding="utf-8")
        == corrections_before
    )
    assert cache_path.exists()
    assert load_journal_entity("dry_alias") is not None
    assert not _audit_log_path(env).exists()


def test_merge_commit_deep_merges_and_logs(speakers_env):
    env = speakers_env()
    env.create_segment("20240101", "143022_300", ["mic_audio"])
    env.create_entity(
        "Alice Alias",
        voiceprints=[
            ("20240101", "143022_300", "mic_audio", 1),
            ("20240101", "143022_300", "mic_audio", 2),
        ],
    )
    env.create_entity(
        "Alice Canonical",
        voiceprints=[("20240101", "143022_300", "mic_audio", 3)],
    )
    env.create_facet_relationship(
        "work",
        "alice_alias",
        description="Works at Acme",
        attached_at=1600000000000,
        observations=["Likes coffee", "Morning person"],
    )
    env.create_facet_relationship(
        "work",
        "alice_canonical",
        description="Senior engineer",
        attached_at=1700000000000,
        observations=["Staff role"],
    )
    env.create_facet_relationship("personal", "alice_alias", description="Hiker")
    env.create_speaker_labels(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "speaker": "alice_alias",
                "confidence": "high",
                "method": "acoustic",
            },
            {
                "sentence_id": 2,
                "speaker": "alice_canonical",
                "confidence": "high",
                "method": "acoustic",
            },
            {
                "sentence_id": 3,
                "speaker": "alice_alias",
                "confidence": "medium",
                "method": "context",
            },
        ],
    )
    env.create_speaker_corrections(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "original_speaker": "alice_alias",
                "corrected_speaker": "alice_alias",
                "timestamp": 1700000000000,
            },
        ],
    )
    cache_path = env.journal / "awareness" / "discovery_clusters.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"clusters": []}', encoding="utf-8")

    result = runner.invoke(
        entities_app,
        ["merge", "alice_alias", "alice_canonical", "--commit"],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    data = json.loads(result.output)
    assert data["merged"] is True
    assert "Alice Alias" in data["identity"]["akas_added"]
    assert data["voiceprints"]["added"] == 2
    assert data["voiceprints"]["target_total"] == 3
    assert "work" in data["facets"]["merged"]
    assert "personal" in data["facets"]["moved"]
    assert data["facets"]["observations_appended"] == 2
    assert data["segments"]["labels_rewritten"] == 1
    assert data["segments"]["corrections_rewritten"] == 1
    assert data["segments"]["errors"] == []
    assert data["audit_log_path"] == str(_audit_log_path(env))
    assert set(data["caches_cleared"]) >= {
        "journal_entity_cache",
        "relationship_caches",
        "observation_cache",
        "entity_loading_cache",
        "discovery_clusters",
    }

    assert load_journal_entity("alice_alias") is None
    canonical = load_journal_entity("alice_canonical")
    assert canonical is not None
    assert "Alice Alias" in canonical["aka"]

    assert _voiceprint_count(env, "alice_canonical") == 3

    labels = _read_json(_labels_path(env, "20240101", "143022_300"))
    speakers = [label["speaker"] for label in labels["labels"]]
    assert "alice_alias" not in speakers
    assert speakers.count("alice_canonical") == 3

    corrections = _read_json(_corrections_path(env, "20240101", "143022_300"))
    for correction in corrections["corrections"]:
        assert correction.get("original_speaker") != "alice_alias"
        assert correction.get("corrected_speaker") != "alice_alias"

    observations_path = (
        env.journal
        / "facets"
        / "work"
        / "entities"
        / "alice_canonical"
        / "observations.jsonl"
    )
    contents = [
        json.loads(line)["content"]
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(contents) == {"Staff role", "Likes coffee", "Morning person"}
    assert not cache_path.exists()

    audit_entries = [
        json.loads(line)
        for line in _audit_log_path(env).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audit_entries) == 1
    assert isinstance(audit_entries[0]["ts"], int)
    assert audit_entries[0]["caller"] == "entities.merge"
    assert audit_entries[0]["source_id"] == "alice_alias"
    assert audit_entries[0]["source_display_name"] == "Alice Alias"
    assert audit_entries[0]["target_id"] == "alice_canonical"
    assert audit_entries[0]["target_display_name"] == "Alice Canonical"
    assert audit_entries[0]["principal_transferred"] is False
    assert set(audit_entries[0]["counts"]) == {
        "identity",
        "voiceprints",
        "facets",
        "segments",
    }


def test_merge_default_keeps_source_as_aka(speakers_env):
    env = speakers_env()
    env.create_entity("Keep Alias")
    env.create_entity("Keep Canon")

    result = runner.invoke(
        entities_app,
        ["merge", "keep_alias", "keep_canon", "--commit"],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    canonical = load_journal_entity("keep_canon")
    assert canonical is not None
    assert "Keep Alias" in canonical["aka"]


def test_merge_no_keep_source_as_aka_keeps_only_existing_aliases(speakers_env):
    env = speakers_env()
    env.create_entity("Skip Alias")
    env.create_entity("Skip Canon")
    _update_entity(env, "skip_alias", aka=["SA", "S.A."])

    result = runner.invoke(
        entities_app,
        [
            "merge",
            "skip_alias",
            "skip_canon",
            "--commit",
            "--no-keep-source-as-aka",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    canonical = load_journal_entity("skip_canon")
    assert canonical is not None
    assert "Skip Alias" not in canonical.get("aka", [])
    assert {"SA", "S.A."} <= set(canonical["aka"])


def test_merge_transfers_principal_from_source_to_target(speakers_env):
    env = speakers_env()
    env.create_entity("Principal Source", is_principal=True)
    env.create_entity("Principal Target")

    result = runner.invoke(
        entities_app,
        ["merge", "principal_source", "principal_target", "--commit"],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    data = json.loads(result.output)
    assert data["identity"]["principal_transferred"] is True
    assert load_journal_entity("principal_source") is None
    target = load_journal_entity("principal_target")
    assert target is not None
    assert target["is_principal"] is True


def test_merge_errors_when_both_entities_are_principal(speakers_env):
    env = speakers_env()
    env.create_entity("First Principal", is_principal=True)
    env.create_entity("Second Principal", is_principal=True)

    result = runner.invoke(
        entities_app,
        ["merge", "first_principal", "second_principal", "--commit"],
    )

    assert result.exit_code == 1, f"{result.output}\n{result.exception!r}"
    data = json.loads(result.output)
    assert data["error"] == "Cannot merge two principal entities."
    assert load_journal_entity("first_principal") is not None
    assert load_journal_entity("second_principal") is not None


def test_merge_errors_on_aka_cross_reference(speakers_env):
    env = speakers_env()
    env.create_entity("Cross Source")
    env.create_entity("Cross Target")
    env.create_entity("Cross Watcher")
    _update_entity(env, "cross_watcher", aka=["cross_source", "Watcher Alias"])

    result = runner.invoke(
        entities_app,
        ["merge", "cross_source", "cross_target", "--commit"],
    )

    assert result.exit_code == 1, f"{result.output}\n{result.exception!r}"
    data = json.loads(result.output)
    assert (
        data["error"]
        == "Cannot merge 'cross_source': referenced in aka lists of entity ids: cross_watcher"
    )
    assert load_journal_entity("cross_source") is not None
    assert load_journal_entity("cross_target") is not None


def test_merge_validation_errors(speakers_env):
    env = speakers_env()
    env.create_entity("Blocked Source")
    env.create_entity("Validation Target")
    _update_entity(env, "blocked_source", blocked=True)

    cases = [
        (
            ["merge", "validation_target", "validation_target", "--commit"],
            "Source and target must be different entities.",
        ),
        (
            ["merge", "missing_source", "validation_target", "--commit"],
            "Source entity not found: missing_source",
        ),
        (
            ["merge", "validation_target", "missing_target", "--commit"],
            "Target entity not found: missing_target",
        ),
        (
            ["merge", "blocked_source", "validation_target", "--commit"],
            "Cannot merge blocked entity: blocked_source",
        ),
    ]

    for argv, expected_error in cases:
        result = runner.invoke(entities_app, argv)
        assert result.exit_code == 1, f"{result.output}\n{result.exception!r}"
        data = json.loads(result.output)
        assert data["error"] == expected_error
