# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entity deep merge."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from apps.speakers.bootstrap import merge_names
from apps.speakers.call import app as speakers_app
from think.entities.journal import load_journal_entity, scan_journal_entities

_runner = CliRunner()

# Match conftest default stream
STREAM = "test"


# ---------------------------------------------------------------------------
# Core merge behavior
# ---------------------------------------------------------------------------


def test_deep_merge_full(speakers_env):
    """Full deep merge: identity, voiceprints, facets, speaker refs, cleanup."""
    env = speakers_env()

    # Create entities with voiceprints
    env.create_entity(
        "Alice Alias",
        voiceprints=[
            ("20240101", "143022_300", "mic_audio", 1),
            ("20240101", "143022_300", "mic_audio", 2),
        ],
    )
    env.create_entity(
        "Alice Canonical",
        voiceprints=[
            ("20240101", "143022_300", "mic_audio", 3),
        ],
    )

    # Facet relationships
    env.create_facet_relationship(
        "work",
        "alice_alias",
        description="Works at Acme",
        attached_at=1600000000000,
    )
    env.create_facet_relationship(
        "work",
        "alice_canonical",
        description="Senior engineer",
        attached_at=1700000000000,
    )
    env.create_facet_relationship("personal", "alice_alias", description="Hiker")

    # Speaker labels referencing alias
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

    # Speaker corrections referencing alias
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

    result = merge_names("Alice Alias", "Alice Canonical")

    assert result["merged"] is True
    assert result["alias"] == "Alice Alias"
    assert result["alias_id"] == "alice_alias"
    assert result["canonical_id"] == "alice_canonical"
    assert "Alice Alias" in result["akas_added"]
    assert result["voiceprints_merged"] == 2
    assert result["labels_rewritten"] == 1
    assert result["corrections_rewritten"] == 1
    assert "work" in result["facets_merged"]
    assert "personal" in result["facets_moved"]
    assert result["errors"] == []

    # Alias entity is gone
    assert load_journal_entity("alice_alias") is None
    assert "alice_alias" not in scan_journal_entities()

    # Canonical has merged data
    canonical = load_journal_entity("alice_canonical")
    assert "Alice Alias" in canonical["aka"]

    # Speaker labels rewritten
    labels_path = (
        env.journal
        / "20240101"
        / STREAM
        / "143022_300"
        / "agents"
        / "speaker_labels.json"
    )
    with open(labels_path) as f:
        labels = json.load(f)
    speakers = [l["speaker"] for l in labels["labels"]]
    assert "alice_alias" not in speakers
    assert speakers.count("alice_canonical") == 3

    # Corrections rewritten
    corr_path = (
        env.journal
        / "20240101"
        / STREAM
        / "143022_300"
        / "agents"
        / "speaker_corrections.json"
    )
    with open(corr_path) as f:
        corr = json.load(f)
    for c in corr["corrections"]:
        assert c.get("original_speaker") != "alice_alias"
        assert c.get("corrected_speaker") != "alice_alias"


def test_alias_entity_deleted(speakers_env):
    """After merge, scan_journal_entities does not return alias_id."""
    env = speakers_env()
    env.create_entity("Gone")
    env.create_entity("Stays")

    merge_names("Gone", "Stays")

    assert "gone" not in scan_journal_entities()
    assert load_journal_entity("gone") is None


def test_akas_transitive(speakers_env):
    """Alias's existing akas are merged into canonical."""
    env = speakers_env()
    alias_dir = env.create_entity("Alias Trans")
    alias_path = alias_dir / "entity.json"
    with open(alias_path) as f:
        data = json.load(f)
    data["aka"] = ["AT", "A.T."]
    with open(alias_path, "w") as f:
        json.dump(data, f)

    env.create_entity("Canonical Trans")

    result = merge_names("Alias Trans", "Canonical Trans")

    assert result["merged"] is True
    assert "Alias Trans" in result["akas_added"]
    assert "AT" in result["akas_added"]
    assert "A.T." in result["akas_added"]

    canonical = load_journal_entity("canonical_trans")
    assert {"Alias Trans", "AT", "A.T."} <= set(canonical["aka"])


def test_emails_merged(speakers_env):
    """Emails from alias are merged into canonical (union, lowercased)."""
    env = speakers_env()
    alias_dir = env.create_entity("Email Alias")
    canonical_dir = env.create_entity("Email Canonical")

    for entity_dir, emails in [
        (alias_dir, ["alias@example.com", "SHARED@Example.COM"]),
        (canonical_dir, ["canonical@example.com", "shared@example.com"]),
    ]:
        path = entity_dir / "entity.json"
        with open(path) as f:
            data = json.load(f)
        data["emails"] = emails
        with open(path, "w") as f:
            json.dump(data, f)

    merge_names("Email Alias", "Email Canonical")

    canonical = load_journal_entity("email_canonical")
    assert set(canonical["emails"]) == {
        "alias@example.com",
        "canonical@example.com",
        "shared@example.com",
    }


# ---------------------------------------------------------------------------
# Facet relationship merging
# ---------------------------------------------------------------------------


def test_facet_move(speakers_env):
    """Alias facet relationship moves to canonical when canonical has none."""
    env = speakers_env()
    env.create_entity("Short")
    env.create_entity("Full Name")
    env.create_facet_relationship(
        "work", "short", description="Consultant", attached_at=1600000000000
    )

    result = merge_names("Short", "Full Name")

    assert "work" in result["facets_moved"]
    rel_path = (
        env.journal / "facets" / "work" / "entities" / "full_name" / "entity.json"
    )
    assert rel_path.exists()
    with open(rel_path) as f:
        rel = json.load(f)
    assert rel["entity_id"] == "full_name"
    assert rel["description"] == "Consultant"

    # Alias dir gone
    assert not (env.journal / "facets" / "work" / "entities" / "short").exists()


def test_facet_merge_timestamps(speakers_env):
    """Facet merge: earliest attached_at, latest updated_at/last_seen."""
    env = speakers_env()
    env.create_entity("Alias P")
    env.create_entity("Canonical P")
    env.create_facet_relationship(
        "work",
        "alias_p",
        attached_at=1500000000000,
        updated_at=1800000000000,
        last_seen="20260301",
    )
    env.create_facet_relationship(
        "work",
        "canonical_p",
        attached_at=1700000000000,
        updated_at=1600000000000,
        last_seen="20260201",
    )

    result = merge_names("Alias P", "Canonical P")

    assert "work" in result["facets_merged"]
    rel_path = (
        env.journal / "facets" / "work" / "entities" / "canonical_p" / "entity.json"
    )
    with open(rel_path) as f:
        rel = json.load(f)
    assert rel["attached_at"] == 1500000000000
    assert rel["updated_at"] == 1800000000000
    assert rel["last_seen"] == "20260301"


def test_facet_merge_description_priority(speakers_env):
    """Canonical description takes priority; alias fills empty."""
    env = speakers_env()
    env.create_entity("Alias D")
    env.create_entity("Canonical D")

    # Both have descriptions: canonical's wins
    env.create_facet_relationship("work", "alias_d", description="From alias")
    env.create_facet_relationship(
        "work", "canonical_d", description="From canonical"
    )

    merge_names("Alias D", "Canonical D")

    rel_path = (
        env.journal / "facets" / "work" / "entities" / "canonical_d" / "entity.json"
    )
    with open(rel_path) as f:
        rel = json.load(f)
    assert rel["description"] == "From canonical"


def test_facet_merge_description_fallback(speakers_env):
    """Alias description used when canonical's is empty."""
    env = speakers_env()
    env.create_entity("Alias E")
    env.create_entity("Canonical E")

    env.create_facet_relationship("work", "alias_e", description="Has desc")
    env.create_facet_relationship("work", "canonical_e", description="")

    merge_names("Alias E", "Canonical E")

    rel_path = (
        env.journal / "facets" / "work" / "entities" / "canonical_e" / "entity.json"
    )
    with open(rel_path) as f:
        rel = json.load(f)
    assert rel["description"] == "Has desc"


def test_facet_observations_merged(speakers_env):
    """Observations from alias are appended to canonical's."""
    env = speakers_env()
    env.create_entity("Alias Obs")
    env.create_entity("Canonical Obs")
    env.create_facet_relationship(
        "work", "alias_obs", observations=["Likes coffee", "Morning person"]
    )
    env.create_facet_relationship(
        "work", "canonical_obs", observations=["Senior role"]
    )

    merge_names("Alias Obs", "Canonical Obs")

    obs_path = (
        env.journal
        / "facets"
        / "work"
        / "entities"
        / "canonical_obs"
        / "observations.jsonl"
    )
    obs = [json.loads(line) for line in obs_path.read_text().strip().split("\n")]
    contents = [o["content"] for o in obs]
    assert "Senior role" in contents
    assert "Likes coffee" in contents
    assert "Morning person" in contents


def test_facet_no_alias_relationship(speakers_env):
    """No change when only canonical has a facet relationship."""
    env = speakers_env()
    env.create_entity("Alias None")
    env.create_entity("Canonical None")
    env.create_facet_relationship(
        "work", "canonical_none", description="Only me"
    )

    result = merge_names("Alias None", "Canonical None")

    assert result["facets_merged"] == []
    assert result["facets_moved"] == []
    rel_path = (
        env.journal
        / "facets"
        / "work"
        / "entities"
        / "canonical_none"
        / "entity.json"
    )
    with open(rel_path) as f:
        rel = json.load(f)
    assert rel["description"] == "Only me"


def test_no_alias_facet_dirs_remain(speakers_env):
    """After merge, no facet has a relationship directory for alias_id."""
    env = speakers_env()
    env.create_entity("Multi Alias")
    env.create_entity("Multi Canon")
    env.create_facet_relationship("work", "multi_alias")
    env.create_facet_relationship("personal", "multi_alias")
    env.create_facet_relationship("work", "multi_canon")

    merge_names("Multi Alias", "Multi Canon")

    facets_dir = env.journal / "facets"
    for facet_entry in facets_dir.iterdir():
        if facet_entry.is_dir():
            alias_dir = facet_entry / "entities" / "multi_alias"
            assert not alias_dir.exists(), f"alias dir exists in {facet_entry.name}"


# ---------------------------------------------------------------------------
# Speaker reference rewriting
# ---------------------------------------------------------------------------


def test_speaker_labels_rewritten(speakers_env):
    """speaker_labels.json files referencing alias_id are rewritten."""
    env = speakers_env()
    env.create_entity("Label Alias")
    env.create_entity("Label Canon")
    env.create_speaker_labels(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "speaker": "label_alias",
                "confidence": "high",
                "method": "acoustic",
            },
            {
                "sentence_id": 2,
                "speaker": "other_entity",
                "confidence": "high",
                "method": "acoustic",
            },
        ],
    )

    result = merge_names("Label Alias", "Label Canon")

    assert result["labels_rewritten"] == 1
    labels_path = (
        env.journal
        / "20240101"
        / STREAM
        / "143022_300"
        / "agents"
        / "speaker_labels.json"
    )
    with open(labels_path) as f:
        data = json.load(f)
    assert data["labels"][0]["speaker"] == "label_canon"
    assert data["labels"][1]["speaker"] == "other_entity"


def test_speaker_corrections_rewritten(speakers_env):
    """speaker_corrections.json files referencing alias_id are rewritten."""
    env = speakers_env()
    env.create_entity("Corr Alias")
    env.create_entity("Corr Canon")
    env.create_speaker_corrections(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "original_speaker": "corr_alias",
                "corrected_speaker": "someone_else",
                "timestamp": 1700000000000,
            },
            {
                "sentence_id": 2,
                "original_speaker": "someone_else",
                "corrected_speaker": "corr_alias",
                "timestamp": 1700000000000,
            },
        ],
    )

    result = merge_names("Corr Alias", "Corr Canon")

    assert result["corrections_rewritten"] == 1
    corr_path = (
        env.journal
        / "20240101"
        / STREAM
        / "143022_300"
        / "agents"
        / "speaker_corrections.json"
    )
    with open(corr_path) as f:
        data = json.load(f)
    assert data["corrections"][0]["original_speaker"] == "corr_canon"
    assert data["corrections"][0]["corrected_speaker"] == "someone_else"
    assert data["corrections"][1]["original_speaker"] == "someone_else"
    assert data["corrections"][1]["corrected_speaker"] == "corr_canon"


def test_fast_path_skips_unrelated_files(speakers_env):
    """Files without alias_id bytes are not modified."""
    env = speakers_env()
    env.create_entity("Fast Alias")
    env.create_entity("Fast Canon")
    env.create_speaker_labels(
        "20240101",
        "143022_300",
        [
            {
                "sentence_id": 1,
                "speaker": "other_person",
                "confidence": "high",
                "method": "acoustic",
            },
        ],
    )

    labels_path = (
        env.journal
        / "20240101"
        / STREAM
        / "143022_300"
        / "agents"
        / "speaker_labels.json"
    )
    mtime_before = labels_path.stat().st_mtime_ns

    result = merge_names("Fast Alias", "Fast Canon")

    assert result["labels_rewritten"] == 0
    assert labels_path.stat().st_mtime_ns == mtime_before


def test_corrupted_labels_logged_not_aborted(speakers_env):
    """Corrupted speaker_labels.json is logged as error but merge continues."""
    env = speakers_env()
    env.create_entity("Corrupt Alias")
    env.create_entity("Corrupt Canon")

    # Write corrupted file containing the alias_id string
    agents_dir = env.journal / "20240101" / STREAM / "143022_300" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "speaker_labels.json").write_text(
        "corrupt_alias {not valid json"
    )

    result = merge_names("Corrupt Alias", "Corrupt Canon")

    assert result["merged"] is True
    assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# Interrupt safety
# ---------------------------------------------------------------------------


def test_idempotent_rerun(speakers_env):
    """Running merge twice: second run is a no-op (alias gone → error)."""
    env = speakers_env()
    env.create_entity("Idem Alias")
    env.create_entity("Idem Canon")

    result1 = merge_names("Idem Alias", "Idem Canon")
    assert result1["merged"] is True

    # Second run: alias name now resolves to canonical via aka → same entity
    result2 = merge_names("Idem Alias", "Idem Canon")
    assert "error" in result2


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------


def test_error_self_merge(speakers_env):
    """Merging entity into itself returns error."""
    env = speakers_env()
    env.create_entity("Same Person")

    result = merge_names("Same Person", "Same Person")
    assert "error" in result


def test_error_blocked_entity(speakers_env):
    """Merging a blocked entity returns error."""
    env = speakers_env()
    entity_dir = env.create_entity("Blocked One")
    path = entity_dir / "entity.json"
    with open(path) as f:
        data = json.load(f)
    data["blocked"] = True
    with open(path, "w") as f:
        json.dump(data, f)

    env.create_entity("Target One")

    result = merge_names("Blocked One", "Target One")
    assert "error" in result
    assert "blocked" in result["error"].lower()


def test_error_principal_as_alias(speakers_env):
    """Merging the principal entity (as alias) returns error."""
    env = speakers_env()
    env.create_entity("Self Person", is_principal=True)
    env.create_entity("Other Person")

    result = merge_names("Self Person", "Other Person")
    assert "error" in result
    assert "principal" in result["error"].lower()


def test_error_principal_as_canonical(speakers_env):
    """Merging into the principal entity (as canonical) returns error."""
    env = speakers_env()
    env.create_entity("Other Person")
    env.create_entity("Self Person", is_principal=True)

    result = merge_names("Other Person", "Self Person")
    assert "error" in result
    assert "principal" in result["error"].lower()


def test_error_nonexistent_entity(speakers_env):
    """Merging non-existent entity returns error."""
    env = speakers_env()
    env.create_entity("Real Person")

    result = merge_names("Ghost", "Real Person")
    assert "error" in result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_discovery_cache_busted(speakers_env):
    """Discovery cache is deleted after merge."""
    env = speakers_env()
    env.create_entity("Cache Alias")
    env.create_entity("Cache Canon")

    awareness_dir = env.journal / "awareness"
    awareness_dir.mkdir(parents=True, exist_ok=True)
    cache_path = awareness_dir / "discovery_clusters.json"
    cache_path.write_text('{"clusters": []}')

    merge_names("Cache Alias", "Cache Canon")

    assert not cache_path.exists()


# ---------------------------------------------------------------------------
# resolve_name_variants integration
# ---------------------------------------------------------------------------


def test_resolve_name_variants_deep_merge(speakers_env):
    """resolve_name_variants(dry_run=False) calls deep merge (alias deleted)."""
    env = speakers_env()

    # Create two entities with identical voiceprints
    embedding = np.random.default_rng(42).standard_normal(256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    embeddings = np.tile(embedding.reshape(1, -1), (5, 1))

    alias_dir = env.create_entity("Alice")
    canonical_dir = env.create_entity("Alice Johnson")

    meta_a = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": "143022_300",
                    "source": "mic_audio",
                    "sentence_id": i,
                }
            )
            for i in range(5)
        ],
        dtype=str,
    )
    meta_b = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": "143022_300",
                    "source": "mic_audio",
                    "sentence_id": i + 10,
                }
            )
            for i in range(5)
        ],
        dtype=str,
    )
    np.savez_compressed(
        alias_dir / "voiceprints.npz", embeddings=embeddings, metadata=meta_a
    )
    np.savez_compressed(
        canonical_dir / "voiceprints.npz", embeddings=embeddings, metadata=meta_b
    )

    from apps.speakers.bootstrap import resolve_name_variants

    stats = resolve_name_variants(dry_run=False)

    assert len(stats["auto_merged"]) == 1

    # Deep merge: alias entity should be deleted
    assert load_journal_entity("alice") is None
    assert "alice" not in scan_journal_entities()


def test_resolve_name_variants_dry_run_unchanged(speakers_env):
    """resolve_name_variants(dry_run=True) does not delete entities."""
    env = speakers_env()

    embedding = np.random.default_rng(42).standard_normal(256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    embeddings = np.tile(embedding.reshape(1, -1), (5, 1))

    alias_dir = env.create_entity("Bob")
    canonical_dir = env.create_entity("Bob Smith")

    meta_a = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": "143022_300",
                    "source": "mic_audio",
                    "sentence_id": i,
                }
            )
            for i in range(5)
        ],
        dtype=str,
    )
    meta_b = np.array(
        [
            json.dumps(
                {
                    "day": "20240101",
                    "segment_key": "143022_300",
                    "source": "mic_audio",
                    "sentence_id": i + 10,
                }
            )
            for i in range(5)
        ],
        dtype=str,
    )
    np.savez_compressed(
        alias_dir / "voiceprints.npz", embeddings=embeddings, metadata=meta_a
    )
    np.savez_compressed(
        canonical_dir / "voiceprints.npz", embeddings=embeddings, metadata=meta_b
    )

    from apps.speakers.bootstrap import resolve_name_variants

    stats = resolve_name_variants(dry_run=True)

    assert len(stats["auto_merged"]) == 1
    # Dry run: both entities still exist
    assert load_journal_entity("bob") is not None
    assert load_journal_entity("bob_smith") is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_error_missing_entity(speakers_env):
    """CLI merge-names outputs error JSON and exits 1 for unknown entity."""
    speakers_env()
    result = _runner.invoke(speakers_app, ["merge-names", "Nobody", "Also Nobody"])
    assert result.exit_code == 1


def test_cli_success(speakers_env):
    """CLI merge-names outputs JSON with deep merge fields on success."""
    env = speakers_env()
    entity_a = env.create_entity("Alice Alias")
    entity_b = env.create_entity("Alice Canonical")

    emb_a = np.random.default_rng(42).standard_normal((3, 256)).astype(np.float32)
    emb_b = np.random.default_rng(99).standard_normal((3, 256)).astype(np.float32)
    meta_a = np.array(
        [json.dumps({"key": f"a_{i}"}) for i in range(3)], dtype=str
    )
    meta_b = np.array(
        [json.dumps({"key": f"b_{i}"}) for i in range(3)], dtype=str
    )
    np.savez_compressed(
        entity_a / "voiceprints.npz", embeddings=emb_a, metadata=meta_a
    )
    np.savez_compressed(
        entity_b / "voiceprints.npz", embeddings=emb_b, metadata=meta_b
    )

    result = _runner.invoke(
        speakers_app, ["merge-names", "Alice Alias", "Alice Canonical"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["merged"] is True
    assert data["canonical_name"] == "Alice Canonical"
    assert data["alias_id"] == "alice_alias"
    assert "segments_scanned" in data
    assert "facets_merged" in data
