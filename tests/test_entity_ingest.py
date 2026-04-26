# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import json
from importlib import import_module

import pytest
from flask import Blueprint, Flask

import convey.state
from think.entities.core import entity_slug
from think.entities.journal import (
    load_all_journal_entities,
    load_journal_entity,
    save_journal_entity,
)

journal_sources = import_module("apps.import.journal_sources")
ingest = import_module("apps.import.ingest")

create_state_directory = journal_sources.create_state_directory
generate_key = journal_sources.generate_key
get_state_directory = journal_sources.get_state_directory
load_journal_source = journal_sources.load_journal_source
save_journal_source = journal_sources.save_journal_source
register_ingest_routes = ingest.register_ingest_routes


@pytest.fixture
def journal_env(tmp_path, monkeypatch):
    monkeypatch.setattr(convey.state, "journal_root", str(tmp_path), raising=False)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    (tmp_path / "apps" / "import" / "journal_sources").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


def _source(name="test-source", key=None, **overrides):
    if key is None:
        key = generate_key()
    source = {
        "name": name,
        "key": key,
        "created_at": 1000,
        "enabled": True,
        "revoked": False,
        "revoked_at": None,
        "stats": {
            "segments_received": 0,
            "entities_received": 0,
            "facets_received": 0,
            "imports_received": 0,
            "config_received": 0,
        },
    }
    source.update(overrides)
    return source


@pytest.fixture
def ingest_env(journal_env):
    key = generate_key()
    source = _source(key=key)
    save_journal_source(source)
    key_prefix = key[:8]
    create_state_directory(journal_env, key_prefix)

    app = Flask(__name__)
    app.config["TESTING"] = True
    bp = Blueprint("import-test", __name__, url_prefix="/app/import")
    register_ingest_routes(bp)
    app.register_blueprint(bp)

    return {
        "root": journal_env,
        "key": key,
        "key_prefix": key_prefix,
        "source": source,
        "client": app.test_client(),
    }


def _post_entities(client, key, key_prefix, entities):
    return client.post(
        f"/app/import/journal/{key_prefix}/ingest/entities",
        headers={"Authorization": f"Bearer {key}"},
        json={"entities": entities},
    )


def _read_state(key_prefix: str) -> dict:
    state_path = get_state_directory(key_prefix) / "entities" / "state.json"
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_log(key_prefix: str) -> list[dict]:
    log_path = get_state_directory(key_prefix) / "entities" / "log.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_staged(key_prefix: str, entity_id: str) -> dict:
    staged_path = (
        get_state_directory(key_prefix) / "entities" / "staged" / f"{entity_id}.json"
    )
    return json.loads(staged_path.read_text(encoding="utf-8"))


def _entity_hash(entity: dict) -> str:
    return hashlib.sha256(
        json.dumps(entity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def test_auto_merge_exact(ingest_env):
    env = ingest_env
    save_journal_entity(
        {
            "id": "alice_johnson",
            "name": "Alice Johnson",
            "type": "Person",
            "aka": ["Ali"],
            "emails": ["alice@old.com"],
            "created_at": 2000,
        }
    )

    source = {
        "name": "Alice Johnson",
        "type": "Person",
        "aka": ["AJ"],
        "emails": ["alice@new.com"],
        "created_at": 1000,
    }

    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200
    assert response.get_json() == {
        "auto_merged": 1,
        "created": 0,
        "staged": 0,
        "skipped": 0,
        "errors": [],
    }

    merged = load_journal_entity("alice_johnson")
    assert merged is not None
    assert merged["aka"] == ["AJ", "Ali"]
    assert merged["emails"] == ["alice@old.com", "alice@new.com"]
    assert merged["created_at"] == 1000

    state = _read_state(env["key_prefix"])
    assert state["id_map"] == {"alice_johnson": "alice_johnson"}
    assert state["received"]["alice_johnson"] == _entity_hash(
        {**source, "id": "alice_johnson"}
    )

    log_entries = _read_log(env["key_prefix"])
    assert len(log_entries) == 1
    entry = log_entries[0]
    assert "ts" in entry
    assert entry["action"] == "auto_merged"
    assert entry["item_type"] == "entity"
    assert entry["item_id"] == "alice_johnson"
    assert entry["match_tier"] == 1
    assert entry["reason"] == "high_confidence_match"
    assert entry["source"]["name"] == "Alice Johnson"
    assert entry["target"]["id"] == "alice_johnson"
    assert entry["target"]["aka"] == ["AJ", "Ali"]
    assert isinstance(entry["fields_changed"], list)


def test_auto_merge_case_insensitive(ingest_env):
    env = ingest_env
    save_journal_entity(
        {"id": "alice_johnson", "name": "alice johnson", "type": "Person"}
    )

    response = _post_entities(
        env["client"],
        env["key"],
        env["key_prefix"],
        [{"name": "Alice Johnson", "type": "Person"}],
    )

    assert response.status_code == 200
    assert response.get_json()["auto_merged"] == 1

    log_entries = _read_log(env["key_prefix"])
    assert log_entries[0]["match_tier"] == 2


def test_auto_merge_email_match(ingest_env):
    env = ingest_env
    save_journal_entity(
        {
            "id": "alice_contact",
            "name": "Alice Contact",
            "type": "Person",
            "emails": ["alice@example.com"],
        }
    )

    response = _post_entities(
        env["client"],
        env["key"],
        env["key_prefix"],
        [{"name": "alice@example.com", "type": "Person"}],
    )

    assert response.status_code == 200
    assert response.get_json()["auto_merged"] == 1

    log_entries = _read_log(env["key_prefix"])
    assert log_entries[0]["match_tier"] == 3


def test_auto_merge_slug_match(ingest_env):
    env = ingest_env
    save_journal_entity({"id": "alice_johnson", "name": "AJ", "type": "Person"})

    response = _post_entities(
        env["client"],
        env["key"],
        env["key_prefix"],
        [{"name": "Alice Johnson", "type": "Person"}],
    )

    assert response.status_code == 200
    assert response.get_json()["auto_merged"] == 1

    log_entries = _read_log(env["key_prefix"])
    assert log_entries[0]["match_tier"] == 4


def test_stage_low_confidence(ingest_env):
    env = ingest_env
    save_journal_entity(
        {"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}
    )

    source = {"name": "Alce Jonson", "type": "Person"}
    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200
    assert response.get_json() == {
        "auto_merged": 0,
        "created": 0,
        "staged": 1,
        "skipped": 0,
        "errors": [],
    }

    staged = _read_staged(env["key_prefix"], "alce_jonson")
    assert staged["reason"] == "low_confidence_match"
    assert staged["match_candidates"] == [
        {"id": "alice_johnson", "name": "Alice Johnson", "tier": 8}
    ]
    state = _read_state(env["key_prefix"])
    assert "alce_jonson" not in state["id_map"]
    assert "alce_jonson" in state["received"]


def test_stage_id_collision(ingest_env):
    env = ingest_env
    save_journal_entity({"id": "test", "name": "Test Entity", "type": "Tool"})

    source = {
        "id": "test",
        "name": "Completely Different Name",
        "type": "Person",
    }
    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200
    assert response.get_json()["staged"] == 1

    staged = _read_staged(env["key_prefix"], "test")
    assert staged["reason"] == "id_collision"
    assert staged["match_candidates"] == [
        {"id": "test", "name": "Test Entity", "tier": None}
    ]
    state = _read_state(env["key_prefix"])
    assert "test" not in state["id_map"]
    assert "test" in state["received"]


def test_stage_principal_conflict(ingest_env):
    env = ingest_env
    save_journal_entity(
        {
            "id": "existing_principal",
            "name": "Existing Principal",
            "type": "Person",
            "is_principal": True,
        }
    )

    source = {"name": "New Principal", "type": "Person", "is_principal": True}
    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200
    assert response.get_json()["staged"] == 1

    staged = _read_staged(env["key_prefix"], "new_principal")
    assert staged["reason"] == "principal_conflict"
    assert staged["match_candidates"] == []
    state = _read_state(env["key_prefix"])
    assert "new_principal" not in state["id_map"]
    assert "new_principal" in state["received"]


def test_auto_create(ingest_env):
    env = ingest_env
    source = {"name": "Fresh Entity", "type": "Tool"}

    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200
    assert response.get_json()["created"] == 1
    assert load_journal_entity("fresh_entity") is not None

    state = _read_state(env["key_prefix"])
    assert state["id_map"] == {"fresh_entity": "fresh_entity"}


def test_idempotent(ingest_env):
    env = ingest_env
    source = {"name": "Repeat Entity", "type": "Tool"}

    first = _post_entities(env["client"], env["key"], env["key_prefix"], [source])
    second = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["created"] == 1
    assert second.get_json() == {
        "auto_merged": 0,
        "created": 0,
        "staged": 0,
        "skipped": 1,
        "errors": [],
    }


def test_idempotent_content_change(ingest_env):
    env = ingest_env
    first = {"name": "Mutable Entity", "type": "Tool"}
    second = {"name": "Mutable Entity", "type": "Tool", "aka": ["Mut"]}

    initial = _post_entities(env["client"], env["key"], env["key_prefix"], [first])
    updated = _post_entities(env["client"], env["key"], env["key_prefix"], [second])

    assert initial.status_code == 200
    assert updated.status_code == 200
    assert updated.get_json()["created"] == 0
    assert updated.get_json()["auto_merged"] == 1
    assert updated.get_json()["skipped"] == 0
    state = _read_state(env["key_prefix"])
    expected_second_entity = {**second, "id": "mutable_entity"}
    assert state["received"]["mutable_entity"] == _entity_hash(expected_second_entity)


def test_error_isolation(ingest_env):
    env = ingest_env
    entities = [
        {"name": "Valid One", "type": "Person"},
        {"no_name": True},
        {"name": "Valid Two", "type": "Tool"},
    ]

    response = _post_entities(env["client"], env["key"], env["key_prefix"], entities)

    assert response.status_code == 200
    assert response.get_json() == {
        "auto_merged": 0,
        "created": 2,
        "staged": 0,
        "skipped": 0,
        "errors": [{"entity_id": "", "error": "Entity name is required"}],
    }
    assert load_journal_entity("valid_one") is not None
    assert load_journal_entity("valid_two") is not None


def test_auth_missing(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/entities",
        json={"entities": []},
    )

    assert response.status_code == 401


def test_auth_invalid(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/entities",
        headers={"Authorization": "Bearer wrong-token"},
        json={"entities": []},
    )

    assert response.status_code == 401


def test_auth_revoked(ingest_env):
    env = ingest_env
    env["source"]["revoked"] = True
    env["source"]["revoked_at"] = 12345
    save_journal_source(env["source"])

    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/entities",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={"entities": []},
    )

    assert response.status_code == 403


def test_key_prefix_mismatch(ingest_env):
    env = ingest_env
    response = env["client"].post(
        "/app/import/journal/deadbeef/ingest/entities",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={"entities": []},
    )

    assert response.status_code == 403


def test_stats_update(ingest_env):
    env = ingest_env
    save_journal_entity(
        {"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}
    )

    response = _post_entities(
        env["client"],
        env["key"],
        env["key_prefix"],
        [
            {"name": "Alice Johnson", "type": "Person"},
            {"name": "Fresh Entity", "type": "Tool"},
            {"name": "Alic Johnson", "type": "Person"},
        ],
    )
    source = load_journal_source(env["key"])

    assert response.status_code == 200
    assert source["stats"]["entities_received"] == 2


def test_state_manifest(ingest_env):
    env = ingest_env
    source = {"name": "Manifest Entity", "type": "Tool"}

    response = _post_entities(env["client"], env["key"], env["key_prefix"], [source])

    assert response.status_code == 200

    state = _read_state(env["key_prefix"])
    expected = {**source, "id": "manifest_entity"}
    assert state == {
        "id_map": {"manifest_entity": "manifest_entity"},
        "received": {"manifest_entity": _entity_hash(expected)},
    }


def test_request_body_validation(ingest_env):
    env = ingest_env

    missing = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/entities",
        headers={"Authorization": f"Bearer {env['key']}"},
        data="not-json",
        content_type="application/json",
    )
    invalid = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/entities",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={"wrong": []},
    )

    assert missing.status_code == 400
    assert missing.get_json() == {"error": "Invalid JSON body"}
    assert invalid.status_code == 400
    assert invalid.get_json() == {"error": "Missing entities array"}


def test_empty_entities_list(ingest_env):
    env = ingest_env
    response = _post_entities(env["client"], env["key"], env["key_prefix"], [])

    assert response.status_code == 200
    assert response.get_json() == {
        "auto_merged": 0,
        "created": 0,
        "staged": 0,
        "skipped": 0,
        "errors": [],
    }


def test_load_all_entities_sees_created_entity(ingest_env):
    env = ingest_env

    response = _post_entities(
        env["client"],
        env["key"],
        env["key_prefix"],
        [{"name": "Visible Entity", "type": "Tool"}],
    )

    assert response.status_code == 200
    assert "visible_entity" in load_all_journal_entities()
    assert entity_slug("Visible Entity") == "visible_entity"
