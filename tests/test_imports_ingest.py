# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import json
from importlib import import_module

import pytest
from flask import Blueprint, Flask

import solstone.convey.state as convey_state

journal_sources = import_module("solstone.apps.import.journal_sources")
ingest = import_module("solstone.apps.import.ingest")

create_state_directory = journal_sources.create_state_directory
generate_key = journal_sources.generate_key
get_state_directory = journal_sources.get_state_directory
load_journal_source = journal_sources.load_journal_source
save_journal_source = journal_sources.save_journal_source
register_ingest_routes = ingest.register_ingest_routes


@pytest.fixture
def journal_env(tmp_path, monkeypatch):
    monkeypatch.setattr(convey_state, "journal_root", str(tmp_path), raising=False)
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


def _sample_import(import_id="20260101_090000"):
    return {
        "id": import_id,
        "import_json": {
            "original_filename": "test.zip",
            "upload_timestamp": 1767258000000,
            "upload_datetime": "2026-01-01T09:00:00",
            "user_timestamp": import_id,
            "file_size": 1234,
            "mime_type": "application/zip",
            "facet": "work",
            "setting": "calendar",
            "file_path": f"imports/{import_id}/test.zip",
        },
        "imported_json": {
            "processed_timestamp": import_id,
            "processing_completed": "2026-01-01T09:10:00",
            "total_files_created": 1,
            "all_created_files": ["20260101/import.ics/090000_300/event.md"],
            "segments": ["090000_300"],
            "source_type": "ics",
            "source_display": "Calendar",
            "entries_written": 1,
            "entities_seeded": 0,
            "date_range": ["20260101", "20260101"],
            "target_day": "20260101",
        },
        "content_manifest": [
            {
                "id": "event-0",
                "title": "Test Event",
                "date": "20260101",
                "type": "event",
            }
        ],
    }


def _import_hash(item: dict) -> str:
    hash_input = json.dumps(
        {
            "import_json": item["import_json"],
            "imported_json": item["imported_json"],
            "content_manifest": item["content_manifest"],
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(hash_input).hexdigest()


def _post_imports(client, key, key_prefix, imports_list):
    return client.post(
        f"/app/import/journal/{key_prefix}/ingest/imports",
        headers={"Authorization": f"Bearer {key}"},
        json={"imports": imports_list},
    )


def _read_state(key_prefix: str) -> dict:
    state_path = get_state_directory(key_prefix) / "imports" / "state.json"
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_log(key_prefix: str) -> list[dict]:
    log_path = get_state_directory(key_prefix) / "imports" / "log.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_auth_required(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/imports",
        json={"imports": []},
    )
    assert response.status_code == 401


def test_key_prefix_mismatch(ingest_env):
    env = ingest_env
    response = env["client"].post(
        "/app/import/journal/deadbeef/ingest/imports",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={"imports": []},
    )
    assert response.status_code == 403


def test_invalid_json(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/imports",
        headers={"Authorization": f"Bearer {env['key']}"},
        data="not-json",
        content_type="application/json",
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "I couldn't read that JSON request."
    assert payload["reason_code"] == "invalid_json_request"
    assert payload["detail"] == "Invalid JSON body"


def test_missing_imports_array(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/imports",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={},
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "I couldn't find a required field."
    assert payload["reason_code"] == "missing_required_field"
    assert payload["detail"] == "Missing imports array"


def test_copy_new_import(ingest_env):
    env = ingest_env
    item = _sample_import()
    response = _post_imports(env["client"], env["key"], env["key_prefix"], [item])
    body = response.get_json()
    import_dir = env["root"] / "imports" / item["id"]

    assert response.status_code == 200
    assert body == {"copied": 1, "skipped": 0, "staged": 0, "errors": []}
    assert (
        json.loads((import_dir / "import.json").read_text(encoding="utf-8"))
        == item["import_json"]
    )
    assert (
        json.loads((import_dir / "imported.json").read_text(encoding="utf-8"))
        == item["imported_json"]
    )
    assert [
        json.loads(line)
        for line in (import_dir / "content_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ] == item["content_manifest"]
    assert _read_state(env["key_prefix"]) == {
        "received": {item["id"]: _import_hash(item)}
    }
    assert _read_log(env["key_prefix"])[0]["action"] == "copied"
    source = load_journal_source(env["key"])
    assert source["stats"]["imports_received"] == 1


def test_dedup_identical(ingest_env):
    env = ingest_env
    item = _sample_import()

    first = _post_imports(env["client"], env["key"], env["key_prefix"], [item])
    second = _post_imports(env["client"], env["key"], env["key_prefix"], [item])

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json() == {"copied": 0, "skipped": 1, "staged": 0, "errors": []}
    assert _read_log(env["key_prefix"])[-1]["reason"] == "idempotent"


def test_id_collision(ingest_env):
    env = ingest_env
    item = _sample_import()
    (env["root"] / "imports" / item["id"]).mkdir(parents=True, exist_ok=True)

    response = _post_imports(env["client"], env["key"], env["key_prefix"], [item])
    body = response.get_json()
    staged_path = (
        get_state_directory(env["key_prefix"])
        / "imports"
        / "staged"
        / f"{item['id']}.json"
    )

    assert response.status_code == 200
    assert body == {"copied": 0, "skipped": 0, "staged": 1, "errors": []}
    assert staged_path.exists()
    assert (
        json.loads(staged_path.read_text(encoding="utf-8"))["reason"] == "id_collision"
    )


def test_multiple_imports(ingest_env):
    env = ingest_env
    first = _sample_import("20260101_090000")
    second = _sample_import("20260101_100000")
    third = _sample_import("20260101_110000")
    _post_imports(env["client"], env["key"], env["key_prefix"], [first])
    (env["root"] / "imports" / third["id"]).mkdir(parents=True, exist_ok=True)

    response = _post_imports(
        env["client"], env["key"], env["key_prefix"], [first, second, third]
    )

    assert response.status_code == 200
    assert response.get_json() == {"copied": 1, "skipped": 1, "staged": 1, "errors": []}


def test_state_manifest(ingest_env):
    env = ingest_env
    item = _sample_import()

    response = _post_imports(env["client"], env["key"], env["key_prefix"], [item])

    assert response.status_code == 200
    assert _read_state(env["key_prefix"]) == {
        "received": {item["id"]: _import_hash(item)}
    }


def test_stats_update(ingest_env):
    env = ingest_env
    first = _sample_import("20260101_090000")
    second = _sample_import("20260101_100000")
    collision = _sample_import("20260101_110000")
    (env["root"] / "imports" / collision["id"]).mkdir(parents=True, exist_ok=True)

    response = _post_imports(
        env["client"], env["key"], env["key_prefix"], [first, second, collision]
    )
    source = load_journal_source(env["key"])

    assert response.status_code == 200
    assert source["stats"]["imports_received"] == 2
