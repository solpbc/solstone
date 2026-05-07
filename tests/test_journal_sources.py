# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import re
import stat
from importlib import import_module

import pytest
from flask import Flask, abort, g, jsonify

import solstone.convey.state as convey_state
from solstone.think.utils import now_ms

journal_sources = import_module("solstone.apps.import.journal_sources")
STATE_AREAS = journal_sources.STATE_AREAS
create_state_directory = journal_sources.create_state_directory
find_journal_source_by_name = journal_sources.find_journal_source_by_name
generate_key = journal_sources.generate_key
get_state_directory = journal_sources.get_state_directory
is_valid_journal_source_name = journal_sources.is_valid_journal_source_name
list_journal_sources = journal_sources.list_journal_sources
load_journal_source = journal_sources.load_journal_source
require_journal_source = journal_sources.require_journal_source
save_journal_source = journal_sources.save_journal_source


@pytest.fixture
def journal_env(tmp_path, monkeypatch):
    monkeypatch.setattr(convey_state, "journal_root", str(tmp_path), raising=False)
    (tmp_path / "apps" / "import" / "journal_sources").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


def _source(name: str, key: str, created_at: int = 0) -> dict:
    return {
        "key": key,
        "name": name,
        "created_at": created_at,
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


@pytest.fixture
def manifest_env(journal_env):
    """Journal env with a saved source and state directory."""
    key = generate_key()
    source = _source("manifest-test", key, created_at=123)
    save_journal_source(source)
    create_state_directory(journal_env, key[:8])
    return {"root": journal_env, "key": key, "source": source}


@pytest.fixture
def manifest_app(manifest_env):
    app = Flask(__name__)

    @app.route("/journal/<key_prefix>/manifest/<area>")
    @require_journal_source
    def journal_source_manifest(key_prefix: str, area: str):
        if g.journal_source["key"][:8] != key_prefix:
            abort(403, description="Key prefix mismatch")
        if area not in STATE_AREAS:
            abort(404, description="Unknown manifest area")
        state_path = get_state_directory(key_prefix) / area / "state.json"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        return jsonify(data)

    return app


def test_generate_key():
    key = generate_key()
    assert len(key) == 43
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", key)


def test_create_and_load(journal_env):
    key = generate_key()
    source = _source("alpha", key, created_at=123)

    assert save_journal_source(source) is True

    loaded = load_journal_source(key)
    assert loaded == source

    source_path = journal_env / "apps" / "import" / "journal_sources" / "alpha.json"
    assert source_path.exists()
    assert stat.S_IMODE(source_path.stat().st_mode) == 0o600


def test_load_wrong_key(journal_env):
    source = _source("alpha", generate_key(), created_at=123)
    assert save_journal_source(source) is True

    assert load_journal_source(generate_key()) is None


def test_list_journal_sources(journal_env):
    first = _source("first", generate_key(), created_at=100)
    second = _source("second", generate_key(), created_at=300)
    third = _source("third", generate_key(), created_at=200)

    assert save_journal_source(first) is True
    assert save_journal_source(second) is True
    assert save_journal_source(third) is True

    assert [source["name"] for source in list_journal_sources()] == [
        "second",
        "third",
        "first",
    ]


def test_find_by_name(journal_env):
    source = _source("alpha", generate_key(), created_at=123)
    assert save_journal_source(source) is True

    assert find_journal_source_by_name("alpha") == source
    assert find_journal_source_by_name("nonexistent") is None


def test_create_state_directory(journal_env):
    state_dir = create_state_directory(journal_env, "abcd1234")

    source_path = state_dir / "source.json"
    assert source_path.exists()
    assert json.loads(source_path.read_text(encoding="utf-8")) == {}

    for area in STATE_AREAS:
        state_path = state_dir / area / "state.json"
        assert state_path.exists()
        assert json.loads(state_path.read_text(encoding="utf-8")) == {}


def test_duplicate_name_rejected(journal_env):
    source = _source("alpha", generate_key(), created_at=123)
    assert save_journal_source(source) is True

    assert find_journal_source_by_name("alpha") == source


def test_invalid_name_rejected(journal_env):
    assert is_valid_journal_source_name("../alpha") is False
    assert (
        save_journal_source(_source("../alpha", generate_key(), created_at=123))
        is False
    )
    assert find_journal_source_by_name("../alpha") is None
    assert not (journal_env.parent / "alpha.json").exists()


def test_revoke_sets_fields(journal_env):
    key = generate_key()
    source = _source("alpha", key, created_at=123)
    assert save_journal_source(source) is True

    revoked_at = now_ms()
    source["revoked"] = True
    source["revoked_at"] = revoked_at
    assert save_journal_source(source) is True

    loaded = load_journal_source(key)
    assert loaded is not None
    assert loaded["revoked"] is True
    assert loaded["revoked_at"] == revoked_at


def test_auth_decorator_valid_key(journal_env):
    key = generate_key()
    source = _source("alpha", key, created_at=123)
    assert save_journal_source(source) is True

    app = Flask(__name__)

    @app.route("/protected")
    @require_journal_source
    def protected():
        return jsonify({"name": g.journal_source["name"]})

    response = app.test_client().get(
        "/protected",
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"name": "alpha"}


def test_auth_decorator_missing_key(journal_env):
    app = Flask(__name__)

    @app.route("/protected")
    @require_journal_source
    def protected():
        return jsonify({"name": g.journal_source["name"]})

    response = app.test_client().get("/protected")

    assert response.status_code == 401


def test_auth_decorator_invalid_key(journal_env):
    app = Flask(__name__)

    @app.route("/protected")
    @require_journal_source
    def protected():
        return jsonify({"name": g.journal_source["name"]})

    response = app.test_client().get(
        "/protected",
        headers={"Authorization": "Bearer does-not-exist"},
    )

    assert response.status_code == 401


def test_auth_decorator_revoked_key(journal_env):
    key = generate_key()
    source = _source("alpha", key, created_at=123)
    source["revoked"] = True
    source["revoked_at"] = now_ms()
    assert save_journal_source(source) is True

    app = Flask(__name__)

    @app.route("/protected")
    @require_journal_source
    def protected():
        return jsonify({"name": g.journal_source["name"]})

    response = app.test_client().get(
        "/protected",
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 403


@pytest.mark.parametrize("area", STATE_AREAS)
def test_manifest_empty_state(manifest_app, manifest_env, area):
    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/{area}",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 200
    assert response.get_json() == {}


def test_manifest_populated_state(manifest_app, manifest_env):
    data = {"days": {"2026-04-01": {"count": 5}}}
    state_path = (
        get_state_directory(manifest_env["key"][:8]) / "segments" / "state.json"
    )
    state_path.write_text(json.dumps(data), encoding="utf-8")

    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 200
    assert response.get_json() == data


def test_manifest_missing_state_file(manifest_app, manifest_env):
    state_path = (
        get_state_directory(manifest_env["key"][:8]) / "segments" / "state.json"
    )
    state_path.unlink()

    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 200
    assert response.get_json() == {}


def test_manifest_malformed_json(manifest_app, manifest_env):
    state_path = (
        get_state_directory(manifest_env["key"][:8]) / "segments" / "state.json"
    )
    state_path.write_text("not valid json{{{", encoding="utf-8")

    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 200
    assert response.get_json() == {}


def test_manifest_invalid_area(manifest_app, manifest_env):
    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/invalid_area",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 404


def test_manifest_key_prefix_mismatch(manifest_app, manifest_env):
    other_prefix = "deadbeef"
    assert other_prefix != manifest_env["key"][:8]

    response = manifest_app.test_client().get(
        f"/journal/{other_prefix}/manifest/segments",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 403


def test_manifest_auth_missing(manifest_app, manifest_env):
    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments"
    )

    assert response.status_code == 401


def test_manifest_auth_invalid(manifest_app, manifest_env):
    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments",
        headers={"Authorization": "Bearer does-not-exist"},
    )

    assert response.status_code == 401


def test_manifest_auth_revoked(manifest_app, manifest_env):
    source = manifest_env["source"]
    source["revoked"] = True
    source["revoked_at"] = now_ms()
    assert save_journal_source(source) is True

    response = manifest_app.test_client().get(
        f"/journal/{manifest_env['key'][:8]}/manifest/segments",
        headers={"Authorization": f"Bearer {manifest_env['key']}"},
    )

    assert response.status_code == 403
