# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from importlib import import_module

import pytest
from flask import Blueprint, Flask

import convey.state
import think.utils

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
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    think.utils._journal_path_cache = None
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


def _post_config(client, key, key_prefix, config):
    return client.post(
        f"/app/import/journal/{key_prefix}/ingest/config",
        headers={"Authorization": f"Bearer {key}"},
        json={"config": config},
    )


def _sample_config():
    return {
        "identity": {"name": "Remote User", "preferred": "Remote", "timezone": "UTC"},
        "convey": {
            "password_hash": "secret_hash",
            "secret": "secret_value",
            "trust_localhost": True,
        },
        "setup": {"completed_at": 12345},
        "env": {"API_KEY": "xyz"},
        "retention": {"days": 30},
    }


def _write_target_config(root, config):
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "journal.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    think.utils._journal_path_cache = None


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_log(key_prefix):
    log_path = get_state_directory(key_prefix) / "config" / "log.jsonl"
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
        f"/app/import/journal/{env['key_prefix']}/ingest/config",
        json={"config": {}},
    )
    assert response.status_code == 401


def test_key_prefix_mismatch(ingest_env):
    env = ingest_env
    response = env["client"].post(
        "/app/import/journal/deadbeef/ingest/config",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={"config": {}},
    )
    assert response.status_code == 403


def test_invalid_json(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/config",
        headers={"Authorization": f"Bearer {env['key']}"},
        data="not-json",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid JSON body"}


def test_missing_config(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/config",
        headers={"Authorization": f"Bearer {env['key']}"},
        json={},
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing config object"}


def test_config_staged(ingest_env):
    env = ingest_env
    target_config = {"identity": {"name": "Local User"}, "retention": {"days": 90}}
    _write_target_config(env["root"], target_config)
    config = _sample_config()

    response = _post_config(env["client"], env["key"], env["key_prefix"], config)
    body = response.get_json()
    state_dir = get_state_directory(env["key_prefix"]) / "config"
    source = load_journal_source(env["key"])

    assert response.status_code == 200
    assert body == {"staged": True, "skipped": False, "diff_fields": 5}
    assert (state_dir / "source_config.json").exists()
    assert (state_dir / "diff.json").exists()
    assert "last_hash" in _read_json(state_dir / "state.json")
    assert _read_log(env["key_prefix"])[0]["action"] == "staged"
    assert source["stats"]["config_received"] == 1


def test_diff_categorization(ingest_env):
    env = ingest_env
    target_config = {"identity": {"name": "Local User"}, "retention": {"days": 90}}
    _write_target_config(env["root"], target_config)
    config = {"identity": {"name": "Remote User"}, "retention": {"days": 30}}

    response = _post_config(env["client"], env["key"], env["key_prefix"], config)
    diff = _read_json(get_state_directory(env["key_prefix"]) / "config" / "diff.json")

    assert response.status_code == 200
    assert diff["identity.name"]["category"] == "transferable"
    assert diff["retention.days"]["category"] == "preference"


def test_never_transfer_excluded(ingest_env):
    env = ingest_env
    _write_target_config(env["root"], {"identity": {"name": "Local User"}})
    config = _sample_config()

    response = _post_config(env["client"], env["key"], env["key_prefix"], config)
    diff = _read_json(get_state_directory(env["key_prefix"]) / "config" / "diff.json")

    assert response.status_code == 200
    assert "convey.password_hash" not in diff
    assert "convey.secret" not in diff
    assert not any(key.startswith("env.") for key in diff)


def test_idempotent(ingest_env):
    env = ingest_env
    _write_target_config(env["root"], {"identity": {"name": "Local User"}})
    config = _sample_config()

    first = _post_config(env["client"], env["key"], env["key_prefix"], config)
    second = _post_config(env["client"], env["key"], env["key_prefix"], config)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json() == {
        "staged": False,
        "skipped": True,
        "reason": "idempotent",
    }


def test_config_always_staged(ingest_env):
    env = ingest_env
    _write_target_config(env["root"], {"identity": {"name": "Local User"}})
    config = {"identity": {"name": "Remote User"}}

    response = _post_config(env["client"], env["key"], env["key_prefix"], config)

    assert response.status_code == 200
    assert response.get_json()["staged"] is True
