# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from importlib import import_module
from io import BytesIO

import pytest
from flask import Blueprint, Flask

import solstone.convey.state as convey_state
from solstone.observe.utils import compute_bytes_sha256

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
    """Set up journal root and source storage."""
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
    """Set up a source and create an app with the ingest route."""
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


def _build_ingest_payload(segments):
    metadata = {
        "segments": [
            {
                "day": segment["day"],
                "stream": segment["stream"],
                "segment_key": segment["segment_key"],
                "files": [filename for filename, _content in segment["files"]],
            }
            for segment in segments
        ]
    }

    data = {"metadata": json.dumps(metadata)}
    for idx, segment in enumerate(segments):
        data[f"files_{idx}"] = [
            (BytesIO(content), filename) for filename, content in segment["files"]
        ]
    return data


def _post_ingest(client, key, key_prefix, segments):
    return client.post(
        f"/app/import/journal/{key_prefix}/ingest/segments",
        headers={"Authorization": f"Bearer {key}"},
        data=_build_ingest_payload(segments),
        content_type="multipart/form-data",
    )


def _read_state(key_prefix: str) -> dict:
    state_path = get_state_directory(key_prefix) / "segments" / "state.json"
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_log(key_prefix: str) -> list[dict]:
    log_path = get_state_directory(key_prefix) / "segments" / "log.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_ingest_new_segments(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [
                ("audio.flac", b"audio one"),
                ("transcript.jsonl", b'{"text":"one"}\n'),
            ],
        },
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143500_300",
            "files": [("audio.flac", b"audio two")],
        },
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    assert response.get_json() == {
        "segments_received": 2,
        "segments_skipped": 0,
        "segments_deconflicted": 0,
        "errors": [],
    }

    first_dir = env["root"] / "chronicle" / "20260413" / "laptop" / "143022_300"
    second_dir = env["root"] / "chronicle" / "20260413" / "laptop" / "143500_300"
    assert (first_dir / "audio.flac").read_bytes() == b"audio one"
    assert (first_dir / "transcript.jsonl").read_bytes() == b'{"text":"one"}\n'
    assert (second_dir / "audio.flac").read_bytes() == b"audio two"

    state_data = _read_state(env["key_prefix"])
    assert set(state_data["20260413"]) == {"laptop/143022_300", "laptop/143500_300"}

    log_entries = _read_log(env["key_prefix"])
    assert [entry["action"] for entry in log_entries] == ["copied", "copied"]
    assert log_entries[0]["item_id"] == "20260413/laptop/143022_300"
    assert log_entries[0]["item_type"] == "segment"
    assert log_entries[0]["reason"] == "new segment"


def test_ingest_duplicate_detection(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [
                ("audio.flac", b"audio one"),
                ("transcript.jsonl", b'{"text":"one"}\n'),
            ],
        },
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143500_300",
            "files": [("audio.flac", b"audio two")],
        },
    ]

    first = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    first_state_raw = (
        get_state_directory(env["key_prefix"]) / "segments" / "state.json"
    ).read_text(encoding="utf-8")

    second = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json() == {
        "segments_received": 0,
        "segments_skipped": 2,
        "segments_deconflicted": 0,
        "errors": [],
    }

    second_state_raw = (
        get_state_directory(env["key_prefix"]) / "segments" / "state.json"
    ).read_text(encoding="utf-8")
    assert second_state_raw == first_state_raw

    log_entries = _read_log(env["key_prefix"])
    assert [entry["action"] for entry in log_entries] == [
        "copied",
        "copied",
        "skipped",
        "skipped",
    ]


def test_ingest_deconfliction(ingest_env, monkeypatch):
    env = ingest_env
    target_dir = env["root"] / "chronicle" / "20260413" / "laptop" / "143022_300"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "audio.flac").write_bytes(b"existing audio")

    monkeypatch.setattr(
        ingest, "find_available_segment", lambda _parent, _seg: "143023_300"
    )

    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"new audio")],
        }
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    assert response.get_json() == {
        "segments_received": 1,
        "segments_skipped": 0,
        "segments_deconflicted": 1,
        "errors": [],
    }
    assert (
        env["root"] / "chronicle" / "20260413" / "laptop" / "143023_300" / "audio.flac"
    ).read_bytes() == b"new audio"

    state_data = _read_state(env["key_prefix"])
    assert "laptop/143023_300" in state_data["20260413"]
    assert "laptop/143022_300" in state_data["20260413"]

    log_entries = _read_log(env["key_prefix"])
    assert log_entries[0]["action"] == "deconflicted"
    assert log_entries[0]["original_key"] == "143022_300"
    assert log_entries[0]["item_id"] == "20260413/laptop/143023_300"


def test_ingest_batch_error_isolation(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "bad-day",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"broken")],
        },
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143500_300",
            "files": [("audio.flac", b"good")],
        },
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    body = response.get_json()
    assert body["segments_received"] == 1
    assert body["segments_skipped"] == 0
    assert body["segments_deconflicted"] == 0
    assert body["errors"] == [
        {
            "segment_key": "143022_300",
            "day": "bad-day",
            "error": "Invalid day format",
        }
    ]
    assert (
        env["root"] / "chronicle" / "20260413" / "laptop" / "143500_300" / "audio.flac"
    ).read_bytes() == b"good"


def test_ingest_missing_metadata(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/segments",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing metadata"}


def test_ingest_malformed_metadata(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/segments",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": "not-json"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid metadata JSON"}


def test_ingest_auth_missing(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/segments",
        data={"metadata": json.dumps({"segments": []})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 401


def test_ingest_auth_invalid(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/segments",
        headers={"Authorization": "Bearer wrong-token"},
        data={"metadata": json.dumps({"segments": []})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 401


def test_ingest_auth_revoked(ingest_env):
    env = ingest_env
    env["source"]["revoked"] = True
    env["source"]["revoked_at"] = 12345
    save_journal_source(env["source"])

    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/segments",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": json.dumps({"segments": []})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 403


def test_ingest_key_prefix_mismatch(ingest_env):
    env = ingest_env
    response = env["client"].post(
        "/app/import/journal/deadbeef/ingest/segments",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": json.dumps({"segments": []})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 403


def test_ingest_callosum_trigger(ingest_env, monkeypatch):
    env = ingest_env
    calls = []

    def mock_emit(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ingest, "emit", mock_emit)

    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"audio one")],
        }
    ]

    first = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    second = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == [
        (("supervisor", "request"), {"cmd": ["sol", "indexer", "--rescan"]})
    ]


def test_ingest_callosum_failure_isolated(ingest_env, monkeypatch):
    env = ingest_env

    def mock_emit(*_args, **_kwargs):
        raise RuntimeError("bridge down")

    monkeypatch.setattr(ingest, "emit", mock_emit)

    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"audio one")],
        }
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    assert response.get_json() == {
        "segments_received": 1,
        "segments_skipped": 0,
        "segments_deconflicted": 0,
        "errors": [],
    }


def test_ingest_skip_ignores_extra_existing_files(ingest_env):
    env = ingest_env
    segment_dir = env["root"] / "chronicle" / "20260413" / "laptop" / "143022_300"
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "audio.flac").write_bytes(b"audio one")
    (segment_dir / "extra.txt").write_bytes(b"keep me")

    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"audio one")],
        }
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    assert response.get_json() == {
        "segments_received": 0,
        "segments_skipped": 1,
        "segments_deconflicted": 0,
        "errors": [],
    }
    assert (segment_dir / "extra.txt").read_bytes() == b"keep me"
    state_data = _read_state(env["key_prefix"])
    assert "laptop/143022_300" in state_data["20260413"]
    assert (
        state_data["20260413"]["laptop/143022_300"]["files"][0]["name"] == "audio.flac"
    )


def test_ingest_stats_update(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"audio one")],
        }
    ]

    first = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    source_after_first = load_journal_source(env["key"])
    second = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    source_after_second = load_journal_source(env["key"])

    assert first.status_code == 200
    assert second.status_code == 200
    assert source_after_first["stats"]["segments_received"] == 1
    assert source_after_second["stats"]["segments_received"] == 1


def test_ingest_state_json_manifest_sync(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [
                ("audio.flac", b"audio one"),
                ("transcript.jsonl", b'{"text":"one"}\n'),
            ],
        }
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    state_data = _read_state(env["key_prefix"])

    assert response.status_code == 200
    assert state_data == {
        "20260413": {
            "laptop/143022_300": {
                "files": [
                    {
                        "name": "audio.flac",
                        "sha256": compute_bytes_sha256(b"audio one"),
                        "size": len(b"audio one"),
                    },
                    {
                        "name": "transcript.jsonl",
                        "sha256": compute_bytes_sha256(b'{"text":"one"}\n'),
                        "size": len(b'{"text":"one"}\n'),
                    },
                ]
            }
        }
    }


def test_ingest_default_stream_segment(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "_default",
            "segment_key": "143022_300",
            "files": [("transcript.jsonl", b'{"text":"default"}\n')],
        }
    ]

    response = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert response.status_code == 200
    assert response.get_json() == {
        "segments_received": 1,
        "segments_skipped": 0,
        "segments_deconflicted": 0,
        "errors": [],
    }

    state_data = _read_state(env["key_prefix"])
    assert "_default/143022_300" in state_data["20260413"]
    assert (
        env["root"]
        / "chronicle"
        / "20260413"
        / "_default"
        / "143022_300"
        / "transcript.jsonl"
    ).read_bytes() == b'{"text":"default"}\n'


def test_ingest_default_stream_idempotent(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "_default",
            "segment_key": "143022_300",
            "files": [("transcript.jsonl", b'{"text":"default"}\n')],
        }
    ]

    first = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    second = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["segments_received"] == 1
    assert second.get_json() == {
        "segments_received": 0,
        "segments_skipped": 1,
        "segments_deconflicted": 0,
        "errors": [],
    }


def test_ingest_idempotent(ingest_env):
    env = ingest_env
    segments = [
        {
            "day": "20260413",
            "stream": "laptop",
            "segment_key": "143022_300",
            "files": [("audio.flac", b"audio one")],
        }
    ]

    first = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    first_state = (
        get_state_directory(env["key_prefix"]) / "segments" / "state.json"
    ).read_text(encoding="utf-8")
    second = _post_ingest(env["client"], env["key"], env["key_prefix"], segments)
    second_state = (
        get_state_directory(env["key_prefix"]) / "segments" / "state.json"
    ).read_text(encoding="utf-8")
    source = load_journal_source(env["key"])

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["segments_received"] == 1
    assert second.get_json() == {
        "segments_received": 0,
        "segments_skipped": 1,
        "segments_deconflicted": 0,
        "errors": [],
    }
    assert first_state == second_state
    assert source["stats"]["segments_received"] == 1
