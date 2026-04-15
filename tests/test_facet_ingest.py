# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from io import BytesIO
from pathlib import Path

import pytest
from flask import Blueprint, Flask

import convey.state

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

    entity_state = {
        "id_map": {
            "source_entity": "target_entity",
            "same_entity": "same_entity",
        },
        "received": {},
    }
    (
        get_state_directory(key_prefix) / "entities" / "state.json"
    ).write_text(json.dumps(entity_state, indent=2), encoding="utf-8")

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


def _post_facets(client, key, key_prefix, facets_metadata, file_map):
    data = {"metadata": json.dumps({"facets": facets_metadata})}
    data.update(file_map)
    return client.post(
        f"/app/import/journal/{key_prefix}/ingest/facets",
        headers={"Authorization": f"Bearer {key}"},
        data=data,
        content_type="multipart/form-data",
    )


def _build_request(facets):
    facets_metadata = []
    file_map = {}
    for facet_idx, facet in enumerate(facets):
        facet_meta = {"name": facet["name"], "files": []}
        for file_idx, file_info in enumerate(facet["files"]):
            facet_meta["files"].append(
                {"path": file_info["path"], "type": file_info["type"]}
            )
            filename = Path(file_info["path"]).name
            file_map[f"files_{facet_idx}_{file_idx}"] = (
                BytesIO(file_info["content"]),
                filename,
            )
        facets_metadata.append(facet_meta)
    return facets_metadata, file_map


def _read_state(key_prefix: str) -> dict:
    state_path = get_state_directory(key_prefix) / "facets" / "state.json"
    return json.loads(state_path.read_text(encoding="utf-8"))


def _read_log(key_prefix: str) -> list[dict]:
    log_path = get_state_directory(key_prefix) / "facets" / "log.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_staged(key_prefix: str, facet: str, file_type: str, relative_path: str) -> dict:
    staged_name = relative_path.replace("/", "__") + ".staged.json"
    staged_path = (
        get_state_directory(key_prefix)
        / "facets"
        / "staged"
        / facet
        / file_type
        / staged_name
    )
    return json.loads(staged_path.read_text(encoding="utf-8"))


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(items: list[dict]) -> bytes:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items).encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_jsonl_bytes(items).decode("utf-8"), encoding="utf-8")


def test_auth_missing(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/facets",
        data={"metadata": json.dumps({"facets": []})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 401


def test_auth_invalid(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/facets",
        headers={"Authorization": "Bearer wrong-token"},
        data={"metadata": json.dumps({"facets": []})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 401


def test_auth_revoked(ingest_env):
    env = ingest_env
    env["source"]["revoked"] = True
    env["source"]["revoked_at"] = 12345
    save_journal_source(env["source"])

    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/facets",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": json.dumps({"facets": []})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 403


def test_key_prefix_mismatch(ingest_env):
    env = ingest_env
    response = env["client"].post(
        "/app/import/journal/deadbeef/ingest/facets",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": json.dumps({"facets": []})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 403


def test_missing_metadata(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/facets",
        headers={"Authorization": f"Bearer {env['key']}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing metadata"}


def test_invalid_metadata_json(ingest_env):
    env = ingest_env
    response = env["client"].post(
        f"/app/import/journal/{env['key_prefix']}/ingest/facets",
        headers={"Authorization": f"Bearer {env['key']}"},
        data={"metadata": "not-json"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid metadata JSON"}


def test_unsafe_facet_name(ingest_env):
    env = ingest_env
    for bad_name in ["../etc", "foo/bar", ".", "..", "FOO"]:
        facets = [
            {
                "name": bad_name,
                "files": [
                    {
                        "path": "news/20260305.md",
                        "type": "news",
                        "content": b"x",
                    }
                ],
            }
        ]
        metadata, file_map = _build_request(facets)
        response = _post_facets(
            env["client"], env["key"], env["key_prefix"], metadata, file_map
        )
        assert response.status_code == 400, f"Expected 400 for facet name: {bad_name}"
        assert response.get_json() == {"error": "Invalid facet name"}


def test_new_facet_all_types(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "personal",
            "files": [
                {"path": "facet.json", "type": "facet_json", "content": _json_bytes({"title": "Personal"})},
                {
                    "path": "entities/same_entity/entity.json",
                    "type": "entity_relationship",
                    "content": _json_bytes({"description": "Close contact", "attached_at": 100}),
                },
                {
                    "path": "entities/same_entity/observations.jsonl",
                    "type": "entity_observations",
                    "content": _jsonl_bytes(
                        [{"content": "Likes tea", "observed_at": 1}]
                    ),
                },
                {
                    "path": "entities/20260305.jsonl",
                    "type": "detected_entities",
                    "content": _jsonl_bytes(
                        [{"id": "same_entity", "name": "Same Entity", "type": "Person"}]
                    ),
                },
                {
                    "path": "activities/activities.jsonl",
                    "type": "activity_config",
                    "content": _jsonl_bytes(
                        [{"id": "coding", "name": "Coding", "priority": "high"}]
                    ),
                },
                {
                    "path": "activities/20260305.jsonl",
                    "type": "activity_records",
                    "content": _jsonl_bytes(
                        [
                            {
                                "id": "coding_093000_300",
                                "activity": "coding",
                                "active_entities": ["same_entity"],
                            }
                        ]
                    ),
                },
                {
                    "path": "activities/20260305/coding_093000_300/session_review.md",
                    "type": "activity_output",
                    "content": b"# Session\n",
                },
                {
                    "path": "todos/20260305.jsonl",
                    "type": "todos",
                    "content": _jsonl_bytes([{"text": "Ship it", "created_at": 10}]),
                },
                {
                    "path": "calendar/20260305.jsonl",
                    "type": "calendar",
                    "content": _jsonl_bytes([{"title": "Standup", "start": "09:00"}]),
                },
                {
                    "path": "news/20260305.md",
                    "type": "news",
                    "content": b"# News\n",
                },
                {
                    "path": "logs/20260305.jsonl",
                    "type": "logs",
                    "content": _jsonl_bytes([{"event": "ingested"}]),
                },
            ],
        }
    ]
    metadata, file_map = _build_request(facets)

    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json() == {
        "created": 11,
        "merged": 0,
        "skipped": 0,
        "staged": 0,
        "errors": [],
    }

    facet_root = env["root"] / "facets" / "personal"
    assert _read_json(facet_root / "facet.json") == {"title": "Personal"}
    assert _read_json(facet_root / "entities" / "same_entity" / "entity.json") == {
        "description": "Close contact",
        "attached_at": 100,
        "entity_id": "same_entity",
    }
    assert _read_jsonl_file(
        facet_root / "entities" / "same_entity" / "observations.jsonl"
    ) == [{"content": "Likes tea", "observed_at": 1}]
    assert _read_jsonl_file(facet_root / "entities" / "20260305.jsonl")[0]["id"] == "same_entity"
    assert _read_jsonl_file(facet_root / "activities" / "activities.jsonl")[0]["id"] == "coding"
    assert _read_jsonl_file(facet_root / "activities" / "20260305.jsonl")[0]["id"] == "coding_093000_300"
    assert (
        facet_root / "activities" / "20260305" / "coding_093000_300" / "session_review.md"
    ).read_text(encoding="utf-8") == "# Session\n"
    assert _read_jsonl_file(facet_root / "todos" / "20260305.jsonl")[0]["text"] == "Ship it"
    assert _read_jsonl_file(facet_root / "calendar" / "20260305.jsonl")[0]["title"] == "Standup"
    assert (facet_root / "news" / "20260305.md").read_text(encoding="utf-8") == "# News\n"
    assert _read_jsonl_file(facet_root / "logs" / "20260305.jsonl")[0]["event"] == "ingested"

    source = load_journal_source(env["key"])
    assert source["stats"]["facets_received"] == 1


def test_existing_facet_merge_entity_relationship(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "entities" / "same_entity" / "entity.json"
    _write_json(
        target_path,
        {"entity_id": "same_entity", "description": "Keep target", "attached_at": 200},
    )

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/same_entity/entity.json",
                    "type": "entity_relationship",
                    "content": _json_bytes({"description": "Source desc", "last_seen": 999}),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert _read_json(target_path) == {
        "description": "Keep target",
        "last_seen": 999,
        "attached_at": 200,
        "entity_id": "same_entity",
    }


def test_existing_facet_merge_observations(ingest_env):
    env = ingest_env
    target_path = (
        env["root"] / "facets" / "work" / "entities" / "same_entity" / "observations.jsonl"
    )
    _write_jsonl(
        target_path,
        [
            {"content": "Likes tea", "observed_at": 1},
            {"content": "Prefers email", "observed_at": 2},
        ],
    )

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/same_entity/observations.jsonl",
                    "type": "entity_observations",
                    "content": _jsonl_bytes(
                        [
                            {"content": "Likes tea", "observed_at": 1},
                            {"content": "Uses vim", "observed_at": 3},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert _read_jsonl_file(target_path) == [
        {"content": "Likes tea", "observed_at": 1},
        {"content": "Prefers email", "observed_at": 2},
        {"content": "Uses vim", "observed_at": 3},
    ]


def test_existing_facet_merge_detected_entities(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "entities" / "20260305.jsonl"
    _write_jsonl(target_path, [{"id": "same_entity", "name": "Same", "type": "Person"}])

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/20260305.jsonl",
                    "type": "detected_entities",
                    "content": _jsonl_bytes(
                        [
                            {"id": "same_entity", "name": "Same", "type": "Person"},
                            {"id": "source_entity", "name": "Source", "type": "Person"},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    items = _read_jsonl_file(target_path)
    assert [item["id"] for item in items] == ["same_entity", "target_entity"]


def test_existing_facet_merge_activity_config(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "activities" / "activities.jsonl"
    _write_jsonl(target_path, [{"id": "coding", "name": "Coding"}])

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "activities/activities.jsonl",
                    "type": "activity_config",
                    "content": _jsonl_bytes(
                        [
                            {"id": "coding", "name": "Coding"},
                            {"id": "meeting", "name": "Meeting"},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert [item["id"] for item in _read_jsonl_file(target_path)] == ["coding", "meeting"]


def test_existing_facet_merge_activity_records(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "activities" / "20260305.jsonl"
    _write_jsonl(
        target_path,
        [{"id": "coding_1", "activity": "coding", "active_entities": ["same_entity"]}],
    )

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "activities/20260305.jsonl",
                    "type": "activity_records",
                    "content": _jsonl_bytes(
                        [
                            {"id": "coding_1", "activity": "coding", "active_entities": ["same_entity"]},
                            {"id": "coding_2", "activity": "coding", "active_entities": ["source_entity"]},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    items = _read_jsonl_file(target_path)
    assert [item["id"] for item in items] == ["coding_1", "coding_2"]
    assert items[1]["active_entities"] == ["target_entity"]


def test_existing_facet_merge_activity_output_skip(ingest_env):
    env = ingest_env
    target_file = (
        env["root"]
        / "facets"
        / "work"
        / "activities"
        / "20260305"
        / "coding_093000_300"
        / "session_review.md"
    )
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("existing\n", encoding="utf-8")

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "activities/20260305/coding_093000_300/session_review.md",
                    "type": "activity_output",
                    "content": b"new\n",
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["skipped"] == 1
    assert target_file.read_text(encoding="utf-8") == "existing\n"


def test_existing_facet_merge_activity_output_copy(ingest_env):
    env = ingest_env
    (env["root"] / "facets" / "work").mkdir(parents=True, exist_ok=True)

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "activities/20260305/coding_093000_300/session_review.md",
                    "type": "activity_output",
                    "content": b"copied\n",
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    target_file = (
        env["root"]
        / "facets"
        / "work"
        / "activities"
        / "20260305"
        / "coding_093000_300"
        / "session_review.md"
    )
    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert target_file.read_text(encoding="utf-8") == "copied\n"


def test_existing_facet_merge_todos(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "todos" / "20260305.jsonl"
    _write_jsonl(target_path, [{"text": "Ship it", "created_at": 1}])

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "todos/20260305.jsonl",
                    "type": "todos",
                    "content": _jsonl_bytes(
                        [
                            {"text": "Ship it", "created_at": 1},
                            {"text": "Review PR", "created_at": 2},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert _read_jsonl_file(target_path) == [
        {"text": "Ship it", "created_at": 1},
        {"text": "Review PR", "created_at": 2},
    ]


def test_existing_facet_merge_calendar(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "calendar" / "20260305.jsonl"
    _write_jsonl(target_path, [{"title": "Standup", "start": "09:00"}])

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "calendar/20260305.jsonl",
                    "type": "calendar",
                    "content": _jsonl_bytes(
                        [
                            {"title": "Standup", "start": "09:00"},
                            {"title": "Demo", "start": "14:00"},
                        ]
                    ),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert _read_jsonl_file(target_path) == [
        {"title": "Standup", "start": "09:00"},
        {"title": "Demo", "start": "14:00"},
    ]


def test_existing_facet_merge_news_skip(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "news" / "20260305.md"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("existing\n", encoding="utf-8")

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "news/20260305.md",
                    "type": "news",
                    "content": b"new\n",
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["skipped"] == 1
    assert target_path.read_text(encoding="utf-8") == "existing\n"


def test_existing_facet_merge_news_copy(ingest_env):
    env = ingest_env
    (env["root"] / "facets" / "work").mkdir(parents=True, exist_ok=True)

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "news/20260305.md",
                    "type": "news",
                    "content": b"headline\n",
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    target_path = env["root"] / "facets" / "work" / "news" / "20260305.md"
    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert target_path.read_text(encoding="utf-8") == "headline\n"


def test_existing_facet_merge_logs(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "logs" / "20260305.jsonl"
    _write_jsonl(target_path, [{"event": "existing"}])

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "logs/20260305.jsonl",
                    "type": "logs",
                    "content": _jsonl_bytes([{"event": "new"}]),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["merged"] == 1
    assert _read_jsonl_file(target_path) == [{"event": "existing"}, {"event": "new"}]


def test_entity_id_remapping(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/source_entity/entity.json",
                    "type": "entity_relationship",
                    "content": _json_bytes({"description": "Remapped"}),
                },
                {
                    "path": "entities/source_entity/observations.jsonl",
                    "type": "entity_observations",
                    "content": _jsonl_bytes([{"content": "Knows Rust", "observed_at": 1}]),
                },
                {
                    "path": "entities/20260305.jsonl",
                    "type": "detected_entities",
                    "content": _jsonl_bytes(
                        [{"id": "source_entity", "name": "Source Entity", "type": "Person"}]
                    ),
                },
                {
                    "path": "activities/20260305.jsonl",
                    "type": "activity_records",
                    "content": _jsonl_bytes(
                        [
                            {
                                "id": "coding_1",
                                "activity": "coding",
                                "active_entities": ["source_entity"],
                            }
                        ]
                    ),
                },
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    facet_root = env["root"] / "facets" / "work"
    assert response.status_code == 200
    assert response.get_json()["created"] == 4
    assert (facet_root / "entities" / "target_entity" / "entity.json").exists()
    assert (facet_root / "entities" / "target_entity" / "observations.jsonl").exists()
    assert not (facet_root / "entities" / "source_entity").exists()
    assert _read_jsonl_file(facet_root / "entities" / "20260305.jsonl")[0]["id"] == "target_entity"
    assert _read_jsonl_file(facet_root / "activities" / "20260305.jsonl")[0]["active_entities"] == [
        "target_entity"
    ]


def test_unmapped_entity_staging(ingest_env):
    env = ingest_env
    content = _json_bytes({"description": "Unknown"})
    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/unknown/entity.json",
                    "type": "entity_relationship",
                    "content": content,
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json() == {
        "created": 0,
        "merged": 0,
        "skipped": 0,
        "staged": 1,
        "errors": [],
    }
    staged = _read_staged(env["key_prefix"], "work", "entity_relationship", "entities/unknown/entity.json")
    assert staged["reason"] == "unmapped_entity"
    assert staged["source_entity_id"] == "unknown"
    assert staged["source_path"] == "entities/unknown/entity.json"
    assert "unknown" in staged["explanation"]


def test_staged_then_retry(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/unknown/entity.json",
                    "type": "entity_relationship",
                    "content": _json_bytes({"description": "test"}),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    first = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert first.status_code == 200
    assert first.get_json()["staged"] == 1

    state_path = get_state_directory(env["key_prefix"]) / "entities" / "state.json"
    entity_state = json.loads(state_path.read_text(encoding="utf-8"))
    entity_state["id_map"]["unknown"] = "unknown"
    state_path.write_text(json.dumps(entity_state), encoding="utf-8")

    metadata, file_map = _build_request(facets)
    second = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert second.status_code == 200
    body = second.get_json()
    assert body["staged"] == 0
    assert body["skipped"] == 1
    assert body["created"] == 0
    assert body["merged"] == 0


def test_facet_json_conflict_staging(ingest_env):
    env = ingest_env
    target_path = env["root"] / "facets" / "work" / "facet.json"
    _write_json(target_path, {"title": "Current"})

    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "facet.json",
                    "type": "facet_json",
                    "content": _json_bytes({"title": "Incoming"}),
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert response.get_json()["staged"] == 1
    staged = _read_staged(env["key_prefix"], "work", "facet_json", "facet.json")
    assert staged["reason"] == "facet_json_conflict"
    assert staged["source_content"] == {"title": "Incoming"}
    assert staged["target_content"] == {"title": "Current"}


def test_idempotent(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "work",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": b"repeat\n"}
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    first = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    metadata, file_map = _build_request(facets)
    second = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["created"] == 1
    assert second.get_json() == {
        "created": 0,
        "merged": 0,
        "skipped": 1,
        "staged": 0,
        "errors": [],
    }


def test_error_isolation(ingest_env):
    env = ingest_env
    (env["root"] / "facets" / "work").mkdir(parents=True, exist_ok=True)
    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "entities/20260305.jsonl",
                    "type": "detected_entities",
                    "content": b"{bad json\n",
                },
                {"path": "news/20260305.md", "type": "news", "content": b"good\n"},
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    body = response.get_json()
    assert body["merged"] == 1
    assert len(body["errors"]) == 1
    assert (env["root"] / "facets" / "work" / "news" / "20260305.md").exists()


def test_error_isolation_across_facets(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "broken",
            "files": [
                {
                    "path": "activities/activities.jsonl",
                    "type": "activity_config",
                    "content": b"{bad json\n",
                }
            ],
        },
        {
            "name": "good",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": b"ok\n"}
            ],
        },
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    body = response.get_json()
    assert body["created"] == 1
    assert len(body["errors"]) == 1
    assert (env["root"] / "facets" / "good" / "news" / "20260305.md").exists()


def test_stats_update(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "alpha",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": b"alpha\n"},
                {
                    "path": "logs/20260305.jsonl",
                    "type": "logs",
                    "content": _jsonl_bytes([{"event": "alpha"}]),
                },
            ],
        },
        {
            "name": "beta",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": b"beta\n"}
            ],
        },
        {
            "name": "gamma",
            "files": [
                {
                    "path": "entities/unknown/entity.json",
                    "type": "entity_relationship",
                    "content": _json_bytes({"description": "unknown"}),
                }
            ],
        },
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)
    source = load_journal_source(env["key"])

    assert response.status_code == 200
    assert source["stats"]["facets_received"] == 2


def test_state_manifest(ingest_env):
    env = ingest_env
    news = b"news\n"
    logs = _jsonl_bytes([{"event": "log"}])
    facets = [
        {
            "name": "work",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": news},
                {"path": "logs/20260305.jsonl", "type": "logs", "content": logs},
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert _read_state(env["key_prefix"]) == {
        "received": {
            "work/news/20260305.md": _hash_bytes(news),
            "work/logs/20260305.jsonl": _hash_bytes(logs),
        }
    }


def test_decision_log(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "work",
            "files": [
                {"path": "news/20260305.md", "type": "news", "content": b"headline\n"}
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    entries = _read_log(env["key_prefix"])
    assert len(entries) == 1
    entry = entries[0]
    assert "ts" in entry
    assert entry["action"] == "facet_file_created"
    assert entry["item_type"] == "news"
    assert entry["item_id"] == "work/news/20260305.md"
    assert entry["facet"] == "work"
    assert entry["reason"] == "new_facet"


def test_error_logged_to_decision_log(ingest_env):
    env = ingest_env
    facets = [
        {
            "name": "work",
            "files": [
                {
                    "path": "todos/20260305.jsonl",
                    "type": "todos",
                    "content": b"{bad\n",
                }
            ],
        }
    ]
    metadata, file_map = _build_request(facets)
    response = _post_facets(env["client"], env["key"], env["key_prefix"], metadata, file_map)

    assert response.status_code == 200
    assert len(response.get_json()["errors"]) == 1

    entries = _read_log(env["key_prefix"])
    assert len(entries) == 1
    assert entries[0]["action"] == "facet_file_error"
    assert entries[0]["facet"] == "work"
