# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import think.utils
from observe.utils import compute_bytes_sha256


def _setup_journal(tmp_path, *, include_stream_json: bool = False):
    first_dir = tmp_path / "chronicle" / "20260413" / "laptop" / "143022_300"
    second_dir = tmp_path / "chronicle" / "20260413" / "laptop" / "150000_600"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    (first_dir / "audio.flac").write_bytes(b"audio-data-one")
    (first_dir / "transcript.jsonl").write_bytes(b"transcript-one")
    (second_dir / "audio.flac").write_bytes(b"audio-data-two")
    (second_dir / "transcript.jsonl").write_bytes(b"transcript-two")

    if include_stream_json:
        (first_dir / "stream.json").write_bytes(b'{"name": "laptop"}')
        (second_dir / "stream.json").write_bytes(b'{"name": "laptop"}')

    return tmp_path


def _set_journal_override(monkeypatch, journal_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal_path))
    think.utils._journal_path_cache = None


def _make_session(
    *, manifest_data=None, get_status=200, post_status=200, post_json=None
):
    mock = MagicMock(spec=requests.Session)
    mock.headers = {}

    get_response = MagicMock()
    get_response.status_code = get_status
    get_response.json.return_value = manifest_data if manifest_data is not None else {}
    get_response.text = "GET error"
    mock.get.return_value = get_response

    post_response = MagicMock()
    post_response.status_code = post_status
    post_response.json.return_value = (
        post_json
        if post_json is not None
        else {
            "segments_received": 1,
            "segments_skipped": 0,
            "segments_deconflicted": 0,
            "errors": [],
        }
    )
    post_response.text = "POST error"
    mock.post.return_value = post_response

    return mock


def _setup_entities(tmp_path):
    """Create test entities in journal fixture."""
    entities_dir = tmp_path / "entities"

    alice_dir = entities_dir / "alice_johnson"
    alice_dir.mkdir(parents=True)
    alice = {
        "id": "alice_johnson",
        "name": "Alice Johnson",
        "type": "Person",
        "created_at": 1000,
    }
    (alice_dir / "entity.json").write_text(json.dumps(alice), encoding="utf-8")

    bob_dir = entities_dir / "bob_smith"
    bob_dir.mkdir(parents=True)
    bob = {"id": "bob_smith", "name": "Bob Smith", "type": "Person", "created_at": 2000}
    (bob_dir / "entity.json").write_text(json.dumps(bob), encoding="utf-8")

    blocked_dir = entities_dir / "blocked_user"
    blocked_dir.mkdir(parents=True)
    blocked = {
        "id": "blocked_user",
        "name": "Blocked",
        "type": "Person",
        "blocked": True,
        "created_at": 3000,
    }
    (blocked_dir / "entity.json").write_text(json.dumps(blocked), encoding="utf-8")

    return {"alice_johnson": alice, "bob_smith": bob, "blocked_user": blocked}


def _entity_hash(entity: dict) -> str:
    return hashlib.sha256(
        json.dumps(entity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _setup_facets(tmp_path):
    """Create test facets in journal fixture."""
    facets_dir = tmp_path / "facets"

    work_dir = facets_dir / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "facet.json").write_text('{"title": "Work"}', encoding="utf-8")

    ent_dir = work_dir / "entities" / "alice"
    ent_dir.mkdir(parents=True)
    (ent_dir / "entity.json").write_text('{"id": "alice"}', encoding="utf-8")
    (ent_dir / "observations.jsonl").write_text('{"text": "obs1"}\n', encoding="utf-8")

    det_dir = work_dir / "entities"
    (det_dir / "20260413.jsonl").write_text('{"entity": "test"}\n', encoding="utf-8")

    todo_dir = work_dir / "todos"
    todo_dir.mkdir(parents=True)
    (todo_dir / "20260413.jsonl").write_text('{"task": "do stuff"}\n', encoding="utf-8")

    personal_dir = facets_dir / "personal"
    personal_dir.mkdir(parents=True)
    (personal_dir / "facet.json").write_text('{"title": "Personal"}', encoding="utf-8")

    news_dir = personal_dir / "news"
    news_dir.mkdir(parents=True)
    (news_dir / "20260413.md").write_text("# News\nHello world\n", encoding="utf-8")

    return facets_dir


def _facet_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_imports(tmp_path):
    """Create test import metadata dirs in journal fixture."""
    imports_dir = tmp_path / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)

    dir1 = imports_dir / "20260101_090000"
    dir1.mkdir()
    import_json_1 = {"original_filename": "cal.zip", "file_size": 100}
    imported_json_1 = {
        "processed_timestamp": "20260101_090000",
        "total_files_created": 1,
    }
    manifest_1 = [{"id": "event-0", "title": "Test Event"}]
    (dir1 / "import.json").write_text(json.dumps(import_json_1), encoding="utf-8")
    (dir1 / "imported.json").write_text(json.dumps(imported_json_1), encoding="utf-8")
    (dir1 / "content_manifest.jsonl").write_text(
        json.dumps(manifest_1[0]) + "\n", encoding="utf-8"
    )

    dir2 = imports_dir / "20260102_100000"
    dir2.mkdir()
    import_json_2 = {"original_filename": "chat.zip", "file_size": 200}
    imported_json_2 = {
        "processed_timestamp": "20260102_100000",
        "total_files_created": 2,
    }
    manifest_2 = [{"id": "conv-0", "title": "Test Convo"}]
    (dir2 / "import.json").write_text(json.dumps(import_json_2), encoding="utf-8")
    (dir2 / "imported.json").write_text(json.dumps(imported_json_2), encoding="utf-8")
    (dir2 / "content_manifest.jsonl").write_text(
        json.dumps(manifest_2[0]) + "\n", encoding="utf-8"
    )

    (imports_dir / "plaud.json").write_text('{"last_sync": 123}', encoding="utf-8")

    source_dir = imports_dir / "abcd1234"
    source_dir.mkdir()
    (source_dir / "segments").mkdir()

    return {
        "20260101_090000": {
            "import_json": import_json_1,
            "imported_json": imported_json_1,
            "content_manifest": manifest_1,
        },
        "20260102_100000": {
            "import_json": import_json_2,
            "imported_json": imported_json_2,
            "content_manifest": manifest_2,
        },
    }


def _import_hash(import_data):
    """Compute hash matching export_imports algorithm."""
    hash_input = json.dumps(
        {
            "import_json": import_data["import_json"],
            "imported_json": import_data["imported_json"],
            "content_manifest": import_data["content_manifest"],
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(hash_input).hexdigest()


def _setup_config(tmp_path):
    """Create test config in journal fixture."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "identity": {"name": "Test", "preferred": "Tester", "timezone": "UTC"},
        "convey": {
            "allow_network_access": False,
            "password_hash": "secret_hash",
            "secret": "secret_val",
            "trust_localhost": True,
        },
        "setup": {"completed_at": 12345},
        "env": {"KEY": "val"},
        "retention": {"days": 30},
    }
    (config_dir / "journal.json").write_text(json.dumps(config), encoding="utf-8")
    return config


class TestExportSegments:
    def test_manifest_query_and_delta(self, tmp_path, monkeypatch):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        manifest_data = {
            "20260413": {
                "laptop/143022_300": {
                    "files": [
                        {
                            "name": "audio.flac",
                            "sha256": compute_bytes_sha256(b"audio-data-one"),
                            "size": len(b"audio-data-one"),
                        },
                        {
                            "name": "transcript.jsonl",
                            "sha256": compute_bytes_sha256(b"transcript-one"),
                            "size": len(b"transcript-one"),
                        },
                    ]
                }
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        assert mock_session.post.call_count == 1
        metadata = json.loads(mock_session.post.call_args.kwargs["data"]["metadata"])
        assert metadata["segments"][0]["segment_key"] == "150000_600"

    def test_dry_run_output(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        mock_session = _make_session(manifest_data={})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=True
            )

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "20260413: 2 segment(s)" in output
        assert "Dry run: would send 2, skip 0" in output

    def test_dry_run_skipped_not_double_counted(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        manifest_data = {
            "20260413": {
                "laptop/143022_300": {
                    "files": [
                        {
                            "name": "audio.flac",
                            "sha256": compute_bytes_sha256(b"audio-data-one"),
                            "size": len(b"audio-data-one"),
                        },
                        {
                            "name": "transcript.jsonl",
                            "sha256": compute_bytes_sha256(b"transcript-one"),
                            "size": len(b"transcript-one"),
                        },
                    ]
                }
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=True
            )

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "20260413: 1 segment(s)" in output
        assert "Dry run: would send 1, skip 1" in output

    def test_retry_on_5xx(self, tmp_path, monkeypatch):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        second_dir = journal / "chronicle" / "20260413" / "laptop" / "150000_600"
        for file_path in second_dir.iterdir():
            file_path.unlink()
        second_dir.rmdir()

        mock_session = _make_session(manifest_data={})
        first = MagicMock(status_code=500, text="server error")
        second = MagicMock(status_code=500, text="server error")
        success = MagicMock(status_code=200, text="ok")
        success.json.return_value = {
            "segments_received": 1,
            "segments_skipped": 0,
            "segments_deconflicted": 0,
            "errors": [],
        }
        mock_session.post.side_effect = [first, second, success]

        with (
            patch("observe.export.requests.Session", return_value=mock_session),
            patch("observe.export.time.sleep") as mock_sleep,
        ):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        assert mock_session.post.call_count == 3
        assert mock_sleep.called

    def test_auth_error_401(self):
        from observe.export import _query_manifest

        mock_session = _make_session(get_status=401)

        with pytest.raises(ValueError, match="Authentication failed"):
            _query_manifest(mock_session, "https://example.com", "test-key")

    def test_auth_error_403(self):
        from observe.export import _query_manifest

        mock_session = _make_session(get_status=403)

        with pytest.raises(ValueError, match="journal source revoked"):
            _query_manifest(mock_session, "https://example.com", "test-key")

    def test_connection_error(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        mock_session = _make_session(manifest_data={})
        mock_session.get.side_effect = requests.ConnectionError

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        assert mock_session.post.call_count == 0
        assert "Connection failed" in capsys.readouterr().out

    def test_idempotent(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        manifest_data = {
            "20260413": {
                "laptop/143022_300": {
                    "files": [
                        {
                            "name": "audio.flac",
                            "sha256": compute_bytes_sha256(b"audio-data-one"),
                            "size": len(b"audio-data-one"),
                        },
                        {
                            "name": "transcript.jsonl",
                            "sha256": compute_bytes_sha256(b"transcript-one"),
                            "size": len(b"transcript-one"),
                        },
                    ]
                },
                "laptop/150000_600": {
                    "files": [
                        {
                            "name": "audio.flac",
                            "sha256": compute_bytes_sha256(b"audio-data-two"),
                            "size": len(b"audio-data-two"),
                        },
                        {
                            "name": "transcript.jsonl",
                            "sha256": compute_bytes_sha256(b"transcript-two"),
                            "size": len(b"transcript-two"),
                        },
                    ]
                },
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        assert mock_session.post.call_count == 0
        assert "up to date" in capsys.readouterr().out

    def test_upload_error_isolation(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        mock_session = _make_session(manifest_data={})
        first = MagicMock(status_code=400, text="bad request")
        second = MagicMock(status_code=200, text="ok")
        second.json.return_value = {
            "segments_received": 1,
            "segments_skipped": 0,
            "segments_deconflicted": 0,
            "errors": [],
        }
        mock_session.post.side_effect = [first, second]

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        assert mock_session.post.call_count == 2
        output = capsys.readouterr().out
        assert "1 sent" in output
        assert "1 failed" in output

    def test_stream_json_excluded(self, tmp_path, monkeypatch):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path, include_stream_json=True)
        _set_journal_override(monkeypatch, journal)

        mock_session = _make_session(manifest_data={})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments(
                "https://example.com", "test-key", ["20260413"], dry_run=False
            )

        for call in mock_session.post.call_args_list:
            post_kwargs = call.kwargs
            metadata = json.loads(post_kwargs["data"]["metadata"])
            assert "stream.json" not in metadata["segments"][0]["files"]
            uploaded_names = [entry[1][0] for entry in post_kwargs["files"]]
            assert "stream.json" not in uploaded_names


class TestExportEntities:
    def test_manifest_delta(self, tmp_path, monkeypatch):
        from observe.export import export_entities

        entities = _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {
                "alice_johnson": _entity_hash(entities["alice_johnson"]),
            }
        }
        post_json = {
            "created": 1,
            "auto_merged": 0,
            "staged": 0,
            "skipped": 0,
            "errors": [],
        }
        mock_session = _make_session(manifest_data=manifest_data, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1
        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_entities = posted_data["entities"]
        posted_ids = [e["id"] for e in posted_entities]
        assert "bob_smith" in posted_ids
        assert "alice_johnson" not in posted_ids
        assert "blocked_user" not in posted_ids

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_entities

        entities = _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {
                "alice_johnson": _entity_hash(entities["alice_johnson"]),
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=True)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "1 new" in output
        assert "1 unchanged" in output

    def test_idempotent(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_entities

        entities = _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {
                "alice_johnson": _entity_hash(entities["alice_johnson"]),
                "bob_smith": _entity_hash(entities["bob_smith"]),
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "up to date" in output

    def test_changed_entity(self, tmp_path, monkeypatch):
        from observe.export import export_entities

        entities = _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {
                "alice_johnson": "stale_hash_that_does_not_match",
                "bob_smith": _entity_hash(entities["bob_smith"]),
            }
        }
        post_json = {
            "created": 0,
            "auto_merged": 1,
            "staged": 0,
            "skipped": 0,
            "errors": [],
        }
        mock_session = _make_session(manifest_data=manifest_data, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1
        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_ids = [e["id"] for e in posted_data["entities"]]
        assert "alice_johnson" in posted_ids
        assert "bob_smith" not in posted_ids

    def test_auth_error_401(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_entities

        _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(get_status=401)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        assert "Authentication failed" in capsys.readouterr().out

    def test_connection_error(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_entities

        _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session()
        mock_session.get.side_effect = requests.ConnectionError

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        assert "Connection failed" in capsys.readouterr().out

    def test_empty_manifest(self, tmp_path, monkeypatch):
        from observe.export import export_entities

        _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {
            "created": 2,
            "auto_merged": 0,
            "staged": 0,
            "skipped": 0,
            "errors": [],
        }
        mock_session = _make_session(manifest_data={}, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1
        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_ids = [e["id"] for e in posted_data["entities"]]
        assert "alice_johnson" in posted_ids
        assert "bob_smith" in posted_ids
        assert "blocked_user" not in posted_ids

    def test_response_errors_reported(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_entities

        _setup_entities(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {
            "created": 1,
            "auto_merged": 0,
            "staged": 0,
            "skipped": 0,
            "errors": ["Entity bob_smith: invalid type"],
        }
        mock_session = _make_session(manifest_data={}, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_entities("https://example.com", "test-key", dry_run=False)

        output = capsys.readouterr().out
        assert "Entity bob_smith: invalid type" in output
        assert "1 error" in output


class TestExportFacets:
    def test_full_export(self, tmp_path, monkeypatch):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {"created": 1, "merged": 0, "skipped": 0, "staged": 0, "errors": []}
        mock_session = _make_session(
            manifest_data={"received": {}}, post_json=post_json
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 2

        calls_by_facet = {}
        for call in mock_session.post.call_args_list:
            metadata = json.loads(call.kwargs["data"]["metadata"])
            facet = metadata["facets"][0]["name"]
            calls_by_facet[facet] = call.kwargs

        assert set(calls_by_facet) == {"personal", "work"}

        personal_files = calls_by_facet["personal"]["files"]
        assert [entry[0] for entry in personal_files] == ["files_0_0", "files_0_1"]
        personal_metadata = json.loads(calls_by_facet["personal"]["data"]["metadata"])
        assert personal_metadata == {
            "facets": [
                {
                    "name": "personal",
                    "files": [
                        {"path": "facet.json", "type": "facet_json"},
                        {"path": "news/20260413.md", "type": "news"},
                    ],
                }
            ]
        }

        work_files = calls_by_facet["work"]["files"]
        assert [entry[0] for entry in work_files] == [
            "files_0_0",
            "files_0_1",
            "files_0_2",
            "files_0_3",
            "files_0_4",
        ]
        work_metadata = json.loads(calls_by_facet["work"]["data"]["metadata"])
        assert work_metadata == {
            "facets": [
                {
                    "name": "work",
                    "files": [
                        {
                            "path": "entities/20260413.jsonl",
                            "type": "detected_entities",
                        },
                        {
                            "path": "entities/alice/entity.json",
                            "type": "entity_relationship",
                        },
                        {
                            "path": "entities/alice/observations.jsonl",
                            "type": "entity_observations",
                        },
                        {"path": "facet.json", "type": "facet_json"},
                        {"path": "todos/20260413.jsonl", "type": "todos"},
                    ],
                }
            ]
        }

    def test_delta_mixed(self, tmp_path, monkeypatch):
        from observe.export import export_facets

        facets_dir = _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        work_dir = facets_dir / "work"
        personal_dir = facets_dir / "personal"
        manifest_data = {
            "received": {
                "work/facet.json": _facet_file_hash(work_dir / "facet.json"),
                "work/entities/alice/entity.json": _facet_file_hash(
                    work_dir / "entities" / "alice" / "entity.json"
                ),
                "work/entities/alice/observations.jsonl": "stale-hash",
                "work/todos/20260413.jsonl": "stale-hash",
                "personal/facet.json": _facet_file_hash(personal_dir / "facet.json"),
                "personal/news/20260413.md": _facet_file_hash(
                    personal_dir / "news" / "20260413.md"
                ),
            }
        }
        post_json = {"created": 0, "merged": 1, "skipped": 0, "staged": 0, "errors": []}
        mock_session = _make_session(manifest_data=manifest_data, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1
        metadata = json.loads(mock_session.post.call_args.kwargs["data"]["metadata"])
        assert metadata["facets"][0]["name"] == "work"
        assert metadata["facets"][0]["files"] == [
            {"path": "entities/20260413.jsonl", "type": "detected_entities"},
            {
                "path": "entities/alice/observations.jsonl",
                "type": "entity_observations",
            },
            {"path": "todos/20260413.jsonl", "type": "todos"},
        ]

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(manifest_data={"received": {}})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=True)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "personal: 2 new, 0 changed, 0 unchanged" in output
        assert "work: 5 new, 0 changed, 0 unchanged" in output
        assert (
            "Dry run: 7 new files, 0 changed, 0 unchanged across 2 facet(s)" in output
        )

    def test_idempotent(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_facets

        facets_dir = _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_received = {}
        for facet_dir in sorted((facets_dir).iterdir()):
            if not facet_dir.is_dir():
                continue
            for file_path in sorted(facet_dir.rglob("*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(facet_dir).as_posix()
                    manifest_received[f"{facet_dir.name}/{rel_path}"] = (
                        _facet_file_hash(file_path)
                    )
        mock_session = _make_session(manifest_data={"received": manifest_received})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        assert "up to date" in capsys.readouterr().out

    def test_error_isolation(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(manifest_data={"received": {}})
        first = MagicMock(status_code=400, text="bad request")
        second = MagicMock(status_code=200, text="ok")
        second.json.return_value = {
            "created": 1,
            "merged": 0,
            "skipped": 0,
            "staged": 0,
            "errors": [],
        }
        mock_session.post.side_effect = [first, second]

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 2
        output = capsys.readouterr().out
        assert "1 sent" in output
        assert "1 failed" in output

    def test_new_facet_vs_changed(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_facets

        facets_dir = _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        work_dir = facets_dir / "work"
        manifest_data = {
            "received": {
                f"work/{file_path.relative_to(work_dir).as_posix()}": "stale-hash"
                for file_path in sorted(work_dir.rglob("*"))
                if file_path.is_file()
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=True)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "personal: 2 new, 0 changed, 0 unchanged" in output
        assert "work: 0 new, 5 changed, 0 unchanged" in output

    def test_skips_invalid_facet_names(self, tmp_path, monkeypatch):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        bad_dir = tmp_path / "facets" / "BadName"
        bad_dir.mkdir(parents=True)
        (bad_dir / "facet.json").write_text('{"title": "Bad"}', encoding="utf-8")
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {"created": 1, "merged": 0, "skipped": 0, "staged": 0, "errors": []}
        mock_session = _make_session(
            manifest_data={"received": {}}, post_json=post_json
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 2
        posted_facets = {
            json.loads(call.kwargs["data"]["metadata"])["facets"][0]["name"]
            for call in mock_session.post.call_args_list
        }
        assert posted_facets == {"personal", "work"}

    def test_skips_events_directory(self, tmp_path, monkeypatch):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        events_dir = tmp_path / "facets" / "work" / "events"
        events_dir.mkdir(parents=True)
        (events_dir / "20260413.jsonl").write_text(
            '{"event": "ignored"}\n', encoding="utf-8"
        )
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {"created": 1, "merged": 0, "skipped": 0, "staged": 0, "errors": []}
        mock_session = _make_session(
            manifest_data={"received": {}}, post_json=post_json
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        calls_by_facet = {
            json.loads(call.kwargs["data"]["metadata"])["facets"][0][
                "name"
            ]: call.kwargs
            for call in mock_session.post.call_args_list
        }
        work_metadata = json.loads(calls_by_facet["work"]["data"]["metadata"])
        uploaded_paths = [
            entry["path"] for entry in work_metadata["facets"][0]["files"]
        ]
        assert "events/20260413.jsonl" not in uploaded_paths

    def test_response_errors_reported(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_facets

        _setup_facets(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        post_json = {
            "created": 1,
            "merged": 0,
            "skipped": 0,
            "staged": 0,
            "errors": [{"facet": "work", "error": "entity merge conflict"}],
        }
        mock_session = _make_session(
            manifest_data={"received": {}}, post_json=post_json
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_facets("https://example.com", "test-key", dry_run=False)

        output = capsys.readouterr().out
        assert "entity merge conflict" in output
        assert "error" in output.lower()


class TestExportImports:
    def test_manifest_delta(self, tmp_path, monkeypatch):
        from observe.export import export_imports

        imports = _setup_imports(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {"20260101_090000": _import_hash(imports["20260101_090000"])}
        }
        post_json = {"copied": 1, "staged": 0, "skipped": 0, "errors": []}
        mock_session = _make_session(manifest_data=manifest_data, post_json=post_json)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_imports("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1
        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_ids = [entry["id"] for entry in posted_data["imports"]]
        assert posted_ids == ["20260102_100000"]

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_imports

        _setup_imports(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(manifest_data={"received": {}})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_imports("https://example.com", "test-key", dry_run=True)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "2 new" in output
        assert "0 changed" in output

    def test_idempotent(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_imports

        imports = _setup_imports(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        manifest_data = {
            "received": {
                import_id: _import_hash(import_data)
                for import_id, import_data in imports.items()
            }
        }
        mock_session = _make_session(manifest_data=manifest_data)

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_imports("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        assert "up to date" in capsys.readouterr().out

    def test_sync_state_excluded(self, tmp_path, monkeypatch):
        from observe.export import export_imports

        _setup_imports(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(
            manifest_data={"received": {}},
            post_json={"copied": 2, "staged": 0, "skipped": 0, "errors": []},
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_imports("https://example.com", "test-key", dry_run=False)

        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_ids = {entry["id"] for entry in posted_data["imports"]}
        assert "plaud.json" not in posted_ids

    def test_source_dir_excluded(self, tmp_path, monkeypatch):
        from observe.export import export_imports

        _setup_imports(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(
            manifest_data={"received": {}},
            post_json={"copied": 2, "staged": 0, "skipped": 0, "errors": []},
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_imports("https://example.com", "test-key", dry_run=False)

        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_ids = {entry["id"] for entry in posted_data["imports"]}
        assert "abcd1234" not in posted_ids


class TestExportConfig:
    def test_config_export(self, tmp_path, monkeypatch):
        from observe.export import export_config

        _setup_config(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(
            manifest_data={},
            post_json={"staged": True, "skipped": False, "diff_fields": 3},
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_config("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 1

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_config

        _setup_config(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(manifest_data={})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_config("https://example.com", "test-key", dry_run=True)

        assert mock_session.post.call_count == 0
        assert "would send snapshot" in capsys.readouterr().out

    def test_idempotent(self, tmp_path, monkeypatch, capsys):
        from observe.export import _strip_never_transfer, export_config

        config = _setup_config(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        stripped = _strip_never_transfer(config)
        content_hash = hashlib.sha256(
            json.dumps(stripped, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        mock_session = _make_session(manifest_data={"last_hash": content_hash})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_config("https://example.com", "test-key", dry_run=False)

        assert mock_session.post.call_count == 0
        assert "up to date" in capsys.readouterr().out

    def test_never_transfer_stripped(self, tmp_path, monkeypatch):
        from observe.export import export_config

        _setup_config(tmp_path)
        _set_journal_override(monkeypatch, tmp_path)

        mock_session = _make_session(
            manifest_data={},
            post_json={"staged": True, "skipped": False, "diff_fields": 3},
        )

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_config("https://example.com", "test-key", dry_run=False)

        posted_data = mock_session.post.call_args.kwargs.get(
            "json"
        ) or mock_session.post.call_args[1].get("json")
        posted_config = posted_data["config"]
        assert posted_config["convey"] == {
            "allow_network_access": False,
            "trust_localhost": True,
        }
        assert posted_config["setup"] == {"completed_at": 12345}
        assert posted_config["env"] == {"KEY": "val"}
