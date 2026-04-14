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
    first_dir = tmp_path / "20260413" / "laptop" / "143022_300"
    second_dir = tmp_path / "20260413" / "laptop" / "150000_600"
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


def _make_session(*, manifest_data=None, get_status=200, post_status=200, post_json=None):
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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

        assert mock_session.post.call_count == 1
        metadata = json.loads(mock_session.post.call_args.kwargs["data"]["metadata"])
        assert metadata["segments"][0]["segment_key"] == "150000_600"

    def test_dry_run_output(self, tmp_path, monkeypatch, capsys):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        mock_session = _make_session(manifest_data={})

        with patch("observe.export.requests.Session", return_value=mock_session):
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=True)

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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=True)

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "20260413: 1 segment(s)" in output
        assert "Dry run: would send 1, skip 1" in output

    def test_retry_on_5xx(self, tmp_path, monkeypatch):
        from observe.export import export_segments

        journal = _setup_journal(tmp_path)
        _set_journal_override(monkeypatch, journal)

        second_dir = journal / "20260413" / "laptop" / "150000_600"
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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

        assert mock_session.post.call_count == 0
        assert "up to date" in capsys.readouterr().out

    def test_only_not_implemented(self, capsys):
        from observe.export import main

        mock_args = MagicMock()
        mock_args.to = "host"
        mock_args.key = "testkey"
        mock_args.only = "facets"
        mock_args.dry_run = False
        mock_args.day = None

        with (
            patch("sys.argv", ["sol export", "--to", "host", "--key", "testkey", "--only", "facets"]),
            patch("observe.export.setup_cli", return_value=mock_args),
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 0
        assert "not yet implemented" in capsys.readouterr().out

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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

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
            export_segments("https://example.com", "test-key", ["20260413"], dry_run=False)

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
