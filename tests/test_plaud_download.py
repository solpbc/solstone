# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import logging
from unittest.mock import Mock

import pytest
import requests


class _Response:
    status_code = 200
    headers = {"Content-Length": "0"}
    text = ""

    def __init__(self, chunks=None, exc=None):
        self._chunks = chunks or []
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size):
        if self._exc:
            raise self._exc
        yield from self._chunks


@pytest.mark.timeout(5)
def test_download_to_file_returns_false_on_read_timeout(tmp_path, caplog):
    from think.importers.plaud import download_to_file

    session = Mock()
    session.get.return_value = _Response(exc=requests.exceptions.ReadTimeout("stalled"))
    dest_path = tmp_path / "recording.opus"

    caplog.set_level(logging.WARNING)

    assert download_to_file(session, "https://example.test/file", dest_path) is False
    assert not dest_path.exists()
    assert "Plaud download for recording.opus failed: stalled" in caplog.text


def test_download_to_file_calls_progress_cb_throttled(tmp_path, monkeypatch):
    from think.importers import plaud

    session = Mock()
    session.get.return_value = _Response(chunks=[b"x"] * 12)
    progress_cb = Mock()
    ticks = iter(range(13))
    monkeypatch.setattr(plaud.time, "monotonic", lambda: next(ticks))

    dest_path = tmp_path / "recording.opus"

    assert (
        plaud.download_to_file(
            session,
            "https://example.test/file",
            dest_path,
            progress_cb=progress_cb,
        )
        is True
    )
    assert dest_path.read_bytes() == b"x" * 12
    assert progress_cb.call_count == 2


def test_sync_inactivity_timer_trips_when_progress_stops(tmp_path, monkeypatch, caplog):
    from think.importers import plaud

    files = [
        {
            "id": "file1",
            "filename": "One",
            "fullname": "one.opus",
            "filesize": 10,
            "start_time": 1737000000000,
            "duration": 60000,
        },
        {
            "id": "file2",
            "filename": "Two",
            "fullname": "two.opus",
            "filesize": 10,
            "start_time": 1737000300000,
            "duration": 60000,
        },
    ]

    monkeypatch.setenv("PLAUD_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(plaud, "SYNC_INACTIVITY_TIMEOUT", 1)
    monkeypatch.setattr(plaud, "list_files", lambda _session, _token: files)
    monkeypatch.setattr(
        plaud, "get_temp_url", lambda *_args: "https://example.test/file"
    )
    monkeypatch.setattr(plaud, "download_to_file", lambda *_args, **_kwargs: False)
    ticks = iter([0.0, 0.5, 2.0])
    monkeypatch.setattr(plaud.time, "monotonic", lambda: next(ticks))

    caplog.set_level(logging.WARNING)

    result = plaud.PlaudBackend().sync(tmp_path, dry_run=False)

    assert any("Sync stalled" in error for error in result["errors"])
    assert "Sync stalled" in caplog.text
