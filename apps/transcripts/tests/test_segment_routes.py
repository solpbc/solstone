# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest

FIXTURE_DAY = "20260304"
FIXTURE_STREAM = "default"
FIXTURE_SEGMENT = "090000_300"


@pytest.mark.parametrize("stream", ["-bad", "Upper", "..bad"])
def test_segment_content_rejects_invalid_stream(client, stream):
    response = client.get(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{stream}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Invalid stream format"}


@pytest.mark.parametrize("stream", ["-bad", "Upper", "..bad"])
def test_delete_segment_rejects_invalid_stream(client, stream):
    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{stream}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid stream format"}


def test_segment_content_missing_segment_does_not_create_phantom_directory(
    client, journal_copy
):
    response = client.get("/app/transcripts/api/segment/29990101/default/090000_300")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Segment directory not found"}
    assert not (journal_copy / "chronicle" / "29990101").exists()
    assert not (
        journal_copy / "chronicle" / "29990101" / "default" / "090000_300"
    ).exists()


def test_delete_missing_segment_does_not_create_phantom_directory(client, journal_copy):
    response = client.delete("/app/transcripts/api/segment/29990101/default/090000_300")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Segment not found"}
    assert not (journal_copy / "chronicle" / "29990101").exists()
    assert not (
        journal_copy / "chronicle" / "29990101" / "default" / "090000_300"
    ).exists()


def test_segment_content_happy_path_returns_segment_payload(client):
    response = client.get(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["segment_key"] == FIXTURE_SEGMENT
    assert data["chunks"]
    assert "media_sizes" in data


def test_delete_segment_happy_path_removes_segment_directory(
    client, journal_copy, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)
    segment_dir = (
        journal_copy / "chronicle" / FIXTURE_DAY / FIXTURE_STREAM / FIXTURE_SEGMENT
    )

    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    assert response.get_json() == {"success": True, "deleted": FIXTURE_SEGMENT}
    assert not segment_dir.exists()


def test_delete_segment_includes_search_index_warning_when_supervisor_is_down(client):
    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "success": True,
        "deleted": FIXTURE_SEGMENT,
        "search_index_warning": True,
    }


def test_delete_segment_omits_search_index_warning_when_supervisor_is_up(
    client, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)

    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    assert response.get_json() == {"success": True, "deleted": FIXTURE_SEGMENT}
