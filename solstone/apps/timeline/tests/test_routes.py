# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import os
import time

from solstone.apps.timeline import routes

DAY = "20260510"
MONTH = "202605"


def test_workspace_root_renders(client):
    response = client.get("/app/timeline/", follow_redirects=True)

    assert response.status_code == 200
    assert b'id="timeline-shell"' in response.data
    assert b"/app/timeline/static/timeline.css" in response.data
    assert b"/app/timeline/static/data-mock.js" in response.data
    assert b"/app/timeline/static/timeline.js" in response.data
    assert b"defer" in response.data


def test_index_shape_and_size(client):
    response = client.get("/app/timeline/api/index")

    assert response.status_code == 200
    assert len(response.data) < 20 * 1024
    payload = response.get_json()
    assert set(payload) == {"now", "today", "months", "year_top"}
    assert len(payload["months"]) == 12
    month = next(m for m in payload["months"] if m["ym"] == MONTH)
    assert month["day_count"] == 1
    assert month["days_with_data"] == [DAY]
    assert month["month_top"][0]["title"] == "Timeline Port"
    assert "days" not in month
    assert [item["month"] for item in payload["year_top"]] == ["202604", "202605"]


def test_month_known_shape(client):
    response = client.get(f"/app/timeline/api/month/{MONTH}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ym"] == MONTH
    assert payload["day_count"] == 1
    assert payload["days_with_data"] == [DAY]
    assert payload["days"][DAY] == {
        "day": DAY,
        "day_top": [
            {
                "title": "Timeline Port",
                "description": "Reviewed the timeline app port.",
                "origin": "20260510/100000_300",
            }
        ],
        "day_rationale": "Fixture day for timeline route tests.",
    }
    assert "hours" not in payload["days"][DAY]
    assert "hours_avail" not in payload["days"][DAY]


def test_month_unknown_returns_404(client):
    response = client.get("/app/timeline/api/month/202501")

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["reason_code"] == "timeline_month_not_found"
    assert payload["detail"] == "no data for 202501"


def test_month_bad_input_returns_400(client):
    response = client.get("/app/timeline/api/month/badinput")

    assert response.status_code == 400
    assert response.get_json()["reason_code"] == "invalid_month"


def test_day_known_includes_hours_avail(client):
    response = client.get(f"/app/timeline/api/day/{DAY}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["day"] == DAY
    assert payload["day_top"][0]["title"] == "Timeline Port"
    assert payload["hours"]["10"]["picks"][0]["title"] == "Default Both"

    hour10 = payload["hours_avail"]["10"]["buckets"][0]
    assert hour10 == {
        "minute": 0,
        "best_origin": "20260510/100000_300",
        "has_audio": True,
        "has_screen": True,
        "segment_count": 1,
    }

    hour11 = payload["hours_avail"]["11"]["buckets"][0]
    assert hour11["best_origin"] == "20260510/default/110000_300"
    assert hour11["has_audio"] is True
    assert hour11["has_screen"] is True

    hour12 = payload["hours_avail"]["12"]["buckets"][0]
    assert hour12["best_origin"] == "20260510/default/120000_300"
    assert hour12["has_audio"] is True
    assert hour12["has_screen"] is False

    hour13 = payload["hours_avail"]["13"]["buckets"][0]
    assert hour13["best_origin"] == "20260510/default/130000_300"
    assert hour13["has_audio"] is False
    assert hour13["has_screen"] is True

    assert payload["hours_avail"]["10"]["buckets"][1]["best_origin"] is None


def test_day_bad_input_returns_400(client):
    response = client.get("/app/timeline/api/day/badinput")

    assert response.status_code == 400
    assert response.get_json()["reason_code"] == "invalid_day"


def test_segment_named_stream_loads_audio_and_screen(client):
    response = client.get(f"/app/timeline/api/segment/{DAY}/default/110000_300")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["day"] == DAY
    assert payload["stream"] == "default"
    assert payload["segment"] == "110000_300"
    assert payload["audio"]["header"]["setting"] == "desk"
    assert len(payload["audio"]["lines"]) == 2
    assert payload["screen"]["filename"] == "desktop.screen.jsonl"
    assert len(payload["screen"]["frames"]) == 2


def test_segment_default_stream_loads_top_level_segment(client):
    response = client.get(f"/app/timeline/api/segment/{DAY}/100000_300")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["stream"] == ""
    assert payload["audio"]["lines"][0]["text"] == "Reviewed timeline data."
    assert payload["screen"]["frames"][0]["analysis"]["primary"] == "code"


def test_segment_unknown_returns_seed_style_payload(client):
    response = client.get(f"/app/timeline/api/segment/{DAY}/unknown/999999_300")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["audio"] is None
    assert payload["screen"] is None
    assert payload["error"].startswith("segment dir not found: ")
    assert payload["error"].endswith("chronicle/20260510/unknown/999999_300")


def test_segment_bad_input_returns_400(client):
    response = client.get(f"/app/timeline/api/segment/{DAY}/default/badseg")

    assert response.status_code == 400
    assert response.get_json()["reason_code"] == "invalid_path"


def test_master_cache_invalidates_on_mtime(client, timeline_env):
    first = client.get("/app/timeline/api/index").get_json()
    first_title = next(m for m in first["months"] if m["ym"] == MONTH)["month_top"][0][
        "title"
    ]
    assert first_title == "Timeline Port"

    timeline_path = timeline_env / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    data["months"][MONTH]["month_top"][0]["title"] = "Updated Timeline"
    timeline_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    bumped = time.time() + 2
    os.utime(timeline_path, (bumped, bumped))

    second = client.get("/app/timeline/api/index").get_json()
    second_title = next(m for m in second["months"] if m["ym"] == MONTH)["month_top"][
        0
    ]["title"]
    assert second_title == "Updated Timeline"


def test_segment_lru_eviction(client, timeline_env):
    segment_root = timeline_env / "chronicle" / DAY / "default"
    for idx in range(33):
        seg = f"14{idx:02d}00_300"
        (segment_root / seg).mkdir()
        routes._load_segment(DAY, "default", seg)

    assert len(routes._seg_cache) <= routes._SEG_CACHE_MAX
