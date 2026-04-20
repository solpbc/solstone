# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx

from convey import create_app
from think.push import devices, runtime, triggers

TEST_KEY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg+Zj7Bk6Dzp080/PU
jTZnJ6kP4KtlHErFO/WuVRTQvkShRANCAARW8djY5HF7K8noSZQRfjP38mIzaufi
/YPI38YuaWmiPIqRmwDOu5rICl4PPLem4k+qtb950rlYCGx3J+MQN9tO
-----END PRIVATE KEY-----
"""


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 27, 8, 45, 0, tzinfo=tz)


def _write_push_config(journal_copy: Path) -> None:
    key_path = journal_copy / "keys" / "apns.p8"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(TEST_KEY, encoding="utf-8")
    config_path = journal_copy / "config" / "journal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["push"] = {
        "apns_key_path": str(key_path),
        "apns_key_id": "KEY123",
        "apns_team_id": "TEAM123",
        "bundle_id": "org.solpbc.solstone-swift",
        "environment": "development",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _seed_activity(journal_copy: Path, facet: str, day: str, rows: list[dict]) -> None:
    path = journal_copy / "facets" / facet / "activities" / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_push_integration_briefing_dispatch_and_log(journal_copy, monkeypatch):
    _write_push_config(journal_copy)
    devices.register_device(
        token="a" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="development",
        platform="ios",
    )
    monkeypatch.setattr(
        "think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: None,
    )
    monkeypatch.setattr("think.push.runtime.CallosumConnection.stop", lambda self: None)
    monkeypatch.setattr(triggers, "datetime", FixedDateTime)
    captured: list[str] = []

    async def fake_post(self, url, *, headers, json):
        captured.append(url)
        return httpx.Response(200)

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        app = create_app(str(journal_copy))
        app.config["TESTING"] = True
        runtime._on_callosum_message(
            {"tract": "cortex", "event": "finish", "name": "morning_briefing"}
        )

    log_path = journal_copy / "push" / "nudge_log.jsonl"
    lines = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert captured == [f"https://api.sandbox.push.apple.com/3/device/{'a' * 64}"]
    assert lines[0]["category"] == "SOLSTONE_DAILY_BRIEFING"
    runtime.stop_all_push_runtime()
    assert runtime.get_runtime_state() is None


def test_push_integration_pre_meeting_and_muted_facet(journal_copy):
    _write_push_config(journal_copy)
    devices.register_device(
        token="b" * 64,
        bundle_id="org.solpbc.solstone-swift",
        environment="development",
        platform="ios",
    )
    muted_facet = journal_copy / "facets" / "muted"
    muted_facet.mkdir(parents=True, exist_ok=True)
    (muted_facet / "facet.json").write_text(
        json.dumps({"muted": True}), encoding="utf-8"
    )
    _seed_activity(
        journal_copy,
        "montague",
        "20260327",
        [
            {
                "id": "anticipated_meeting_090000_0327",
                "source": "anticipated",
                "start": "09:00",
                "title": "Launch sync",
            }
        ],
    )
    _seed_activity(
        journal_copy,
        "muted",
        "20260327",
        [
            {
                "id": "anticipated_meeting_090000_muted",
                "source": "anticipated",
                "start": "09:00",
                "title": "Muted meeting",
            }
        ],
    )
    captured: list[str] = []

    async def fake_post(self, url, *, headers, json):
        captured.append(headers["apns-collapse-id"])
        return httpx.Response(200)

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        triggers.check_pre_meeting_prep(datetime(2026, 3, 27, 8, 45, 0))

    log_path = journal_copy / "push" / "nudge_log.jsonl"
    lines = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert captured == ["meeting.anticipated_meeting_090000_0327"]
    assert any(line["category"] == "SOLSTONE_PRE_MEETING_PREP" for line in lines)
