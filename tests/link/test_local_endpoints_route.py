# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solstone.think.link.interface_watcher import set_interface_watcher
from solstone.think.link.local_endpoints import LocalEndpoint


class _StubWatcher:
    def __init__(self, endpoints: list[LocalEndpoint]) -> None:
        self._endpoints = endpoints

    def snapshot(self) -> list[LocalEndpoint]:
        return list(self._endpoints)


@pytest.fixture
def link_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    journal = tmp_path / "journal"
    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "convey": {"trust_localhost": True},
                "setup": {"completed_at": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

    from solstone.convey import create_app

    app = create_app(str(journal))
    set_interface_watcher(None)
    try:
        yield app.test_client()
    finally:
        set_interface_watcher(None)


def test_local_endpoints_returns_empty_without_watcher(link_client) -> None:
    response = link_client.get("/app/link/local-endpoints")

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == {"v", "endpoints", "ttl_s", "generated_at"}
    assert payload["v"] == 1
    assert payload["endpoints"] == []
    assert payload["ttl_s"] == 3600
    assert isinstance(payload["generated_at"], str)


def test_local_endpoints_returns_watcher_snapshot(link_client) -> None:
    set_interface_watcher(
        _StubWatcher(
            [
                LocalEndpoint(ip="192.168.1.10", port=7657, scope="lan"),
                LocalEndpoint(ip="fd00::1", port=7657, scope="ula"),
            ]
        )
    )

    response = link_client.get("/app/link/local-endpoints")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["endpoints"] == [
        {"ip": "192.168.1.10", "port": 7657, "scope": "lan"},
        {"ip": "fd00::1", "port": 7657, "scope": "ula"},
    ]
    assert all(set(ep) == {"ip", "port", "scope"} for ep in payload["endpoints"])


def test_local_endpoints_non_loopback_404(link_client) -> None:
    with link_client.session_transaction() as session:
        session["logged_in"] = True
    response = link_client.get(
        "/app/link/local-endpoints",
        environ_base={"REMOTE_ADDR": "192.168.1.5"},
    )

    assert response.status_code == 404
