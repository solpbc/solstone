# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

from think.pairing import config


def _write_config(journal: Path, payload: dict) -> None:
    config_path = journal / "config" / "journal.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def _read_config(journal: Path) -> dict:
    return json.loads((journal / "config" / "journal.json").read_text(encoding="utf-8"))


def test_pairing_config_defaults(journal_copy):
    payload = _read_config(journal_copy)
    payload.pop("pairing", None)
    payload["identity"] = {"name": "", "preferred": ""}
    _write_config(journal_copy, payload)

    assert config.get_host_url() == "http://localhost:5015"
    assert config.get_token_ttl_seconds() == 600
    assert config.get_owner_identity() == ""


def test_pairing_host_url_reads_trimmed_value(journal_copy):
    payload = _read_config(journal_copy)
    payload["pairing"] = {
        "host_url": " https://example.test/base ",
        "token_ttl_seconds": 600,
    }
    _write_config(journal_copy, payload)

    assert config.get_host_url() == "https://example.test/base"


def test_pairing_host_url_synthesizes_recorded_convey_port(journal_copy):
    payload = _read_config(journal_copy)
    payload["pairing"] = {"host_url": None}
    _write_config(journal_copy, payload)
    health_dir = journal_copy / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "convey.port").write_text("6123", encoding="utf-8")

    assert config.get_host_url() == "http://localhost:6123"


def test_pairing_token_ttl_clamps(journal_copy):
    payload = _read_config(journal_copy)

    payload["pairing"] = {"token_ttl_seconds": 59}
    _write_config(journal_copy, payload)
    assert config.get_token_ttl_seconds() == 60

    payload["pairing"] = {"token_ttl_seconds": 60}
    _write_config(journal_copy, payload)
    assert config.get_token_ttl_seconds() == 60

    payload["pairing"] = {"token_ttl_seconds": 600}
    _write_config(journal_copy, payload)
    assert config.get_token_ttl_seconds() == 600

    payload["pairing"] = {"token_ttl_seconds": 3600}
    _write_config(journal_copy, payload)
    assert config.get_token_ttl_seconds() == 3600

    payload["pairing"] = {"token_ttl_seconds": 3601}
    _write_config(journal_copy, payload)
    assert config.get_token_ttl_seconds() == 3600


def test_pairing_owner_identity_fallbacks(journal_copy):
    payload = _read_config(journal_copy)
    payload["identity"] = {"name": "Sol", "preferred": "Sol Preferred"}
    _write_config(journal_copy, payload)
    assert config.get_owner_identity() == "Sol Preferred"

    payload["identity"] = {"name": "Sol", "preferred": "  "}
    _write_config(journal_copy, payload)
    assert config.get_owner_identity() == "Sol"

    payload["identity"] = {"name": " ", "preferred": None}
    _write_config(journal_copy, payload)
    assert config.get_owner_identity() == ""
