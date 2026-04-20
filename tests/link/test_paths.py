# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import pytest

from think.link.paths import (
    DEFAULT_RELAY_URL,
    LinkState,
    account_token_path,
    load_account_token,
    relay_url,
    save_account_token,
    state_path,
)


def _set_journal(monkeypatch: pytest.MonkeyPatch, journal: Path) -> None:
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))


def test_link_state_load_or_create_creates_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    state = LinkState.load_or_create()

    assert isinstance(state.instance_id, str)
    assert state.instance_id
    assert state.home_label == "solstone"
    assert state_path().exists()


def test_link_state_load_or_create_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    first = LinkState.load_or_create()
    first_payload = state_path().read_text("utf-8")
    second = LinkState.load_or_create()
    second_payload = state_path().read_text("utf-8")

    assert second.instance_id == first.instance_id
    assert second.home_label == first.home_label
    assert second_payload == first_payload


def test_link_state_load_or_create_custom_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    created = LinkState.load_or_create(default_label="laptop")
    loaded = LinkState.load_or_create()

    assert created.home_label == "laptop"
    assert loaded.instance_id == created.instance_id
    assert loaded.home_label == "laptop"


def test_relay_url_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_journal(monkeypatch, tmp_path)
    monkeypatch.setenv("SOL_LINK_RELAY_URL", "https://example.test/")

    assert relay_url() == "https://example.test"


def test_relay_url_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_journal(monkeypatch, tmp_path)
    config_path = tmp_path / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"link": {"relay_url": "https://cfg.test"}}),
        encoding="utf-8",
    )

    assert relay_url() == "https://cfg.test"


def test_relay_url_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_journal(monkeypatch, tmp_path)
    monkeypatch.delenv("SOL_LINK_RELAY_URL", raising=False)

    assert relay_url() == DEFAULT_RELAY_URL


def test_load_account_token_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    assert load_account_token() is None


def test_save_and_load_account_token_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    save_account_token("tok.123")

    token_path = account_token_path()
    assert load_account_token() == "tok.123"
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_save_account_token_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_journal(monkeypatch, tmp_path)

    save_account_token("tok.123")

    token_path = account_token_path()
    assert token_path.exists()
    assert not any(path.name.endswith(".tmp") for path in token_path.parent.iterdir())
