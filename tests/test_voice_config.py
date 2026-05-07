# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from solstone.think.voice import config


def test_voice_config_defaults(monkeypatch):
    monkeypatch.setattr(config, "get_config", lambda: {"agent": {"name": "sol"}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert config.get_openai_api_key() is None
    assert config.get_voice_model() == "gpt-realtime"
    assert config.get_brain_model() == "haiku"


def test_voice_config_prefers_journal_key(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "voice": {
                "openai_api_key": "sk-config",
                "model": "gpt-realtime-mini",
                "brain_model": "sonnet",
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    assert config.get_openai_api_key() == "sk-config"
    assert config.get_voice_model() == "gpt-realtime-mini"
    assert config.get_brain_model() == "sonnet"


def test_voice_config_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(
        config, "get_config", lambda: {"voice": {"openai_api_key": " "}}
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    assert config.get_openai_api_key() == "sk-env"
