# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from think.agents import _is_retryable_error
from think.models import (
    TYPE_DEFAULTS,
    get_backup_provider,
    is_provider_healthy,
    should_recheck_health,
)


def test_is_provider_healthy_all_failed():
    health_data = {
        "results": [
            {"provider": "google", "ok": False},
            {"provider": "google", "ok": False},
        ]
    }
    assert is_provider_healthy("google", health_data) is False


def test_is_provider_healthy_some_passed():
    health_data = {
        "results": [
            {"provider": "google", "ok": False},
            {"provider": "google", "ok": True},
        ]
    }
    assert is_provider_healthy("google", health_data) is True


def test_is_provider_healthy_no_data():
    assert is_provider_healthy("google", None) is True


def test_is_provider_healthy_no_results_for_provider():
    health_data = {"results": [{"provider": "anthropic", "ok": False}]}
    assert is_provider_healthy("google", health_data) is True


def test_should_recheck_health_stale():
    checked_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    health_data = {"checked_at": checked_at}
    assert should_recheck_health(health_data) is True


def test_should_recheck_health_fresh():
    checked_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    health_data = {"checked_at": checked_at}
    assert should_recheck_health(health_data) is False


def test_get_backup_provider_from_config(monkeypatch):
    monkeypatch.setattr(
        "think.models.get_config",
        lambda: {"providers": {"generate": {"provider": "google", "backup": "openai"}}},
    )
    assert get_backup_provider("generate") == "openai"


def test_get_backup_provider_fallback_constant(monkeypatch):
    monkeypatch.setattr("think.models.get_config", lambda: {})
    assert get_backup_provider("generate") == TYPE_DEFAULTS["generate"]["backup"]
    assert get_backup_provider("cogitate") == TYPE_DEFAULTS["cogitate"]["backup"]


def test_get_backup_provider_none_when_same_as_primary(monkeypatch):
    monkeypatch.setattr(
        "think.models.get_config",
        lambda: {
            "providers": {
                "generate": {"provider": "openai", "backup": "openai"},
            }
        },
    )
    assert get_backup_provider("generate") is None


def _mock_base_agent_config() -> dict:
    return {
        "type": "cogitate",
        "path": None,
        "sources": {},
        "system_instruction": "",
        "user_instruction": "",
        "prompt": "",
        "disabled": False,
    }


def _patch_prepare_config_dependencies(monkeypatch):
    monkeypatch.setattr(
        "think.muse.get_agent", lambda *args, **kwargs: _mock_base_agent_config()
    )
    monkeypatch.setattr(
        "think.muse.key_to_context", lambda _name: "muse.system.default"
    )
    monkeypatch.setattr(
        "think.models.resolve_provider",
        lambda _context, _type: ("google", "gemini-3-flash-preview"),
    )


def test_preflight_swap_unhealthy_primary(monkeypatch):
    from think.agents import prepare_config

    _patch_prepare_config_dependencies(monkeypatch)
    monkeypatch.setattr(
        "think.models.load_health_status",
        lambda: {"results": [{"provider": "google", "ok": False}]},
    )
    monkeypatch.setattr("think.models.should_recheck_health", lambda _h: False)
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.setattr(
        "think.models.resolve_model_for_provider",
        lambda _context, _provider, _type="generate": "claude-sonnet-4-5",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = prepare_config({"name": "default", "prompt": "hello"})

    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-sonnet-4-5"
    assert config["fallback_from"] == "google"


def test_preflight_no_swap_healthy_primary(monkeypatch):
    from think.agents import prepare_config

    _patch_prepare_config_dependencies(monkeypatch)
    monkeypatch.setattr(
        "think.models.load_health_status",
        lambda: {"results": [{"provider": "google", "ok": True}]},
    )
    monkeypatch.setattr("think.models.should_recheck_health", lambda _h: False)

    config = prepare_config({"name": "default", "prompt": "hello"})

    assert config["provider"] == "google"
    assert "fallback_from" not in config


def test_preflight_no_swap_no_backup_key(monkeypatch):
    from think.agents import prepare_config

    _patch_prepare_config_dependencies(monkeypatch)
    monkeypatch.setattr(
        "think.models.load_health_status",
        lambda: {"results": [{"provider": "google", "ok": False}]},
    )
    monkeypatch.setattr("think.models.should_recheck_health", lambda _h: False)
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = prepare_config({"name": "default", "prompt": "hello"})

    assert config["provider"] == "google"
    assert "fallback_from" not in config


def test_on_failure_retry_cogitate(monkeypatch):
    from think.agents import _execute_with_tools

    events = []
    attempts = {"primary": 0, "backup": 0}

    async def fail_cogitate(*_args, **_kwargs):
        attempts["primary"] += 1
        raise RuntimeError("primary down")

    async def pass_cogitate(*_args, **kwargs):
        attempts["backup"] += 1
        on_event = kwargs.get("on_event")
        if on_event:
            on_event({"event": "finish", "result": "backup result"})
        return "backup result"

    monkeypatch.setattr(
        "think.providers.PROVIDER_REGISTRY", {"google": "x", "anthropic": "y"}
    )
    monkeypatch.setattr(
        "think.providers.get_provider_module",
        lambda provider: SimpleNamespace(
            run_cogitate=fail_cogitate if provider == "google" else pass_cogitate
        ),
    )
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.setattr(
        "think.models.resolve_model_for_provider",
        lambda _context, _provider, _type="cogitate": "claude-sonnet-4-5",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = {
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "health_stale": False,
        "context": "muse.system.default",
    }

    asyncio.run(_execute_with_tools(config, events.append))

    assert attempts["primary"] == 1
    assert attempts["backup"] == 1
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-sonnet-4-5"
    assert config["fallback_from"] == "google"
    assert any(e.get("event") == "fallback" for e in events)


def test_on_failure_retry_cogitate_uses_context_from_name(monkeypatch):
    from think.agents import _execute_with_tools

    events = []
    seen = {}

    async def fail_cogitate(*_args, **_kwargs):
        raise RuntimeError("primary down")

    async def pass_cogitate(*_args, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event:
            on_event({"event": "finish", "result": "backup result"})
        return "backup result"

    def resolve_model(context, _provider, _type="cogitate"):
        seen["context"] = context
        return "claude-sonnet-4-5"

    monkeypatch.setattr(
        "think.providers.PROVIDER_REGISTRY", {"google": "x", "anthropic": "y"}
    )
    monkeypatch.setattr(
        "think.providers.get_provider_module",
        lambda provider: SimpleNamespace(
            run_cogitate=fail_cogitate if provider == "google" else pass_cogitate
        ),
    )
    monkeypatch.setattr(
        "think.muse.key_to_context",
        lambda _name: "muse.system.default",
    )
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.setattr("think.models.resolve_model_for_provider", resolve_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = {
        "name": "default",
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "health_stale": False,
    }

    asyncio.run(_execute_with_tools(config, events.append))

    assert seen["context"] == "muse.system.default"


def test_on_failure_retry_generate(monkeypatch):
    from think.agents import _execute_generate

    events = []
    calls = {"count": 0}

    def mock_generate_with_result(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("primary generate failed")
        assert kwargs.get("provider") == "anthropic"
        assert kwargs.get("model") == "claude-sonnet-4-5"
        return {"text": "backup text", "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr(
        "think.muse.key_to_context", lambda _name: "muse.system.default"
    )
    monkeypatch.setattr("think.models.generate_with_result", mock_generate_with_result)
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.setattr(
        "think.models.resolve_model_for_provider",
        lambda _context, _provider, _type="generate": "claude-sonnet-4-5",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = {
        "name": "default",
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "prompt": "hello",
        "health_stale": False,
    }

    asyncio.run(_execute_generate(config, events.append))

    assert calls["count"] == 2
    assert config["provider"] == "anthropic"
    assert config["fallback_from"] == "google"
    assert any(e.get("event") == "fallback" for e in events)
    assert events[-1]["event"] == "finish"
    assert events[-1]["result"] == "backup text"


def test_on_failure_no_retry_value_error(monkeypatch):
    from think.agents import _execute_generate

    events = []
    assert _is_retryable_error(ValueError("bad input")) is False

    def bad_generate(**_kwargs):
        raise ValueError("bad input")

    monkeypatch.setattr(
        "think.muse.key_to_context", lambda _name: "muse.system.default"
    )
    monkeypatch.setattr("think.models.generate_with_result", bad_generate)

    config = {
        "name": "default",
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "prompt": "hello",
        "health_stale": False,
    }

    with pytest.raises(ValueError, match="bad input"):
        asyncio.run(_execute_generate(config, events.append))

    assert not any(e.get("event") == "fallback" for e in events)


def test_on_failure_both_fail_raises_original(monkeypatch):
    from think.agents import _execute_generate

    events = []
    calls = {"count": 0}

    def always_fail(**kwargs):
        calls["count"] += 1
        if kwargs.get("provider") == "anthropic":
            raise RuntimeError("backup failed")
        raise RuntimeError("primary failed")

    monkeypatch.setattr(
        "think.muse.key_to_context", lambda _name: "muse.system.default"
    )
    monkeypatch.setattr("think.models.generate_with_result", always_fail)
    monkeypatch.setattr("think.models.get_backup_provider", lambda _type: "anthropic")
    monkeypatch.setattr(
        "think.models.resolve_model_for_provider",
        lambda _context, _provider, _type="generate": "claude-sonnet-4-5",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = {
        "name": "default",
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "prompt": "hello",
        "health_stale": False,
    }

    with pytest.raises(RuntimeError, match="primary failed"):
        asyncio.run(_execute_generate(config, events.append))

    assert calls["count"] == 2


def test_fallback_event_emitted():
    from think.agents import _run_agent

    events = []
    config = {
        "type": "cogitate",
        "name": "default",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "prompt": "hello",
        "fallback_from": "google",
    }

    asyncio.run(_run_agent(config, events.append, dry_run=True))

    fallback_events = [e for e in events if e.get("event") == "fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0]["reason"] == "preflight"


def test_recheck_requested_on_stale(monkeypatch):
    from think.agents import _execute_with_tools

    async def pass_cogitate(*_args, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event:
            on_event({"event": "finish", "result": "ok"})
        return "ok"

    recheck_mock = MagicMock()

    monkeypatch.setattr("think.providers.PROVIDER_REGISTRY", {"google": "x"})
    monkeypatch.setattr(
        "think.providers.get_provider_module",
        lambda _provider: SimpleNamespace(run_cogitate=pass_cogitate),
    )
    monkeypatch.setattr("think.models.request_health_recheck", recheck_mock)

    config = {
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "health_stale": True,
    }

    asyncio.run(_execute_with_tools(config, lambda _e: None))

    recheck_mock.assert_called_once()
    assert config["health_stale"] is False


def test_main_async_no_duplicate_error_when_evented(monkeypatch, capsys):
    from think.agents import main_async

    ndjson_input = json.dumps({"name": "default", "prompt": "hello"})
    monkeypatch.setattr("sys.stdin", StringIO(ndjson_input))

    async def fake_run_agent(_config, emit_event, dry_run=False):
        emit_event({"event": "error", "error": "provider failed"})
        exc = RuntimeError("provider failed")
        setattr(exc, "_evented", True)
        raise exc

    mock_args = MagicMock()
    mock_args.verbose = False
    mock_args.dry_run = False
    mock_args.subcommand = None

    monkeypatch.setattr("think.agents.setup_cli", lambda _parser: mock_args)
    monkeypatch.setattr(
        "think.agents.setup_logging",
        lambda _verbose=False: MagicMock(),
    )
    monkeypatch.setattr(
        "think.agents.prepare_config", lambda _request: {"type": "cogitate"}
    )
    monkeypatch.setattr("think.agents.validate_config", lambda _config: None)
    monkeypatch.setattr("think.agents._run_agent", fake_run_agent)

    asyncio.run(main_async())

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    error_events = [event for event in events if event.get("event") == "error"]
    assert len(error_events) == 1
