# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for the Ollama (Local) provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from think.models import OLLAMA_FLASH, OLLAMA_LITE, OLLAMA_PRO


def _ollama_provider():
    import importlib

    return importlib.reload(importlib.import_module("think.providers.ollama"))


# ---------------------------------------------------------------------------
# _strip_model_prefix
# ---------------------------------------------------------------------------


class TestStripModelPrefix:
    def test_strips_ollama_local_prefix(self):
        provider = _ollama_provider()
        assert provider._strip_model_prefix("ollama-local/qwen3.5:9b") == "qwen3.5:9b"

    def test_strips_prefix_from_complex_name(self):
        provider = _ollama_provider()
        assert (
            provider._strip_model_prefix("ollama-local/qwen3.5:35b-a3b-bf16")
            == "qwen3.5:35b-a3b-bf16"
        )

    def test_no_prefix_passthrough(self):
        provider = _ollama_provider()
        assert provider._strip_model_prefix("llama3.1:8b") == "llama3.1:8b"


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_string_contents(self):
        provider = _ollama_provider()
        msgs = provider._build_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_string_with_system(self):
        provider = _ollama_provider()
        msgs = provider._build_messages("hello", system_instruction="be helpful")
        assert msgs == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]

    def test_list_of_strings(self):
        provider = _ollama_provider()
        msgs = provider._build_messages(["line1", "line2"])
        assert msgs == [{"role": "user", "content": "line1\nline2"}]

    def test_list_of_dicts_passthrough(self):
        provider = _ollama_provider()
        input_msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        msgs = provider._build_messages(input_msgs)
        assert msgs == input_msgs

    def test_list_of_dicts_with_system(self):
        provider = _ollama_provider()
        input_msgs = [{"role": "user", "content": "hi"}]
        msgs = provider._build_messages(input_msgs, system_instruction="sys")
        assert msgs == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_non_string_contents(self):
        provider = _ollama_provider()
        msgs = provider._build_messages(42)
        assert msgs == [{"role": "user", "content": "42"}]


# ---------------------------------------------------------------------------
# _build_request_body
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    def test_basic_body(self):
        provider = _ollama_provider()
        body = provider._build_request_body(
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_output_tokens=1024,
            json_output=False,
            thinking_budget=None,
        )
        assert body["model"] == "qwen3.5:9b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.3
        assert body["options"]["num_predict"] == 1024
        assert body["think"] is False

    def test_thinking_enabled(self):
        provider = _ollama_provider()
        body = provider._build_request_body(
            "m",
            [{"role": "user", "content": "hi"}],
            0.3,
            1024,
            False,
            thinking_budget=4096,
        )
        assert body["think"] is True

    def test_thinking_disabled_zero(self):
        provider = _ollama_provider()
        body = provider._build_request_body(
            "m",
            [{"role": "user", "content": "hi"}],
            0.3,
            1024,
            False,
            thinking_budget=0,
        )
        assert body["think"] is False

    def test_json_output(self):
        provider = _ollama_provider()
        body = provider._build_request_body(
            "m", [{"role": "user", "content": "hi"}], 0.3, 1024, True, None
        )
        assert body["format"] == "json"

    def test_no_json_output(self):
        provider = _ollama_provider()
        body = provider._build_request_body(
            "m", [{"role": "user", "content": "hi"}], 0.3, 1024, False, None
        )
        assert "format" not in body


# ---------------------------------------------------------------------------
# _normalize_finish_reason
# ---------------------------------------------------------------------------


class TestNormalizeFinishReason:
    def test_stop(self):
        provider = _ollama_provider()
        assert (
            provider._normalize_finish_reason({"done": True, "done_reason": "stop"})
            == "stop"
        )

    def test_length_to_max_tokens(self):
        provider = _ollama_provider()
        assert (
            provider._normalize_finish_reason({"done": True, "done_reason": "length"})
            == "max_tokens"
        )

    def test_done_no_reason(self):
        provider = _ollama_provider()
        assert provider._normalize_finish_reason({"done": True}) == "stop"

    def test_not_done(self):
        provider = _ollama_provider()
        assert provider._normalize_finish_reason({"done": False}) is None

    def test_unknown_passthrough(self):
        provider = _ollama_provider()
        assert (
            provider._normalize_finish_reason({"done": True, "done_reason": "other"})
            == "other"
        )


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_normal_usage(self):
        provider = _ollama_provider()
        result = provider._extract_usage(
            {
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
        )
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_missing_fields(self):
        provider = _ollama_provider()
        result = provider._extract_usage({})
        assert result == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }


# ---------------------------------------------------------------------------
# _extract_thinking
# ---------------------------------------------------------------------------


class TestExtractThinking:
    def test_with_thinking(self):
        provider = _ollama_provider()
        result = provider._extract_thinking(
            {"message": {"content": "4", "thinking": "Let me calculate..."}}
        )
        assert result == [{"summary": "Let me calculate..."}]

    def test_no_thinking(self):
        provider = _ollama_provider()
        result = provider._extract_thinking({"message": {"content": "4"}})
        assert result is None

    def test_empty_thinking(self):
        provider = _ollama_provider()
        result = provider._extract_thinking(
            {"message": {"content": "4", "thinking": "   "}}
        )
        assert result is None

    def test_no_message(self):
        provider = _ollama_provider()
        result = provider._extract_thinking({})
        assert result is None


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_full_response(self):
        provider = _ollama_provider()
        data = {
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "thinking": "Reasoning...",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 25,
            "eval_count": 10,
        }
        result = provider._parse_response(data)
        assert result["text"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["input_tokens"] == 25
        assert result["usage"]["output_tokens"] == 10
        assert result["thinking"] == [{"summary": "Reasoning..."}]

    def test_no_thinking(self):
        provider = _ollama_provider()
        data = {
            "message": {"role": "assistant", "content": "4"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 2,
        }
        result = provider._parse_response(data)
        assert result["text"] == "4"
        assert result["thinking"] is None


# ---------------------------------------------------------------------------
# run_generate
# ---------------------------------------------------------------------------


def _make_ollama_response(
    content="Hello!",
    thinking=None,
    done=True,
    done_reason="stop",
    prompt_eval_count=10,
    eval_count=5,
):
    """Build a mock native Ollama /api/chat response dict."""
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": "qwen3.5:9b",
        "message": message,
        "done": done,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


class TestRunGenerate:
    def test_basic_generation(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            result = provider.run_generate("hello", model=OLLAMA_FLASH)

        assert result["text"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["input_tokens"] == 10

    def test_model_prefix_stripped(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate("hello", model="ollama-local/qwen3.5:9b")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["model"] == "qwen3.5:9b"

    def test_thinking_enabled(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response(thinking="Reasoning...")
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            result = provider.run_generate(
                "hello", model=OLLAMA_FLASH, thinking_budget=4096
            )

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["think"] is True
        assert result["thinking"] == [{"summary": "Reasoning..."}]

    def test_thinking_disabled_when_none(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate("hello", model=OLLAMA_FLASH, thinking_budget=None)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["think"] is False

    def test_thinking_disabled_when_zero(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate("hello", model=OLLAMA_FLASH, thinking_budget=0)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["think"] is False

    def test_json_output(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response(
            content='{"key": "value"}'
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate("hello", model=OLLAMA_FLASH, json_output=True)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["format"] == "json"

    def test_system_instruction(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate(
                "hello", model=OLLAMA_FLASH, system_instruction="be concise"
            )

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        messages = body["messages"]
        assert messages[0] == {"role": "system", "content": "be concise"}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_timeout(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            provider.run_generate("hello", model=OLLAMA_FLASH, timeout_s=30.0)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["timeout"] == 30.0


# ---------------------------------------------------------------------------
# run_agenerate
# ---------------------------------------------------------------------------


class TestRunAgenerate:
    def test_async_generation(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_async_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client

            result = asyncio.run(provider.run_agenerate("hello", model=OLLAMA_FLASH))

        assert result["text"] == "Hello!"
        assert result["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# run_cogitate
# ---------------------------------------------------------------------------


class TestRunCogitate:
    def test_raises_not_implemented(self):
        provider = _ollama_provider()

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            asyncio.run(provider.run_cogitate({"prompt": "test"}, on_event=None))


# ---------------------------------------------------------------------------
# list_models / validate_key
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_model_list(self):
        provider = _ollama_provider()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3.5:9b", "size": 6600000000},
                {"name": "llama3.1:8b", "size": 4900000000},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get.return_value = mock_client

            result = provider.list_models()

        assert len(result) == 2
        assert result[0]["name"] == "qwen3.5:9b"


class TestValidateKey:
    def test_reachable(self):
        provider = _ollama_provider()

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"version": "0.18.3"}
            mock_get.return_value = mock_response

            result = provider.validate_key("ignored")

        assert result == {"valid": True}

    def test_unreachable(self):
        provider = _ollama_provider()

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = provider.validate_key("ignored")

        assert result["valid"] is False
        assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

import httpx


class TestModelConstants:
    def test_default_models_have_prefix(self):
        assert OLLAMA_PRO.startswith("ollama-local/")
        assert OLLAMA_FLASH.startswith("ollama-local/")
        assert OLLAMA_LITE.startswith("ollama-local/")

    def test_get_model_provider(self):
        from think.models import get_model_provider

        assert get_model_provider(OLLAMA_PRO) == "ollama"
        assert get_model_provider(OLLAMA_FLASH) == "ollama"
        assert get_model_provider(OLLAMA_LITE) == "ollama"

    def test_provider_defaults_exist(self):
        from think.models import PROVIDER_DEFAULTS

        assert "ollama" in PROVIDER_DEFAULTS
        assert 1 in PROVIDER_DEFAULTS["ollama"]
        assert 2 in PROVIDER_DEFAULTS["ollama"]
        assert 3 in PROVIDER_DEFAULTS["ollama"]

    def test_calc_token_cost_zero(self):
        from think.models import calc_token_cost

        result = calc_token_cost(
            {
                "model": OLLAMA_FLASH,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        assert result is not None
        assert result["total_cost"] == 0.0

    def test_provider_registry(self):
        from think.providers import PROVIDER_METADATA, PROVIDER_REGISTRY

        assert "ollama" in PROVIDER_REGISTRY
        assert "ollama" in PROVIDER_METADATA
        assert PROVIDER_METADATA["ollama"]["label"] == "Ollama (Local)"
        assert PROVIDER_METADATA["ollama"]["env_key"] == ""
