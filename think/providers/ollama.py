#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Ollama (Local) provider for direct LLM generation.

This module provides the Ollama provider for run_generate/run_agenerate
functions returning GenerateResult. It connects to a local Ollama instance
via its native ``/api/chat`` endpoint using ``httpx``.

The native API is used instead of the OpenAI-compatible endpoint because
it provides reliable control over the ``think`` parameter, which controls
model reasoning/thinking behavior. The OpenAI-compatible endpoint silently
ignores this parameter on models like Qwen3.5.

Cogitate (tool-calling agents) is not yet supported; ``run_cogitate()``
raises ``NotImplementedError``. Configure a cloud provider or backup for
agent workloads.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name with ``ollama-local/`` prefix (e.g., ``ollama-local/qwen3.5:9b``).
    The prefix is stripped before sending to the Ollama API.
temperature : float
    Temperature for generation (default: 0.3).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction for the model.
json_output : bool
    Whether to request JSON response format.
thinking_budget : int, optional
    Token budget for model thinking. When > 0, enables Ollama's ``think``
    parameter. When None or 0, thinking is explicitly disabled.
timeout_s : float, optional
    Request timeout in seconds.
**kwargs
    Additional provider-specific options (absorbed for forward compatibility).

Environment Variables
---------------------
OLLAMA_BASE_URL : str
    Base URL for the Ollama server (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import httpx

from .shared import GenerateResult

LOG = logging.getLogger("think.providers.ollama")

_OLLAMA_LOCAL_PREFIX = "ollama-local/"
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def _get_base_url() -> str:
    """Get Ollama base URL from environment or default."""
    return os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _get_client() -> httpx.Client:
    """Get or create cached sync httpx client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            base_url=_get_base_url(),
            timeout=_DEFAULT_TIMEOUT,
        )
    return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    """Get or create cached async httpx client."""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            base_url=_get_base_url(),
            timeout=_DEFAULT_TIMEOUT,
        )
    return _async_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_model_prefix(model: str) -> str:
    """Strip the ``ollama-local/`` prefix for the Ollama API.

    The Ollama API expects bare model names like ``qwen3.5:9b``, but
    Solstone uses the ``ollama-local/`` prefix for provider routing.
    """
    if model.startswith(_OLLAMA_LOCAL_PREFIX):
        return model[len(_OLLAMA_LOCAL_PREFIX) :]
    return model


def _build_messages(
    contents: Any,
    system_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Convert contents and system instruction to chat messages.

    Parameters
    ----------
    contents
        String, list of strings, or list of message dicts with ``role`` keys.
    system_instruction
        Optional system prompt, prepended as a system message.

    Returns
    -------
    list[dict[str, str]]
        Messages in ``[{role, content}, ...]`` format.
    """
    messages: list[dict[str, str]] = []

    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            messages.extend(contents)
        else:
            messages.append(
                {"role": "user", "content": "\n".join(str(c) for c in contents)}
            )
    else:
        messages.append({"role": "user", "content": str(contents)})

    return messages


def _build_request_body(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    json_output: bool,
    thinking_budget: int | None,
) -> dict[str, Any]:
    """Build the native Ollama /api/chat request body.

    Parameters
    ----------
    model
        Bare model name (prefix already stripped).
    messages
        Chat messages list.
    temperature
        Sampling temperature.
    max_output_tokens
        Maximum response tokens (``num_predict`` in Ollama).
    json_output
        Whether to request JSON response format.
    thinking_budget
        Thinking token budget; > 0 enables, None/0 disables.

    Returns
    -------
    dict
        Request body for ``POST /api/chat``.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
        },
    }

    # Thinking control: this is the reason we use the native API.
    # The OpenAI-compat endpoint ignores this parameter.
    if thinking_budget is not None and thinking_budget > 0:
        body["think"] = True
    else:
        body["think"] = False

    if json_output:
        body["format"] = "json"

    return body


def _normalize_finish_reason(data: dict[str, Any]) -> str | None:
    """Normalize Ollama's done_reason to standard values.

    Returns ``"stop"``, ``"max_tokens"``, or None.
    """
    if not data.get("done"):
        return None

    reason = data.get("done_reason", "")
    if reason == "stop":
        return "stop"
    elif reason == "length":
        return "max_tokens"
    elif reason:
        return reason
    return "stop"  # done=True with no reason implies normal completion


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    """Extract normalized usage dict from native Ollama response.

    Ollama uses ``prompt_eval_count`` and ``eval_count`` instead of the
    OpenAI-style ``prompt_tokens`` / ``completion_tokens``.
    """
    input_tokens = data.get("prompt_eval_count", 0) or 0
    output_tokens = data.get("eval_count", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_thinking(data: dict[str, Any]) -> list | None:
    """Extract thinking content from native Ollama response.

    The native API returns a ``thinking`` field on the message when
    thinking is enabled.
    """
    message = data.get("message", {})
    thinking = message.get("thinking")
    if thinking and isinstance(thinking, str) and thinking.strip():
        return [{"summary": thinking.strip()}]
    return None


def _parse_response(data: dict[str, Any]) -> GenerateResult:
    """Parse the native Ollama /api/chat response into GenerateResult."""
    message = data.get("message", {})
    text = message.get("content", "")

    return GenerateResult(
        text=text,
        usage=_extract_usage(data),
        finish_reason=_normalize_finish_reason(data),
        thinking=_extract_thinking(data),
    )


# ---------------------------------------------------------------------------
# run_generate / run_agenerate
# ---------------------------------------------------------------------------


def run_generate(
    contents: str | list[Any],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously via local Ollama.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_client()
    api_model = _strip_model_prefix(model)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        api_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
    )

    response = client.post(
        "/api/chat",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json())


async def run_agenerate(
    contents: str | list[Any],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously via local Ollama.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_async_client()
    api_model = _strip_model_prefix(model)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        api_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
    )

    response = await client.post(
        "/api/chat",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json())


# ---------------------------------------------------------------------------
# run_cogitate (not yet implemented)
# ---------------------------------------------------------------------------


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Tool-calling agent execution — not yet implemented for Ollama.

    Raises
    ------
    NotImplementedError
        Always. Configure a cloud provider for cogitate or set a backup provider.
    """
    raise NotImplementedError(
        "Ollama cogitate support is not yet implemented. "
        "Configure a cloud provider for cogitate or set a backup provider."
    )


# ---------------------------------------------------------------------------
# list_models / validate_key
# ---------------------------------------------------------------------------


def list_models() -> list[dict]:
    """List available models from the local Ollama instance.

    Returns
    -------
    list[dict]
        List of model info dicts from the Ollama ``/api/tags`` endpoint.
    """
    client = _get_client()
    response = client.get("/api/tags")
    response.raise_for_status()
    return response.json().get("models", [])


def validate_key(api_key: str) -> dict:
    """Check that the local Ollama instance is reachable.

    The ``api_key`` parameter is ignored — Ollama requires no authentication.
    Connectivity is validated by hitting the version endpoint.

    Returns ``{"valid": True}`` if reachable, ``{"valid": False, "error": "..."}``
    if not.
    """
    try:
        base_url = _get_base_url()
        response = httpx.get(f"{base_url}/api/version", timeout=5)
        response.raise_for_status()
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)}


__all__ = [
    "run_generate",
    "run_agenerate",
    "run_cogitate",
    "list_models",
    "validate_key",
]
