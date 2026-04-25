#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini provider for agents and direct LLM generation.

This module provides the Google Gemini provider for the ``sol providers check`` CLI
and run_generate/run_agenerate functions returning GenerateResult.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name to use.
temperature : float
    Temperature for generation (default: 0.3).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction for the model.
json_output : bool
    Whether to request JSON response format.
thinking_budget : int, optional
    Token budget for model thinking.
timeout_s : float, optional
    Request timeout in seconds.
**kwargs
    Provider-specific options (client).
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

from think.models import GEMINI_FLASH
from think.utils import now_ms

from .cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    build_cogitate_env,
)
from .shared import (
    GenerateResult,
    JSONEventCallback,
    ThinkingEvent,
    safe_raw,
)

GEMINI_MAX_OUTPUT_TOKENS = 65536
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_MODEL = GEMINI_FLASH

logger = logging.getLogger(__name__)

# Backend detection cache
_detected_backend: str | None = None

_COGITATE_POLICY_PATH = Path(__file__).parent.parent / "policies" / "cogitate.toml"


def _structured_to_google_contents(
    messages: list[dict[str, str]],
) -> list[types.Content]:
    """Map role/content dicts to Gemini-native Content objects."""
    mapped: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            google_role = "user"
        elif role == "assistant":
            google_role = "model"
        else:
            raise ValueError(f"Unknown message role: {role!r}")
        mapped.append(
            types.Content(
                role=google_role,
                parts=[types.Part(text=msg["content"])],
            )
        )
    return mapped


# ---------------------------------------------------------------------------
# Client and helper functions for generate/agenerate
# ---------------------------------------------------------------------------


def _probe_backend(api_key: str) -> str:
    """Probe AI Studio endpoint to classify key type.

    Returns ``"aistudio"`` when the key works against the AI Studio models
    endpoint (HTTP 200) or ``"vertex"`` otherwise. Network errors default
    to ``"aistudio"`` for backward compatibility.
    """
    try:
        import httpx

        resp = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=5,
        )
        return "aistudio" if resp.status_code == 200 else "vertex"
    except Exception:
        return "aistudio"


def _detect_backend(api_key: str) -> str:
    """Return cached backend detection result, probing on first call."""
    global _detected_backend
    if _detected_backend is not None:
        return _detected_backend
    _detected_backend = _probe_backend(api_key)
    return _detected_backend


def _get_effective_backend(api_key: str) -> str:
    """Return effective backend, checking config override before cache.

    Reads ``providers.google_backend`` from journal config. Values
    ``"aistudio"`` or ``"vertex"`` bypass detection; ``"auto"`` (the default
    when the key is absent) uses :func:`_detect_backend`.
    """
    from think.utils import get_config

    configured = get_config().get("providers", {}).get("google_backend", "auto")
    if configured in ("aistudio", "vertex"):
        return configured
    return _detect_backend(api_key)


def get_or_create_client(client: genai.Client | None = None) -> genai.Client:
    """Get existing client or create new one.

    For Vertex AI backend, uses service account credentials from config
    or falls back to GOOGLE_APPLICATION_CREDENTIALS env var.
    For AI Studio / auto-detect, uses GOOGLE_API_KEY.
    """
    if client is not None:
        return client

    from think.utils import get_config

    config = get_config()
    providers_config = config.get("providers", {})

    http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=8))

    api_key = os.getenv("GOOGLE_API_KEY")

    # Determine backend
    configured_backend = providers_config.get("google_backend", "auto")
    if configured_backend == "vertex":
        backend = "vertex"
    elif configured_backend == "aistudio":
        backend = "aistudio"
    elif api_key:
        backend = _get_effective_backend(api_key)
    else:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    if backend == "vertex":
        creds_path = providers_config.get("vertex_credentials")

        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "http_options": http_options,
        }

        if creds_path and os.path.exists(creds_path):
            import json as _json

            from google.oauth2.service_account import Credentials

            creds = Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client_kwargs["credentials"] = creds
            with open(creds_path, encoding="utf-8") as _f:
                _sa_data = _json.load(_f)
            if "project_id" in _sa_data:
                client_kwargs["project"] = _sa_data["project_id"]
        elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Vertex AI backend requires service account credentials. "
                "Configure in Settings or set GOOGLE_APPLICATION_CREDENTIALS."
            )
        # else: GOOGLE_APPLICATION_CREDENTIALS is set, SDK auto-discovers

        client = genai.Client(**client_kwargs)
    else:
        # AI Studio path
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=http_options,
        )

    return client


def _compute_agent_thinking_params(
    max_output_tokens: int, thinking_budget: int | None
) -> tuple[int, int]:
    """Compute total tokens and effective thinking budget for agent run.

    Args:
        max_output_tokens: Maximum output tokens from config.
        thinking_budget: Thinking budget from config, or None for dynamic.

    Returns:
        Tuple of (total_tokens, effective_thinking_budget).
        total_tokens = max_output_tokens + (thinking_budget or 0)
        effective_thinking_budget = thinking_budget if provided, else -1 (dynamic)
    """
    total_tokens = max_output_tokens + (thinking_budget or 0)
    effective_thinking_budget = thinking_budget if thinking_budget is not None else -1
    return total_tokens, effective_thinking_budget


def _build_generate_config(
    temperature: float,
    max_output_tokens: int,
    system_instruction: str | None,
    json_output: bool,
    thinking_budget: int | None,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig.

    Note: Gemini's max_output_tokens is actually the total budget (thinking + output).
    We compute this internally: total = max_output_tokens + thinking_budget.
    """
    # Compute total tokens: output + thinking budget
    total_tokens = max_output_tokens + (thinking_budget or 0)
    if total_tokens > GEMINI_MAX_OUTPUT_TOKENS:
        clamped_max_output = min(max_output_tokens, GEMINI_MAX_OUTPUT_TOKENS)
        clamped_thinking = max(0, GEMINI_MAX_OUTPUT_TOKENS - clamped_max_output)
        logging.getLogger(__name__).warning(
            "Clamping Gemini token budget: max_output_tokens=%s thinking_budget=%s "
            "clamped_max_output_tokens=%s clamped_thinking_budget=%s",
            max_output_tokens,
            thinking_budget,
            clamped_max_output,
            clamped_thinking,
        )
        thinking_budget = clamped_thinking
        total_tokens = clamped_max_output + clamped_thinking

    config_args: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": total_tokens,
    }

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    if json_output:
        config_args["response_mime_type"] = "application/json"
        if json_schema is not None:
            config_args["response_json_schema"] = json_schema

    # Set thinking config when caller explicitly specified a budget.
    # thinking_budget=0 must explicitly disable thinking (not omit config),
    # otherwise Gemini applies its own default budget consuming output tokens.
    if thinking_budget is not None:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    if timeout_s:
        # Convert seconds to milliseconds for the SDK
        timeout_ms = int(timeout_s * 1000)
        config_args["http_options"] = types.HttpOptions(timeout=timeout_ms)

    return types.GenerateContentConfig(**config_args)


def _extract_response_text(response: Any) -> str:
    """Extract text from response.

    Returns response.text if available, or a friendly completion message
    if the response is empty. Raises on safety filter blocks.

    Parameters
    ----------
    response
        The response from the model.
    """
    if response is None:
        raise ValueError("No response from model")

    # Check for error conditions in candidates
    finish_reason = _extract_finish_reason(response)
    if finish_reason and "SAFETY" in finish_reason.upper():
        raise ValueError(f"Response blocked by safety filters: {finish_reason}")

    # Extract text, or generate friendly message if empty
    text = response.text if response.text else ""
    if text:
        return text

    # Empty text - generate user-friendly completion message
    return _format_completion_message(finish_reason, had_tool_calls=False)


def _normalize_finish_reason(response: Any) -> str | None:
    """Normalize finish_reason to standard values.

    Returns normalized string: "stop", "max_tokens", "safety", or None.
    """
    raw = _extract_finish_reason(response)
    if not raw:
        return None

    # Normalize (handle both enum names and string values)
    reason = raw.upper().replace("FINISHREASON.", "")

    if reason == "STOP":
        return "stop"
    elif reason == "MAX_TOKENS":
        return "max_tokens"
    elif "SAFETY" in reason:
        return "safety"
    elif reason == "RECITATION":
        return "recitation"
    else:
        return reason.lower()


def _extract_usage(response: Any) -> dict | None:
    """Extract normalized usage dict from response."""
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return None

    metadata = response.usage_metadata
    usage: dict[str, int] = {
        "input_tokens": getattr(metadata, "prompt_token_count", 0),
        "output_tokens": getattr(metadata, "candidates_token_count", 0),
        "total_tokens": getattr(metadata, "total_token_count", 0),
    }
    # Only include optional fields if non-zero
    cached = getattr(metadata, "cached_content_token_count", 0)
    if cached:
        usage["cached_tokens"] = cached
    reasoning = getattr(metadata, "thoughts_token_count", 0)
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def _extract_thinking(response: Any) -> list | None:
    """Extract thinking blocks from response.

    Returns list of ThinkingBlock dicts or None if no thinking.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return None

    thinking_blocks = []
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thinking_blocks.append({"summary": part.text})

    return thinking_blocks if thinking_blocks else None


def _extract_finish_reason(response: Any) -> str | None:
    """Extract finish_reason from response candidates.

    Returns the finish_reason string (e.g., "STOP", "MAX_TOKENS") or None
    if not available.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return None

    candidate = response.candidates[0]
    if hasattr(candidate, "finish_reason") and candidate.finish_reason:
        # Convert enum to string if needed
        reason = candidate.finish_reason
        if hasattr(reason, "name"):
            return reason.name
        return str(reason)
    return None


def _format_completion_message(finish_reason: str | None, had_tool_calls: bool) -> str:
    """Create a user-friendly completion message based on finish reason.

    Parameters
    ----------
    finish_reason
        The finish_reason from the response (e.g., "STOP", "MAX_TOKENS").
    had_tool_calls
        Whether tool calls were executed during this run.

    Returns
    -------
    str
        A concise, user-friendly completion message.
    """
    if not finish_reason:
        finish_reason = "UNKNOWN"

    # Normalize finish reason (handle both enum names and string values)
    reason = finish_reason.upper().replace("FINISHREASON.", "")

    if reason == "STOP":
        if had_tool_calls:
            return "Completed via tools."
        return "Completed."
    elif reason == "MAX_TOKENS":
        return "Reached token limit."
    elif "SAFETY" in reason:
        return "Blocked by safety filters."
    elif reason == "RECITATION":
        return "Stopped due to recitation."
    elif "TOOL" in reason or "FUNCTION" in reason:
        # UNEXPECTED_TOOL_CALL, MALFORMED_FUNCTION_CALL, etc.
        return "Tool execution incomplete."
    else:
        # Unknown reason - include it for debugging
        return f"Completed ({reason.lower()})."


def _log_empty_response_diagnostics(
    response: Any, finish_reason: str | None, had_tool_calls: bool
) -> None:
    """Log diagnostic information when response.text is empty.

    Helps debug intermittent empty response issues with Gemini models.
    """
    # Build diagnostic info
    diag = {
        "finish_reason": finish_reason,
        "had_tool_calls": had_tool_calls,
        "has_candidates": hasattr(response, "candidates") and bool(response.candidates),
    }

    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        diag["has_content"] = candidate.content is not None
        if candidate.content:
            diag["has_parts"] = bool(getattr(candidate.content, "parts", None))
            if hasattr(candidate.content, "parts") and candidate.content.parts:
                diag["num_parts"] = len(candidate.content.parts)
                # Check what types of parts we have
                part_types = []
                for part in candidate.content.parts:
                    if getattr(part, "thought", False):
                        part_types.append("thinking")
                    elif getattr(part, "text", None):
                        part_types.append("text")
                    elif hasattr(part, "function_call"):
                        part_types.append("function_call")
                    elif hasattr(part, "function_response"):
                        part_types.append("function_response")
                    else:
                        part_types.append("other")
                diag["part_types"] = part_types

    # Check for AFC history (indicates tools were auto-called)
    if hasattr(response, "automatic_function_calling_history"):
        afc_history = response.automatic_function_calling_history
        diag["afc_history_length"] = len(afc_history) if afc_history else 0

    logger.info(f"Empty response.text diagnostics: {diag}")


# ---------------------------------------------------------------------------
# run_generate / run_agenerate functions
# ---------------------------------------------------------------------------


def run_generate(
    contents: str | list[Any],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = kwargs.get("client")

    client = get_or_create_client(client)
    if isinstance(contents, str):
        contents = [contents]
    elif (
        isinstance(contents, list)
        and contents
        and isinstance(contents[0], dict)
        and "role" in contents[0]
    ):
        contents = _structured_to_google_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        json_schema=json_schema,
        timeout_s=timeout_s,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return GenerateResult(
        text=_extract_response_text(response),
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


async def run_agenerate(
    contents: str | list[Any],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = kwargs.get("client")

    client = get_or_create_client(client)
    if isinstance(contents, str):
        contents = [contents]
    elif (
        isinstance(contents, list)
        and contents
        and isinstance(contents[0], dict)
        and "role" in contents[0]
    ):
        contents = _structured_to_google_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        json_schema=json_schema,
        timeout_s=timeout_s,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return GenerateResult(
        text=_extract_response_text(response),
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------


def _emit_thinking_events(
    response: Any, model: str, callback: JSONEventCallback
) -> None:
    """Extract and emit thinking events from a response.

    In the Google GenAI SDK, thinking content appears in response.candidates[].content.parts[]
    where each Part has a `thought` boolean indicating if it's thinking content, and the
    actual thinking text is in `part.text`.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return

    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            # part.thought is a boolean indicating this is thinking content
            # part.text contains the actual thinking summary
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thinking_event: ThinkingEvent = {
                    "event": "thinking",
                    "ts": now_ms(),
                    "summary": part.text,
                    "model": model,
                }
                callback.emit(thinking_event)


def _translate_gemini(
    event: dict[str, Any],
    aggregator: ThinkingAggregator,
    callback: JSONEventCallback,
    usage_out: dict[str, Any] | None = None,
    pending_tools: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Translate a Gemini CLI JSONL event into our standard Event types.

    Args:
        event: Raw JSONL event dict from the Gemini CLI.
        aggregator: ThinkingAggregator for buffering text.
        callback: JSONEventCallback for emitting events.
        usage_out: Optional mutable dict to receive usage stats from result events.
        pending_tools: Optional mutable dict tracking tool_id -> {tool, args}.

    Returns:
        The CLI session ID from init events, or None.
    """
    event_type = event.get("type")

    if event_type == "init":
        return event.get("session_id")

    if event_type == "message":
        role = event.get("role")
        if role == "user":
            return None
        if role == "assistant" and event.get("delta"):
            content = event.get("content", "")
            if content:
                aggregator.accumulate(content)
            return None
        return None

    if event_type == "tool_use":
        aggregator.flush_as_thinking(raw_events=[event])
        tool_name = event.get("tool_name", "")
        tool_id = event.get("tool_id")
        tool_args = event.get("parameters")
        if pending_tools is not None and tool_id:
            pending_tools[tool_id] = {"tool": tool_name, "args": tool_args}
        callback.emit(
            {
                "event": "tool_start",
                "tool": tool_name,
                "args": tool_args,
                "call_id": tool_id,
                "raw": safe_raw([event]),
                "ts": now_ms(),
            }
        )
        return None

    if event_type == "tool_result":
        tool_id = event.get("tool_id")
        tool_info = {}
        if pending_tools is not None and tool_id:
            tool_info = pending_tools.pop(tool_id, {})
        callback.emit(
            {
                "event": "tool_end",
                "tool": tool_info.get("tool", ""),
                "args": tool_info.get("args"),
                "call_id": tool_id,
                "result": event.get("output"),
                "raw": safe_raw([event]),
                "ts": now_ms(),
            }
        )
        return None

    if event_type == "result":
        stats = event.get("stats") or {}
        if usage_out is not None and stats:
            input_tokens = stats.get("input_tokens", 0)
            output_tokens = stats.get("output_tokens", 0)
            total_tokens = stats.get("total_tokens", 0)
            usage_out.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )
            if stats.get("cached"):
                usage_out["cached_tokens"] = stats["cached"]
            # CLI doesn't break out thinking tokens, but they're the
            # difference between total and input+output.
            reasoning = total_tokens - input_tokens - output_tokens
            if reasoning > 0:
                usage_out["reasoning_tokens"] = reasoning
        return None

    # Unknown event type — log and skip
    logger.debug("Unknown Gemini CLI event type: %s", event_type)
    return None


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with tool-calling support via Google Gemini.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    model = config.get("model", _DEFAULT_MODEL)
    session_id = config.get("session_id")
    callback = JSONEventCallback(on_event)

    try:
        # Assemble prompt from config fields
        prompt_body, system_instruction = assemble_prompt(
            config,
            sol_tool_name="run_shell_command" if not config.get("write") else None,
        )

        # Gemini CLI has no --system-prompt flag; prepend to prompt body
        if system_instruction:
            prompt_body = system_instruction + "\n\n" + prompt_body

        # Approval posture:
        #   - Write-enabled talents (coder) run unpolicied yolo: full tool registry,
        #     write_file / replace allowed.
        #   - Read-only cogitate talents run yolo + a scoped policy: full tool
        #     registry (no plan-mode stripping), but write_file / replace denied
        #     and run_shell_command narrowed to `sol` invocations.
        # Plan mode strips run_shell_command from the registry, which drove the
        # tool-name hallucination loop documented in
        # vpe/workspace/gemini-cli-tool-hallucination-research.md. Deprecated
        # --allowed-tools controls auto-approval, not availability, so it can't
        # replace the policy file for this purpose.
        cmd = [
            "gemini",
            "-p",
            "-",
            "-o",
            "stream-json",
            "--approval-mode",
            "yolo",
            "-m",
            model,
            "--sandbox=none",
        ]
        if not config.get("write"):
            cmd.extend(["--policy", str(_COGITATE_POLICY_PATH)])

        # Resume from previous session if continuing
        if session_id:
            cmd.extend(["--resume", session_id])

        # Mutable containers for translate closure
        usage: dict[str, Any] = {}
        pending_tools: dict[str, dict[str, Any]] = {}

        def translate(
            event: dict[str, Any], agg: ThinkingAggregator, cb: JSONEventCallback
        ) -> str | None:
            return _translate_gemini(event, agg, cb, usage, pending_tools)

        aggregator = ThinkingAggregator(callback, model=model)
        cwd_value = config.get("cwd")
        runner = CLIRunner(
            cmd=cmd,
            prompt_text=prompt_body,
            translate=translate,
            callback=callback,
            aggregator=aggregator,
            cwd=Path(cwd_value) if cwd_value else None,
            env=build_cogitate_env("GOOGLE_API_KEY"),
        )

        result = await runner.run()

        # Emit finish event (CLIRunner does not emit one)
        finish_event: dict[str, Any] = {
            "event": "finish",
            "result": result,
            "ts": now_ms(),
        }
        if usage:
            finish_event["usage"] = usage
        if runner.cli_session_id:
            finish_event["cli_session_id"] = runner.cli_session_id
        callback.emit(finish_event)
        return result
    except Exception as exc:
        callback.emit(
            {
                "event": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise


def list_models() -> list[dict]:
    """List available Google Gemini models.

    Returns
    -------
    list[dict]
        List of raw model info objects from the Google Gemini API.
    """
    client = get_or_create_client()
    return [m.model_dump() for m in client.models.list()]


def validate_key(api_key: str) -> dict:
    """Validate a Google API key by listing models.

    Creates a temporary client with the provided key. Never uses
    the cached client or environment variables.

    Returns {"valid": True, "backend": "aistudio"|"vertex"} or
    {"valid": False, "error": "..."}.
    """
    global _detected_backend
    try:
        # Probe backend for this specific key (always probes, bypasses cache).
        backend = _probe_backend(api_key)

        client_kwargs = {
            "api_key": api_key,
            "http_options": types.HttpOptions(timeout=10000),
            "vertexai": backend == "vertex",
        }

        client = genai.Client(**client_kwargs)
        list(client.models.list(config={"page_size": 1}))
        _detected_backend = backend  # only cache after successful validation
        return {"valid": True, "backend": backend}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_vertex_credentials(
    creds_path: str,
) -> dict:
    """Validate Vertex AI service account credentials by listing models.

    Creates a temporary client with the provided SA credentials.

    Returns {"valid": True, "email": "..."} or {"valid": False, "error": "..."}.
    """
    try:
        import json as _json

        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "credentials": creds,
            "http_options": types.HttpOptions(timeout=10000),
        }
        with open(creds_path, encoding="utf-8") as _f:
            _sa_data = _json.load(_f)
        if "project_id" in _sa_data:
            client_kwargs["project"] = _sa_data["project_id"]

        client = genai.Client(**client_kwargs)
        list(client.models.list(config={"page_size": 1}))
        return {"valid": True, "email": creds.service_account_email}
    except Exception as e:
        return {"valid": False, "error": str(e)}


__all__ = [
    "run_cogitate",
    "run_generate",
    "run_agenerate",
    "get_or_create_client",
    "validate_vertex_credentials",
    "_detect_backend",
    "_get_effective_backend",
    "list_models",
    "validate_key",
    "validate_vertex_credentials",
]
