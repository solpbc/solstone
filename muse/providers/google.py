#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini provider for agents and direct LLM generation.

This module provides the Google Gemini provider for the ``sol agents`` CLI
and standardized generate/agenerate functions for direct LLM calls.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import errors as google_errors
from google.genai import types

from muse.models import GEMINI_FLASH
from think.utils import create_mcp_client

from ..agents import JSONEventCallback, ThinkingEvent

_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_MODEL = GEMINI_FLASH
_MAX_TOOL_ITERATIONS = 25  # Maximum agentic loop iterations before forcing a response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client and helper functions for generate/agenerate
# ---------------------------------------------------------------------------


def get_or_create_client(client: Optional[genai.Client] = None) -> genai.Client:
    """Get existing client or create new one.

    Parameters
    ----------
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one
        using GOOGLE_API_KEY from environment.

    Returns
    -------
    genai.Client
        The provided client or a newly created one.
    """
    if client is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(api_key=api_key)
    return client


def _normalize_contents(
    contents: Union[str, List[Any], List[types.Content]],
) -> List[Any]:
    """Normalize contents to list format."""
    if isinstance(contents, str):
        return [contents]
    return contents


def _build_generate_config(
    temperature: float,
    max_output_tokens: int,
    system_instruction: Optional[str],
    json_output: bool,
    thinking_budget: Optional[int],
    cached_content: Optional[str],
    timeout_s: Optional[float] = None,
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig.

    Note: Gemini's max_output_tokens is actually the total budget (thinking + output).
    We compute this internally: total = max_output_tokens + thinking_budget.
    """
    # Compute total tokens: output + thinking budget
    total_tokens = max_output_tokens + (thinking_budget or 0)

    config_args: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": total_tokens,
    }

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    if json_output:
        config_args["response_mime_type"] = "application/json"

    if thinking_budget:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    if cached_content:
        config_args["cached_content"] = cached_content

    if timeout_s:
        # Convert seconds to milliseconds for the SDK
        timeout_ms = int(timeout_s * 1000)
        config_args["http_options"] = types.HttpOptions(timeout=timeout_ms)

    return types.GenerateContentConfig(**config_args)


def _validate_response(
    response: Any, max_output_tokens: int, thinking_budget: Optional[int] = None
) -> str:
    """Validate response and extract text."""
    if response is None or response.text is None:
        # Try to extract text from candidates if available
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check for finish reason to understand why we got no text
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)
                if "MAX_TOKENS" in finish_reason:
                    total_tokens = max_output_tokens + (thinking_budget or 0)
                    raise ValueError(
                        f"Model hit token limit ({total_tokens} total = {max_output_tokens} output + "
                        f"{thinking_budget or 0} thinking) before producing output. "
                        f"Try increasing max_output_tokens or reducing thinking_budget."
                    )
                elif "SAFETY" in finish_reason:
                    raise ValueError(
                        f"Response blocked by safety filters: {finish_reason}"
                    )
                elif "STOP" not in finish_reason:
                    raise ValueError(f"Response failed with reason: {finish_reason}")

            # Try to extract text from parts if available
            if (
                hasattr(candidate, "content")
                and hasattr(candidate.content, "parts")
                and candidate.content.parts
            ):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text

        # If we still don't have text, raise an error with details
        error_msg = "No text in response"
        if response:
            if hasattr(response, "candidates") and not response.candidates:
                error_msg = "No candidates in response"
            elif hasattr(response, "prompt_feedback"):
                error_msg = f"Response issue: {response.prompt_feedback}"
        raise ValueError(error_msg)

    return response.text


# ---------------------------------------------------------------------------
# Standardized generate/agenerate functions
# ---------------------------------------------------------------------------


def generate(
    contents: Union[str, List[Any]],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Generate text using Google Gemini.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    model : str
        Model name to use.
    temperature : float
        Temperature for generation.
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
    context : str, optional
        Context string for token usage logging.
    **kwargs
        Additional Google-specific options:
        - cached_content: Name of cached content to use
        - client: Existing genai.Client to reuse

    Returns
    -------
    str
        Response text from the model.
    """
    from muse.models import log_token_usage

    cached_content = kwargs.get("cached_content")
    client = kwargs.get("client")

    client = get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout_s=timeout_s,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response, context=context)
    return text


async def agenerate(
    contents: Union[str, List[Any]],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Async generate text using Google Gemini.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    model : str
        Model name to use.
    temperature : float
        Temperature for generation.
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
    context : str, optional
        Context string for token usage logging.
    **kwargs
        Additional Google-specific options:
        - cached_content: Name of cached content to use
        - client: Existing genai.Client to reuse

    Returns
    -------
    str
        Response text from the model.
    """
    from muse.models import log_token_usage

    cached_content = kwargs.get("cached_content")
    client = kwargs.get("client")

    client = get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout_s=timeout_s,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response, context=context)
    return text


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------


def _emit_thinking_events(
    response: Any, model: str, callback: JSONEventCallback
) -> None:
    """Extract and emit thinking events from a response."""
    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, "thought") and candidate.thought:
                thinking_event: ThinkingEvent = {
                    "event": "thinking",
                    "ts": int(time.time() * 1000),
                    "summary": candidate.thought,
                    "model": model,
                }
                callback.emit(thinking_event)

    # Also check for thinking at the response level
    if hasattr(response, "thought") and response.thought:
        thinking_event: ThinkingEvent = {
            "event": "thinking",
            "ts": int(time.time() * 1000),
            "summary": response.thought,
            "model": model,
        }
        callback.emit(thinking_event)


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(
        self,
        writer: JSONEventCallback,
        agent_id: str | None = None,
        persona: str | None = None,
    ) -> None:
        self.writer = writer
        self._counter = 0
        self.session = None
        self.agent_id = agent_id
        self.persona = persona

    def attach(self, session: Any) -> None:
        self.session = session
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self._counter += 1
            call_id = f"{name}-{self._counter}"
            self.writer.emit(
                {
                    "event": "tool_start",
                    "tool": name,
                    "args": arguments,
                    "call_id": call_id,
                }
            )

            # Build _meta dict for passing agent identity
            meta = {}
            if self.agent_id:
                meta["agent_id"] = self.agent_id
            if self.persona:
                meta["persona"] = self.persona

            result = await original(
                name=name,
                arguments=arguments,
                meta=meta,
                **kwargs,
            )

            # Extract content from CallToolResult if needed
            if hasattr(result, "content"):
                # MCP CallToolResult object - extract text from TextContent objects
                if isinstance(result.content, list):
                    # Handle array of content items
                    extracted_content = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            # TextContent object - extract the text
                            extracted_content.append(item.text)
                        else:
                            # Other content types - keep as is
                            extracted_content.append(item)
                    # If single text content, return as string, otherwise as list
                    result_data = (
                        extracted_content[0]
                        if len(extracted_content) == 1
                        else extracted_content
                    )
                else:
                    result_data = result.content
            else:
                # Direct result (dict, string, etc.)
                result_data = result

            self.writer.emit(
                {
                    "event": "tool_end",
                    "tool": name,
                    "args": arguments,
                    "result": result_data,
                    "call_id": call_id,
                }
            )
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


async def run_agent(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Run a single prompt through the Google Gemini agent and return the response.

    Args:
        config: Complete configuration dictionary including prompt, instruction, model, etc.
        on_event: Optional event callback
    """
    # Extract values from unified config
    prompt = config.get("prompt", "")
    if not prompt:
        raise ValueError("Missing 'prompt' in config")

    # Model is required - Cortex always provides it via resolve_provider()
    model = config.get("model")
    if not model:
        raise ValueError("Missing 'model' in config - should be set by Cortex")

    max_tokens = config.get("max_tokens", _DEFAULT_MAX_TOKENS)
    disable_mcp = config.get("disable_mcp", False)
    persona = config.get("persona", "default")

    callback = JSONEventCallback(on_event)

    try:
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
                "provider": "google",
            }
        )

        # Extract instruction and extra_context from config
        system_instruction = config.get("instruction", "")
        first_user = config.get("extra_context", "")

        # Build history - check for continuation first
        continue_from = config.get("continue_from")
        if continue_from:
            # Load previous conversation history using shared function
            from ..agents import parse_agent_events_to_turns

            turns = parse_agent_events_to_turns(continue_from)
            # Convert to Google's format
            history = []
            for turn in turns:
                role = "model" if turn["role"] == "assistant" else turn["role"]
                history.append(
                    types.Content(role=role, parts=[types.Part(text=turn["content"])])
                )
        else:
            # Fresh conversation
            history = []
            if first_user:
                history.append(
                    types.Content(role="user", parts=[types.Part(text=first_user)])
                )

        # Create client
        client = genai.Client(api_key=api_key)

        # Create fresh chat session
        chat = client.aio.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            history=history,
        )

        # Configure tools based on disable_mcp flag
        if not disable_mcp:
            mcp_url = config.get("mcp_server_url")
            if not mcp_url:
                raise RuntimeError("MCP server URL not provided in config")

            # Create MCP client and attach hooks
            async with create_mcp_client(str(mcp_url)) as mcp:
                # Attach tool logging hooks to the MCP session
                agent_id = config.get("agent_id")
                tool_hooks = ToolLoggingHooks(
                    callback, agent_id=agent_id, persona=persona
                )
                tool_hooks.attach(mcp.session)

                # Extract allowed tools from config
                allowed_tools = config.get("tools", None)

                # Configure function calling mode based on tool filtering
                if allowed_tools and isinstance(allowed_tools, list):
                    logger.info(f"Filtering tools to: {allowed_tools}")
                    function_calling_config = types.FunctionCallingConfig(
                        mode="ANY",  # Restrict to only allowed functions
                        allowed_function_names=allowed_tools,
                    )
                else:
                    function_calling_config = types.FunctionCallingConfig(mode="AUTO")

                cfg = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    tools=[mcp.session],
                    tool_config=types.ToolConfig(
                        function_calling_config=function_calling_config
                    ),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=-1,  # Enable dynamic thinking
                    ),
                )

                # Agentic loop - continue until model produces text or hits iteration limit
                current_message = prompt
                for iteration in range(_MAX_TOOL_ITERATIONS):
                    response = await chat.send_message(current_message, config=cfg)

                    # Extract thinking from each iteration
                    _emit_thinking_events(response, model, callback)

                    # Check if we got a text response
                    if response.text:
                        break

                    # No text - check if model wants more tool calls
                    function_calls = getattr(response, "function_calls", None)
                    if not function_calls:
                        # No text and no function calls - unexpected state
                        logger.warning(
                            f"Iteration {iteration + 1}: No text and no function calls in response"
                        )
                        break

                    # Model returned function calls - the SDK's automatic function calling
                    # should have handled them, but if we're here with no text, we need
                    # to prompt the model to continue. This can happen when the SDK's
                    # AFC exits but the model's last turn was a tool call.
                    logger.debug(
                        f"Iteration {iteration + 1}: Got {len(function_calls)} function calls, continuing"
                    )
                    current_message = "Please continue with the task or provide a summary if complete."
                else:
                    # Hit iteration limit - ask model to summarize
                    logger.warning(
                        f"Hit max iterations ({_MAX_TOOL_ITERATIONS}), requesting summary"
                    )
                    response = await chat.send_message(
                        "Please provide a summary of what was accomplished.",
                        config=types.GenerateContentConfig(
                            max_output_tokens=max_tokens
                        ),
                    )
                    _emit_thinking_events(response, model, callback)
        else:
            # No MCP tools - just basic config
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=-1,  # Enable dynamic thinking
                ),
            )

            response = await chat.send_message(prompt, config=cfg)
            _emit_thinking_events(response, model, callback)

        text = response.text
        if not text:
            # Provide more context about why response might be empty
            finish_reason = None
            function_calls = getattr(response, "function_calls", None)
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason)

            detail_parts = []
            if finish_reason:
                detail_parts.append(f"finish_reason={finish_reason}")
            if function_calls:
                detail_parts.append(f"pending_function_calls={len(function_calls)}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            raise RuntimeError(f"Model returned empty response{detail}")

        # Extract usage from response
        usage_dict = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            usage_dict = {
                "input_tokens": getattr(metadata, "prompt_token_count", 0),
                "output_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0),
            }
            # Only include optional fields if non-zero
            cached = getattr(metadata, "cached_content_token_count", 0)
            if cached:
                usage_dict["cached_tokens"] = cached
            reasoning = getattr(metadata, "thoughts_token_count", 0)
            if reasoning:
                usage_dict["reasoning_tokens"] = reasoning

        callback.emit(
            {
                "event": "finish",
                "result": text,
                "usage": usage_dict,
                "ts": int(time.time() * 1000),
            }
        )
        return text
    except google_errors.ServerError as exc:
        # Google API server error (5xx) - transient, may be retried
        error_msg = f"Google API server error: {exc}"
        logger.warning(error_msg)
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "error_type": "server_error",
                "retryable": True,
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise RuntimeError(error_msg) from exc
    except google_errors.ClientError as exc:
        # Google API client error (4xx) - likely a config or request issue
        error_msg = f"Google API client error: {exc}"
        logger.error(error_msg)
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "error_type": "client_error",
                "retryable": False,
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise RuntimeError(error_msg) from exc
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


__all__ = [
    "run_agent",
    "generate",
    "agenerate",
    "get_or_create_client",
]
