# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from think.utils import get_journal

GEMINI_FLASH = "gemini-3-flash-preview"
GEMINI_PRO = "gemini-3-pro-preview"
GEMINI_LITE = "gemini-2.5-flash-lite"

# Mapping from config string names to model constants
GEMINI_MODEL_NAMES = {
    "lite": GEMINI_LITE,
    "flash": GEMINI_FLASH,
    "pro": GEMINI_PRO,
}

GPT_5 = "gpt-5.2"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-5"
CLAUDE_SONNET_4 = "claude-sonnet-4-5"
CLAUDE_HAIKU_4 = "claude-haiku-4-5"


def resolve_provider(context: str) -> tuple[str, str]:
    """Resolve context to provider and model based on configuration.

    Matches context against configured contexts using exact match first,
    then glob patterns (via fnmatch), falling back to defaults.

    Parameters
    ----------
    context
        Context string (e.g., "observe.describe.frame", "insight.meetings").

    Returns
    -------
    tuple[str, str]
        (provider_name, model) tuple. Provider is one of "google", "openai",
        "anthropic". Model is the full model identifier string.
    """
    import fnmatch

    from think.utils import get_config

    config = get_config()
    providers = config.get("providers", {})

    # Get defaults
    default = providers.get("default", {"provider": "google", "model": GEMINI_FLASH})
    default_provider = default.get("provider", "google")
    default_model = default.get("model", GEMINI_FLASH)

    contexts = providers.get("contexts", {})
    if not contexts or not context:
        return (default_provider, default_model)

    # Check for exact match first
    if context in contexts:
        match = contexts[context]
        return (
            match.get("provider", default_provider),
            match.get("model", default_model),
        )

    # Check glob patterns - most specific (longest non-wildcard prefix) wins
    matches = []
    for pattern, match_config in contexts.items():
        if fnmatch.fnmatch(context, pattern):
            # Calculate specificity: length of pattern before first wildcard
            specificity = len(pattern.split("*")[0])
            matches.append((specificity, pattern, match_config))

    if matches:
        # Sort by specificity descending, take the most specific match
        matches.sort(key=lambda x: x[0], reverse=True)
        _, _, match_config = matches[0]
        return (
            match_config.get("provider", default_provider),
            match_config.get("model", default_model),
        )

    return (default_provider, default_model)


def get_or_create_client(client: Optional[Any] = None) -> Any:
    """Get existing Gemini client or create new one.

    This is a backward-compatibility wrapper that delegates to muse.google.

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
    from muse.google import get_or_create_client as _get_or_create_client

    return _get_or_create_client(client)


def log_token_usage(
    model: str,
    usage: Union[Dict[str, Any], Any],
    context: Optional[str] = None,
    segment: Optional[str] = None,
) -> None:
    """Log token usage to journal with unified schema.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash")
    usage : dict or response object
        Usage data in provider-specific format, OR a Gemini response object.
        Dict formats supported:
        - OpenAI format: {input_tokens, output_tokens, total_tokens,
                         details: {input: {cached_tokens}, output: {reasoning_tokens}}}
        - Gemini format: {prompt_token_count, candidates_token_count,
                         cached_content_token_count, thoughts_token_count, total_token_count}
        - Unified format: {input_tokens, output_tokens, total_tokens,
                          cached_tokens, reasoning_tokens, requests}
        Response objects: Gemini GenerateContentResponse with usage_metadata attribute
    context : str, optional
        Context string (e.g., "module.function:123" or "agent.persona.id").
        If None, auto-detects from call stack.
    segment : str, optional
        Segment key (e.g., "143022_300") for attribution.
        If None, falls back to SEGMENT_KEY environment variable.
    """
    try:
        journal = get_journal()

        # Extract from Gemini response object if needed
        if hasattr(usage, "usage_metadata"):
            try:
                metadata = usage.usage_metadata
                usage = {
                    "prompt_token_count": getattr(metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(
                        metadata, "candidates_token_count", 0
                    ),
                    "cached_content_token_count": getattr(
                        metadata, "cached_content_token_count", 0
                    ),
                    "thoughts_token_count": getattr(
                        metadata, "thoughts_token_count", 0
                    ),
                    "total_token_count": getattr(metadata, "total_token_count", 0),
                }
            except Exception:
                return  # Can't extract, fail silently

        # Auto-detect calling context if not provided
        if context is None:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None

            # Skip frames that contain "gemini" in function name
            while caller_frame and "gemini" in caller_frame.f_code.co_name.lower():
                caller_frame = caller_frame.f_back

            if caller_frame:
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                func_name = caller_frame.f_code.co_name
                line_num = caller_frame.f_lineno

                # Clean up module name
                for prefix in ["think.", "observe.", "convey.", "muse."]:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]
                        break

                context = f"{module_name}.{func_name}:{line_num}"

        # Normalize usage data to unified schema
        normalized_usage: Dict[str, int] = {}

        # Handle OpenAI format with nested details
        if "input_tokens" in usage or "output_tokens" in usage:
            normalized_usage["input_tokens"] = usage.get("input_tokens", 0)
            normalized_usage["output_tokens"] = usage.get("output_tokens", 0)
            normalized_usage["total_tokens"] = usage.get("total_tokens", 0)

            # Extract nested details
            details = usage.get("details", {})
            if details:
                input_details = details.get("input", {})
                if input_details and input_details.get("cached_tokens"):
                    normalized_usage["cached_tokens"] = input_details["cached_tokens"]

                output_details = details.get("output", {})
                if output_details and output_details.get("reasoning_tokens"):
                    normalized_usage["reasoning_tokens"] = output_details[
                        "reasoning_tokens"
                    ]

            # Optional requests field for OpenAI
            if "requests" in usage and usage["requests"] is not None:
                normalized_usage["requests"] = usage["requests"]

            # Pass through Anthropic cache fields if present
            if usage.get("cached_tokens"):
                normalized_usage["cached_tokens"] = usage["cached_tokens"]
            if usage.get("cache_creation_tokens"):
                normalized_usage["cache_creation_tokens"] = usage[
                    "cache_creation_tokens"
                ]

        # Handle Gemini format
        elif "prompt_token_count" in usage or "candidates_token_count" in usage:
            normalized_usage["input_tokens"] = usage.get("prompt_token_count", 0)
            normalized_usage["output_tokens"] = usage.get("candidates_token_count", 0)
            normalized_usage["total_tokens"] = usage.get("total_token_count", 0)

            if usage.get("cached_content_token_count"):
                normalized_usage["cached_tokens"] = usage["cached_content_token_count"]
            if usage.get("thoughts_token_count"):
                normalized_usage["reasoning_tokens"] = usage["thoughts_token_count"]

        # Already in unified format
        else:
            normalized_usage = {k: v for k, v in usage.items() if isinstance(v, int)}

        # Build token log entry
        token_data = {
            "timestamp": time.time(),
            "model": model,
            "context": context,
            "usage": normalized_usage,
        }

        # Add segment: prefer parameter, fallback to env (set by think/insight, observe handlers)
        segment_key = segment or os.getenv("SEGMENT_KEY")
        if segment_key:
            token_data["segment"] = segment_key

        # Save to journal/tokens/<YYYYMMDD>.jsonl (one file per day)
        tokens_dir = Path(journal) / "tokens"
        tokens_dir.mkdir(exist_ok=True)

        filename = time.strftime("%Y%m%d.jsonl")
        filepath = tokens_dir / filename

        # Atomic append - safe for parallel writers
        with open(filepath, "a") as f:
            f.write(json.dumps(token_data) + "\n")

    except Exception:
        # Silently fail - logging shouldn't break the main flow
        pass


def get_model_provider(model: str) -> str:
    """Get the provider name from a model identifier.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash", "claude-sonnet-4-5")

    Returns
    -------
    str
        Provider name: "openai", "google", "anthropic", or "unknown"
    """
    model_lower = model.lower()

    if model_lower.startswith("gpt"):
        return "openai"
    elif model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("claude"):
        return "anthropic"
    else:
        return "unknown"


def calc_token_cost(token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculate cost for a token usage record.

    Parameters
    ----------
    token_data : dict
        Token usage record from journal logs with structure:
        {
            "model": "gemini-2.5-flash",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 500,
                "cached_tokens": 800,
                "reasoning_tokens": 200,
                ...
            }
        }

    Returns
    -------
    dict or None
        Cost breakdown:
        {
            "total_cost": 0.00123,
            "input_cost": 0.00075,
            "output_cost": 0.00048,
            "currency": "USD"
        }
        Returns None if pricing unavailable or calculation fails.
    """
    try:
        from genai_prices import Usage, calc_price

        model = token_data.get("model")
        usage_data = token_data.get("usage", {})

        if not model or not usage_data:
            return None

        # Get provider ID
        provider_id = get_model_provider(model)
        if provider_id == "unknown":
            return None

        # Map our token fields to genai_prices Usage format
        # Note: Gemini reports reasoning_tokens separately, but they're billed at
        # output token rates. genai-prices doesn't have a separate field for reasoning,
        # so we add them to output_tokens for correct pricing.
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cached_tokens = usage_data.get("cached_tokens", 0)
        reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        # Add reasoning tokens to output for pricing (Gemini bills them as output)
        total_output_tokens = output_tokens + reasoning_tokens

        # Create Usage object
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=total_output_tokens,
            cache_read_tokens=cached_tokens if cached_tokens > 0 else None,
        )

        # Calculate price
        result = calc_price(
            usage=usage,
            model_ref=model,
            provider_id=provider_id,
        )

        # Return simplified cost breakdown
        return {
            "total_cost": float(result.total_price),
            "input_cost": float(result.input_price),
            "output_cost": float(result.output_price),
            "currency": "USD",
        }

    except Exception:
        # Silently fail if pricing unavailable
        return None


def gemini_generate(
    contents: Union[str, List[Any]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[Any] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
) -> str:
    """
    Simplified wrapper for Google Gemini generation with common defaults.

    This is a backward-compatibility wrapper that delegates to muse.google.generate().

    Parameters
    ----------
    contents : str or List
        The content to send to the model. Can be:
        - A string (will be converted to a list with one string)
        - A list of strings, types.Part objects, or mixed content
        - A list of types.Content objects for complex conversations
    model : str
        Model name to use (default: GEMINI_FLASH)
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for the model's response output (default: 8192 * 2).
        Note: This is the output budget only. The total token budget sent to
        Gemini's API is computed as max_output_tokens + thinking_budget.
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking. When set, the total token budget
        becomes max_output_tokens + thinking_budget.
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.
    timeout_s : float, optional
        Request timeout in seconds.
    context : str, optional
        Context string for token usage logging (e.g., "insight.decisions.markdown").
        If not provided, auto-detects from call stack.

    Returns
    -------
    str
        Response text from the model
    """
    from muse.google import generate as google_generate

    return google_generate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        context=context,
        cached_content=cached_content,
        client=client,
    )


async def gemini_agenerate(
    contents: Union[str, List[Any]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[Any] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
) -> str:
    """
    Async wrapper for Google Gemini generation with common defaults.

    This is a backward-compatibility wrapper that delegates to muse.google.agenerate().

    Parameters
    ----------
    contents : str or List
        The content to send to the model. Can be:
        - A string (will be converted to a list with one string)
        - A list of strings, types.Part objects, or mixed content
        - A list of types.Content objects for complex conversations
    model : str
        Model name to use (default: GEMINI_FLASH)
    temperature : float
        Temperature for generation (default: 0.3)
    max_output_tokens : int
        Maximum tokens for the model's response output (default: 8192 * 2).
        Note: This is the output budget only. The total token budget sent to
        Gemini's API is computed as max_output_tokens + thinking_budget.
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking. When set, the total token budget
        becomes max_output_tokens + thinking_budget.
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.
    timeout_s : float, optional
        Request timeout in seconds.
    context : str, optional
        Context string for token usage logging (e.g., "insight.decisions.markdown").
        If not provided, auto-detects from call stack.

    Returns
    -------
    str
        Response text from the model
    """
    from muse.google import agenerate as google_agenerate

    return await google_agenerate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        context=context,
        cached_content=cached_content,
        client=client,
    )


# ---------------------------------------------------------------------------
# Unified generate/agenerate with provider routing
# ---------------------------------------------------------------------------


def generate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """Generate text using the configured provider for the given context.

    Routes the request to the appropriate backend (Google, OpenAI, or Anthropic)
    based on the providers configuration in journal.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    context : str
        Context string for routing and token logging (e.g., "insight.meetings").
        This is required and determines which provider/model to use.
    temperature : float
        Temperature for generation (default: 0.3).
    max_output_tokens : int
        Maximum tokens for the model's response output.
    system_instruction : str, optional
        System instruction for the model.
    json_output : bool
        Whether to request JSON response format.
    thinking_budget : int, optional
        Token budget for model thinking (ignored by providers that don't support it).
    timeout_s : float, optional
        Request timeout in seconds.
    **kwargs
        Additional provider-specific options passed through to the backend.

    Returns
    -------
    str
        Response text from the model.

    Raises
    ------
    ValueError
        If the resolved provider is not supported.
    """

    provider, model = resolve_provider(context)

    if provider == "google":
        from muse.google import generate as backend_generate
    elif provider == "openai":
        from muse.openai import generate as backend_generate
    elif provider == "anthropic":
        from muse.anthropic import generate as backend_generate
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return backend_generate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        context=context,
        **kwargs,
    )


async def agenerate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """Async generate text using the configured provider for the given context.

    Routes the request to the appropriate backend (Google, OpenAI, or Anthropic)
    based on the providers configuration in journal.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    context : str
        Context string for routing and token logging (e.g., "insight.meetings").
        This is required and determines which provider/model to use.
    temperature : float
        Temperature for generation (default: 0.3).
    max_output_tokens : int
        Maximum tokens for the model's response output.
    system_instruction : str, optional
        System instruction for the model.
    json_output : bool
        Whether to request JSON response format.
    thinking_budget : int, optional
        Token budget for model thinking (ignored by providers that don't support it).
    timeout_s : float, optional
        Request timeout in seconds.
    **kwargs
        Additional provider-specific options passed through to the backend.

    Returns
    -------
    str
        Response text from the model.

    Raises
    ------
    ValueError
        If the resolved provider is not supported.
    """

    provider, model = resolve_provider(context)

    if provider == "google":
        from muse.google import agenerate as backend_agenerate
    elif provider == "openai":
        from muse.openai import agenerate as backend_agenerate
    elif provider == "anthropic":
        from muse.anthropic import agenerate as backend_agenerate
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return await backend_agenerate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        context=context,
        **kwargs,
    )


__all__ = [
    "GEMINI_PRO",
    "GEMINI_FLASH",
    "GEMINI_LITE",
    "GEMINI_MODEL_NAMES",
    "GPT_5",
    "GPT_5_MINI",
    "GPT_5_NANO",
    "CLAUDE_OPUS_4",
    "CLAUDE_SONNET_4",
    "CLAUDE_HAIKU_4",
    "get_or_create_client",
    "gemini_generate",
    "gemini_agenerate",
    "generate",
    "agenerate",
    "log_token_usage",
    "get_model_provider",
    "calc_token_cost",
]
