import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_LITE = "gemini-2.5-flash-lite"

GPT_5 = "gpt-5.1"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-5"
CLAUDE_SONNET_4 = "claude-sonnet-4-5"
CLAUDE_HAIKU_4 = "claude-haiku-4-5"


def _get_or_create_client(client: Optional[genai.Client]) -> genai.Client:
    """Get existing client or create new one."""
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
    timeout: Optional[int] = None,
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig.

    Note: Gemini's max_output_tokens is actually the total budget (thinking + output).
    We compute this internally: total = max_output_tokens + thinking_budget.
    """
    # Compute total tokens: output + thinking budget
    total_tokens = max_output_tokens + (thinking_budget or 0)

    config_args = {
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

    if timeout:
        config_args["http_options"] = types.HttpOptions(timeout=timeout)

    return types.GenerateContentConfig(**config_args)


def _validate_response(
    response, max_output_tokens: int, thinking_budget: Optional[int] = None
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


def log_token_usage(
    model: str,
    usage: Union[Dict[str, Any], Any],
    context: Optional[str] = None,
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
    """
    try:
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return

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
                for prefix in ["think.", "hear.", "see.", "convey.", "muse."]:
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
    contents: Union[str, List[Any], List[types.Content]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[genai.Client] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Simplified wrapper for genai.models.generate_content with common defaults.

    Parameters
    ----------
    contents : str, List, or List[types.Content]
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
    timeout : int, optional
        Request timeout in milliseconds. Minimum is 10000 (10 seconds).

    Returns
    -------
    str
        Response text from the model
    """
    client = _get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout=timeout,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response)
    return text


async def gemini_agenerate(
    contents: Union[str, List[Any], List[types.Content]],
    model: str = GEMINI_FLASH,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    cached_content: Optional[str] = None,
    client: Optional[genai.Client] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Async wrapper for genai.aio.models.generate_content with common defaults.

    Parameters
    ----------
    contents : str, List, or List[types.Content]
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
    timeout : int, optional
        Request timeout in milliseconds. Minimum is 10000 (10 seconds).

    Returns
    -------
    str
        Response text from the model
    """
    client = _get_or_create_client(client)
    contents = _normalize_contents(contents)
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        cached_content=cached_content,
        timeout=timeout,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens, thinking_budget)
    log_token_usage(model=model, usage=response)
    return text


__all__ = [
    "GEMINI_PRO",
    "GEMINI_FLASH",
    "GEMINI_LITE",
    "GPT_5",
    "GPT_5_MINI",
    "GPT_5_NANO",
    "CLAUDE_OPUS_4",
    "CLAUDE_SONNET_4",
    "gemini_generate",
    "gemini_agenerate",
    "log_token_usage",
    "get_model_provider",
    "calc_token_cost",
]
