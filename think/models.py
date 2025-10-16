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

GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-1"
CLAUDE_SONNET_4 = "claude-sonnet-4-5-20250929"


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
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig."""
    config_args = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
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

    return types.GenerateContentConfig(**config_args)


def _validate_response(response, max_output_tokens: int) -> str:
    """Validate response and extract text."""
    if response is None or response.text is None:
        # Try to extract text from candidates if available
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check for finish reason to understand why we got no text
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)
                if "MAX_TOKENS" in finish_reason:
                    raise ValueError(
                        f"Model hit max_output_tokens limit ({max_output_tokens}) before producing output. "
                        f"Try increasing max_output_tokens."
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
    usage: Dict[str, Any],
    context: Optional[str] = None,
) -> None:
    """Log token usage to journal with unified schema.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash")
    usage : dict
        Usage data in provider-specific format. Supports:
        - OpenAI format: {input_tokens, output_tokens, total_tokens,
                         details: {input: {cached_tokens}, output: {reasoning_tokens}}}
        - Gemini format: {prompt_token_count, candidates_token_count,
                         cached_content_token_count, thoughts_token_count, total_token_count}
        - Unified format: {input_tokens, output_tokens, total_tokens,
                          cached_tokens, reasoning_tokens, requests}
    context : str, optional
        Context string (e.g., "module.function:123" or "agent.persona.id").
        If None, auto-detects from call stack.
    """
    try:
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return

        # Auto-detect calling context if not provided
        if context is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_frame = frame.f_back
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                func_name = caller_frame.f_code.co_name
                line_num = caller_frame.f_lineno

                # Clean up module name
                for prefix in ["think.", "hear.", "see.", "convey.", "muse."]:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix):]
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
                    normalized_usage["reasoning_tokens"] = output_details["reasoning_tokens"]

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


def _log_token_usage(response, model: str) -> None:
    """Log Gemini token usage to journal (legacy wrapper)."""
    if hasattr(response, "usage_metadata"):
        try:
            usage = response.usage_metadata
            usage_dict = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "cached_content_token_count": getattr(usage, "cached_content_token_count", 0),
                "thoughts_token_count": getattr(usage, "thoughts_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
            }

            # Use unified logging function
            log_token_usage(model=model, usage=usage_dict)

        except Exception:
            # Silently fail - logging shouldn't break the main flow
            pass


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
        Maximum output tokens (default: 8192 * 2)
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.

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
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens)
    _log_token_usage(response, model)
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
        Maximum output tokens (default: 8192 * 2)
    system_instruction : str, optional
        System instruction for the model
    json_output : bool
        Whether to request JSON response format (default: False)
    thinking_budget : int, optional
        Token budget for model thinking
    cached_content : str, optional
        Name of cached content to use
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one.

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
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = _validate_response(response, max_output_tokens)
    _log_token_usage(response, model)
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
]
