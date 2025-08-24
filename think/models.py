import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_LITE = "gemini-2.5-flash-lite"

GPT_5 = "gpt-5"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-20250514"
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
CLAUDE_HAIKU_3_5 = "claude-3-5-haiku-latest"


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
    # Get or create client
    if client is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(api_key=api_key)

    # Normalize contents to list if it's a plain string
    # But don't touch it if it's already a list (could contain Parts or Content objects)
    if isinstance(contents, str):
        contents = [contents]

    # Build config
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

    # Make the API call
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_args),
    )

    # Check if response is valid and has text
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

    # Log token usage if we have usage metadata
    if hasattr(response, "usage_metadata"):
        try:
            journal = os.getenv("JOURNAL_PATH")
            if journal:
                # Auto-detect calling context
                context = None
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_frame = frame.f_back
                    module_name = caller_frame.f_globals.get("__name__", "unknown")
                    func_name = caller_frame.f_code.co_name
                    line_num = caller_frame.f_lineno

                    # Clean up module name
                    for prefix in ["think.", "hear.", "see.", "dream."]:
                        if module_name.startswith(prefix):
                            module_name = module_name[len(prefix) :]
                            break

                    context = f"{module_name}.{func_name}:{line_num}"

                # Build token log entry
                usage = response.usage_metadata
                token_data = {
                    "timestamp": time.time(),
                    "timestamp_str": time.strftime("%Y%m%d_%H%M%S"),
                    "model": model,
                    "context": context,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "candidates_tokens": getattr(
                            usage, "candidates_token_count", 0
                        ),
                        "cached_tokens": getattr(
                            usage, "cached_content_token_count", 0
                        ),
                        "thoughts_tokens": getattr(usage, "thoughts_token_count", 0),
                        "total_tokens": getattr(usage, "total_token_count", 0),
                    },
                }

                # Save to journal/tokens/<timestamp>.json
                tokens_dir = Path(journal) / "tokens"
                tokens_dir.mkdir(exist_ok=True)

                filename = f"{int(time.time() * 1000)}.json"
                filepath = tokens_dir / filename

                with open(filepath, "w") as f:
                    json.dump(token_data, f, indent=2)

        except Exception:
            # Silently fail - logging shouldn't break the main flow
            pass

    return response.text


__all__ = [
    "GEMINI_PRO",
    "GEMINI_FLASH",
    "GEMINI_LITE",
    "GPT_5",
    "GPT_5_MINI",
    "GPT_5_NANO",
    "CLAUDE_OPUS_4",
    "CLAUDE_SONNET_4",
    "CLAUDE_HAIKU_3_5",
    "gemini_generate",
]
