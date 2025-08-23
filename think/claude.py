#!/usr/bin/env python3
"""Claude Code SDK backend agent implementation.

This module exposes agent functionality for interacting with Claude Code
via the SDK and is used by the ``think-agents`` CLI.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Callable, Optional

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    CLINotFoundError,
    ProcessError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from .agents import JSONEventCallback, ThinkingEvent
from .models import CLAUDE_SONNET_4
from .utils import agent_instructions

DEFAULT_MODEL = CLAUDE_SONNET_4
DEFAULT_MAX_TOKENS = 8096 * 2


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)
    return logging.getLogger(__name__)


async def run_agent(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through the Claude Code SDK and return the response."""
    callback = JSONEventCallback(on_event)

    try:
        # Get journal path for file permissions
        journal_path = os.getenv("JOURNAL_PATH")
        if not journal_path:
            raise RuntimeError("JOURNAL_PATH not set")

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
                "backend": "claude",
            }
        )

        # Get persona instructions
        system_instruction, first_user, _ = agent_instructions(persona)

        # Combine prompts
        full_prompt = []
        if first_user:
            full_prompt.append(first_user)
        full_prompt.append(prompt)
        combined_prompt = "\n\n".join(full_prompt)

        # Configure Claude Code options
        options = ClaudeCodeOptions(
            system_prompt=system_instruction,
            model=model,
            # Allow file operations and git commands in journal directory
            allowed_tools=[
                f"Read({journal_path}/**)",
                f"Write({journal_path}/**)",
                f"Edit({journal_path}/**)",
                f"MultiEdit({journal_path}/**)",
                f"LS({journal_path}/**)",
                f"Glob({journal_path}/**)",
                "Bash(git:*)",
                "Bash(ls:*)",
                "Bash(cat:*)",
                "Bash(mkdir:*)",
                "Bash(rm:*)",
                "Bash(pwd)",
                "Bash(echo:*)",
            ],
            disallowed_tools=["mcp_*"],  # Disable MCP tools
            permission_mode="acceptEdits",  # Auto-accept file edits
            max_turns=3,  # Allow multiple turns for complex tasks
        )

        # Track tool calls for pairing start/end events
        tool_calls = {}
        response_text = []

        # Stream responses from Claude Code
        async for message in query(prompt=combined_prompt, options=options):
            if isinstance(message, AssistantMessage):
                # Process each content block in the assistant's message
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Regular text response
                        response_text.append(block.text)

                    elif isinstance(block, ToolUseBlock):
                        # Tool being called
                        tool_id = getattr(block, "id", str(time.time()))
                        tool_name = getattr(block, "name", "unknown")
                        tool_input = getattr(block, "input", {})

                        tool_calls[tool_id] = {
                            "name": tool_name,
                            "input": tool_input,
                        }

                        callback.emit(
                            {
                                "event": "tool_start",
                                "tool": tool_name,
                                "args": tool_input,
                                "call_id": tool_id,
                            }
                        )

                    elif isinstance(block, ToolResultBlock):
                        # Tool result received
                        tool_id = getattr(block, "tool_use_id", None)
                        content = getattr(block, "content", "")

                        if tool_id and tool_id in tool_calls:
                            tool_info = tool_calls[tool_id]
                            callback.emit(
                                {
                                    "event": "tool_end",
                                    "tool": tool_info["name"],
                                    "args": tool_info["input"],
                                    "result": content,
                                    "call_id": tool_id,
                                }
                            )

                    # Handle other block types if needed
                    elif hasattr(block, "thinking"):
                        # Thinking/reasoning block
                        thinking_content = getattr(block, "thinking", "")
                        if thinking_content:
                            thinking_event: ThinkingEvent = {
                                "event": "thinking",
                                "ts": int(time.time() * 1000),
                                "summary": thinking_content,
                                "model": model,
                            }
                            callback.emit(thinking_event)

            elif isinstance(message, UserMessage):
                # User message in conversation (shouldn't happen in our case)
                pass

            # Handle other message types or raw events
            elif hasattr(message, "__dict__"):
                # Check for streaming events or other message types
                msg_dict = message.__dict__ if hasattr(message, "__dict__") else {}

                # Look for tool events in the message structure
                if msg_dict.get("type") == "tool_use":
                    tool_id = msg_dict.get("id", str(time.time()))
                    tool_name = msg_dict.get("name", "unknown")
                    tool_input = msg_dict.get("input", {})

                    tool_calls[tool_id] = {
                        "name": tool_name,
                        "input": tool_input,
                    }

                    callback.emit(
                        {
                            "event": "tool_start",
                            "tool": tool_name,
                            "args": tool_input,
                            "call_id": tool_id,
                        }
                    )

                elif msg_dict.get("type") == "tool_result":
                    tool_id = msg_dict.get("tool_use_id")
                    content = msg_dict.get("content", "")

                    if tool_id and tool_id in tool_calls:
                        tool_info = tool_calls[tool_id]
                        callback.emit(
                            {
                                "event": "tool_end",
                                "tool": tool_info["name"],
                                "args": tool_info["input"],
                                "result": content,
                                "call_id": tool_id,
                            }
                        )

        # Combine all response text
        final_text = "".join(response_text).strip()

        callback.emit({"event": "finish", "result": final_text})
        return final_text

    except CLINotFoundError:
        error_msg = "Claude Code CLI not found. Please install with: npm install -g @anthropic-ai/claude-code"
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "trace": traceback.format_exc(),
            }
        )
        raise RuntimeError(error_msg)

    except ProcessError as e:
        error_msg = (
            f"Claude Code process failed with exit code {e.exit_code}: {e.stderr}"
        )
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "trace": traceback.format_exc(),
            }
        )
        raise RuntimeError(error_msg)

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


async def run_prompt(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` (alias for run_agent)."""
    return await run_agent(
        prompt, model=model, max_tokens=max_tokens, on_event=on_event, persona=persona
    )


__all__ = [
    "run_agent",
    "run_prompt",
    "setup_logging",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
