#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Claude Code SDK backend agent implementation.

This module exposes agent functionality for interacting with Claude Code
via the SDK and is used by the ``muse-agents`` CLI.

The Claude backend provides read-only access to the entire journal with
diagnostic shell commands for system analysis and health checks.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLINotFoundError,
    ProcessError,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from muse.models import CLAUDE_SONNET_4
from think.utils import get_journal

from .agents import JSONEventCallback, ThinkingEvent

# Add local claude installation to PATH if it exists
_claude_bin = Path.home() / ".claude" / "local" / "node_modules" / ".bin"
if _claude_bin.exists():
    current_path = os.environ.get("PATH", "")
    if str(_claude_bin) not in current_path:
        os.environ["PATH"] = f"{_claude_bin}:{current_path}"

_DEFAULT_MODEL = CLAUDE_SONNET_4


def _get_readonly_tools(journal_path: str) -> list[str]:
    """Return allowed tools for read-only journal access with diagnostic commands."""
    return [
        # Read-only file access to journal
        f"Read({journal_path}/**)",
        f"Glob({journal_path}/**)",
        f"LS({journal_path}/**)",
        # Diagnostic shell commands (read-only)
        "Bash(ls:*)",
        "Bash(cat:*)",
        "Bash(head:*)",
        "Bash(tail:*)",
        "Bash(grep:*)",
        "Bash(jq:*)",
        "Bash(wc:*)",
        "Bash(date:*)",
        "Bash(pgrep:*)",
        "Bash(basename:*)",
        "Bash(dirname:*)",
        "Bash(test:*)",
        "Bash(stat:*)",
        "Bash(find:*)",
    ]


async def run_agent(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Run a single prompt through the Claude Code SDK and return the response.

    Uses persona configuration from the unified config dict.
    The Claude backend provides read-only access to the entire journal
    with diagnostic shell commands for system analysis.

    Args:
        config: Complete configuration dictionary including prompt, instruction, model, etc.
        on_event: Optional event callback
    """
    # Extract values from unified config
    prompt = config.get("prompt", "")
    if not prompt:
        raise ValueError("Missing 'prompt' in config")

    model = config.get("model", _DEFAULT_MODEL)
    max_turns = config.get("max_turns", 32)
    persona = config.get("persona", "default")

    callback = JSONEventCallback(on_event)

    try:
        # Get journal path for file permissions
        journal_path = get_journal()

        # Extract instruction from config
        system_instruction = config.get("instruction", "")

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
                "provider": "claude",
                "journal_path": journal_path,
                "ts": int(time.time() * 1000),
            }
        )

        # Configure Claude Code options with read-only journal access
        options = ClaudeAgentOptions(
            system_prompt=system_instruction,
            model=model,
            cwd=journal_path,  # Set working directory to journal root
            allowed_tools=_get_readonly_tools(journal_path),
            disallowed_tools=["mcp_*"],  # Disable MCP tools
            permission_mode="bypassPermissions",  # Skip prompts, rely on allowed_tools
            max_turns=max_turns,
        )

        # Track tool calls for pairing start/end events
        tool_calls = {}
        response_text = []

        # Stream responses from Claude Code
        async for message in query(prompt=prompt, options=options):
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

                    elif isinstance(block, ThinkingBlock):
                        # Thinking/reasoning block
                        thinking_content = block.thinking
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


__all__ = [
    "run_agent",
]
