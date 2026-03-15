# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Conversation memory service for solstone.

Manages conversation exchange storage, retrieval, and context injection
for the unified muse agent. Three layers of recall:

- Layer 1: Recent exchanges (last ~10 turns), loaded directly into context
- Layer 2: Today's earlier exchanges, summarized compactly
- Layer 3: Older conversations, searchable via journal search (automatic —
  exchanges are stored as journal entries indexed by FTS5)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from think.utils import get_journal, now_ms

logger = logging.getLogger(__name__)

# Append-only exchange log for fast recent retrieval
EXCHANGES_FILE = "conversation/exchanges.jsonl"

# Journal stream name for conversation segments
CONVERSATION_STREAM = "conversation"

# Marker in unified muse for memory injection
INJECTION_MARKER = "CONVERSATION_MEMORY_INJECTION_POINT"

# Context budget: max characters for agent response in recent exchanges
MAX_RESPONSE_CHARS = 300

# Max characters for user message in compact summaries
MAX_MESSAGE_CHARS = 100

# Default number of recent exchanges for layer 1
DEFAULT_RECENT_LIMIT = 10


# ---------------------------------------------------------------------------
# Exchange Recording
# ---------------------------------------------------------------------------


def record_exchange(
    *,
    ts: int | None = None,
    facet: str = "",
    app: str = "",
    path: str = "",
    user_message: str = "",
    agent_response: str = "",
    muse: str = "",
    agent_id: str = "",
) -> None:
    """Record a conversation exchange to journal storage.

    Writes to two locations:
    1. conversation/exchanges.jsonl — append-only quick-read index
    2. YYYYMMDD/conversation/HHMMSS_1/agents/conversation.md — journal entry
       for FTS5 search indexing (matches */*/*/agents/*.md formatter pattern)

    Also runs lightweight entity extraction on the conversation text.
    """
    if not user_message or not agent_response:
        return

    if ts is None:
        ts = now_ms()

    journal = get_journal()

    exchange = {
        "ts": ts,
        "facet": facet,
        "app": app,
        "path": path,
        "user_message": user_message,
        "agent_response": agent_response,
        "muse": muse,
        "agent_id": agent_id,
    }

    # 1. Append to exchanges.jsonl (fast-read index)
    jsonl_path = Path(journal) / EXCHANGES_FILE
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(exchange, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write exchange to JSONL")

    # 2. Write journal segment for search indexing
    dt = datetime.fromtimestamp(ts / 1000)
    day = dt.strftime("%Y%m%d")
    time_key = dt.strftime("%H%M%S")
    segment = f"{time_key}_1"

    seg_dir = Path(journal) / day / CONVERSATION_STREAM / segment / "agents"
    seg_dir.mkdir(parents=True, exist_ok=True)

    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    md_parts = ["# Conversation Exchange\n"]
    md_parts.append(f"**Time:** {time_str}")
    if facet:
        md_parts.append(f"**Facet:** {facet}")
    if app:
        app_info = f"{app} ({path})" if path else app
        md_parts.append(f"**App:** {app_info}")
    md_parts.append("")
    md_parts.append("## User\n")
    md_parts.append(user_message)
    md_parts.append("")
    md_parts.append("## Sol\n")
    md_parts.append(agent_response)

    md_content = "\n".join(md_parts) + "\n"

    md_path = seg_dir / "conversation.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    except Exception:
        logger.exception("Failed to write conversation journal entry")

    # 3. Entity extraction
    _extract_entities(
        user_message + " " + agent_response, facet=facet, day=day
    )


def _extract_entities(text: str, *, facet: str, day: str) -> None:
    """Detect known entity names mentioned in conversation text.

    Matches against attached entities for the active facet. Any matches
    are recorded as detected entities for the day, integrating with the
    existing entity signal infrastructure.
    """
    if not facet:
        return

    try:
        from think.entities.loading import load_entities

        entities = load_entities(facet)
        if not entities:
            return

        text_lower = text.lower()

        for entity in entities:
            name = entity.get("name", "")
            if not name or len(name) < 3:
                continue

            # Word boundary match for entity name
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", text_lower):
                try:
                    from think.entities.saving import save_detected_entity

                    save_detected_entity(
                        facet=facet,
                        day=day,
                        entity_type=entity.get("type", "Person"),
                        name=name,
                        description="Mentioned in conversation",
                    )
                except ValueError:
                    pass  # Already detected today — expected
                except Exception:
                    logger.debug(
                        "Failed to record entity detection: %s", name, exc_info=True
                    )
    except Exception:
        logger.debug("Entity extraction from conversation failed", exc_info=True)


# ---------------------------------------------------------------------------
# Exchange Retrieval
# ---------------------------------------------------------------------------


def get_recent_exchanges(
    limit: int = DEFAULT_RECENT_LIMIT,
    facet: str | None = None,
) -> list[dict]:
    """Read the most recent conversation exchanges.

    Args:
        limit: Maximum number of exchanges to return.
        facet: If provided, only return exchanges from this facet.

    Returns:
        List of exchange dicts, most recent last.
    """
    journal = get_journal()
    jsonl_path = Path(journal) / EXCHANGES_FILE

    if not jsonl_path.exists():
        return []

    exchanges = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    if facet and ex.get("facet") != facet:
                        continue
                    exchanges.append(ex)
                except json.JSONDecodeError:
                    continue
    except Exception:
        logger.exception("Failed to read exchanges")
        return []

    return exchanges[-limit:]


def get_today_exchanges(facet: str | None = None) -> list[dict]:
    """Read all conversation exchanges from today.

    Args:
        facet: If provided, only return exchanges from this facet.

    Returns:
        List of exchange dicts from today, chronological order.
    """
    journal = get_journal()
    jsonl_path = Path(journal) / EXCHANGES_FILE

    if not jsonl_path.exists():
        return []

    today = datetime.now().strftime("%Y%m%d")
    exchanges = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    ts = ex.get("ts", 0)
                    ex_day = datetime.fromtimestamp(ts / 1000).strftime("%Y%m%d")
                    if ex_day != today:
                        continue
                    if facet and ex.get("facet") != facet:
                        continue
                    exchanges.append(ex)
                except (json.JSONDecodeError, ValueError, OSError):
                    continue
    except Exception:
        logger.exception("Failed to read today's exchanges")
        return []

    return exchanges


# ---------------------------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------------------------


def _format_exchange(ex: dict, *, compact: bool = False) -> str:
    """Format a single exchange for context injection.

    Args:
        ex: Exchange dict.
        compact: If True, return a one-liner summary.

    Returns:
        Formatted string.
    """
    ts = ex.get("ts", 0)
    try:
        time_str = datetime.fromtimestamp(ts / 1000).strftime("%H:%M")
    except (ValueError, OSError):
        time_str = "??:??"

    app = ex.get("app", "")
    facet_val = ex.get("facet", "")

    context_parts = [time_str]
    if app:
        context_parts.append(app)
    if facet_val:
        context_parts.append(facet_val)
    context = " · ".join(context_parts)

    user_msg = ex.get("user_message", "")
    agent_resp = ex.get("agent_response", "")

    if compact:
        truncated = user_msg[:MAX_MESSAGE_CHARS]
        if len(user_msg) > MAX_MESSAGE_CHARS:
            truncated += "..."
        return f"- [{context}] {truncated}"

    # Full exchange with truncated response
    truncated_resp = agent_resp[:MAX_RESPONSE_CHARS]
    if len(agent_resp) > MAX_RESPONSE_CHARS:
        truncated_resp += "..."

    return f"[{context}] User: {user_msg}\nSol: {truncated_resp}"


def build_memory_context(
    facet: str | None = None,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
) -> str:
    """Build the full conversation memory context block.

    Assembles layer 1 (recent exchanges) and layer 2 (today's summary)
    into a formatted block for injection into the unified muse prompt.

    Args:
        facet: Active facet for filtering.
        recent_limit: Number of recent exchanges for layer 1.

    Returns:
        Formatted memory context string, or empty string if no history.
    """
    recent = get_recent_exchanges(limit=recent_limit, facet=facet)
    if not recent:
        return ""

    today_all = get_today_exchanges(facet=facet)

    parts = []

    # Layer 2: Earlier today (exchanges beyond the recent set)
    if len(today_all) > len(recent):
        earlier = today_all[: -len(recent)]
        if earlier:
            parts.append("### Earlier Today\n")
            for ex in earlier:
                parts.append(_format_exchange(ex, compact=True))
            parts.append("")

    # Layer 1: Recent exchanges (full detail)
    parts.append("### Recent Conversations\n")
    parts.append(
        "The following are your most recent exchanges with the user:\n"
    )
    for ex in recent:
        parts.append(_format_exchange(ex, compact=False))
        parts.append("")

    return "\n".join(parts).strip()


def inject_memory(user_instruction: str, memory_context: str) -> str:
    """Replace the CONVERSATION_MEMORY_INJECTION_POINT with memory context.

    Args:
        user_instruction: The unified muse's user instruction text.
        memory_context: Formatted conversation memory to inject.

    Returns:
        Modified user instruction with memory context injected.
    """
    if INJECTION_MARKER not in user_instruction:
        return user_instruction

    # Replace the entire HTML comment block containing the marker
    pattern = r"<!--\s*" + re.escape(INJECTION_MARKER) + r".*?-->"

    if memory_context:
        replacement = memory_context
    else:
        replacement = "No conversation history yet."

    return re.sub(pattern, replacement, user_instruction, flags=re.DOTALL)
