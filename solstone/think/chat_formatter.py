# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from typing import Any

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)
from solstone.think.utils import get_config


def format_chat(
    entries: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Format chat stream JSONL entries into markdown chunks."""
    owner_name, agent_name = _resolve_labels(context or {})
    chunks: list[dict[str, Any]] = []

    for entry in entries:
        kind = str(entry.get("kind") or "").strip()
        markdown = _format_entry(kind, entry, owner_name, agent_name)
        if not markdown:
            continue

        chunks.append(
            {
                "timestamp": int(entry.get("ts", 0) or 0),
                "markdown": markdown,
                "source": entry,
            }
        )

    return chunks, {"indexer": {"agent": "chat"}}


def _resolve_labels(context: dict[str, Any]) -> tuple[str, str]:
    owner_name = str(context.get("owner_name") or "").strip()
    agent_name = str(context.get("agent_name") or "").strip()

    config = get_config()
    if not owner_name:
        identity = config.get("identity", {})
        owner_name = str(
            identity.get("preferred") or identity.get("name") or ""
        ).strip()
    if not agent_name:
        agent_name = str(config.get("agent", {}).get("name") or "").strip()

    return (owner_name or "Owner", agent_name or "Sol")


def _format_entry(
    kind: str,
    entry: dict[str, Any],
    owner_name: str,
    agent_name: str,
) -> str | None:
    if kind == "owner_message":
        return _speaker_line(owner_name, entry.get("text"))
    if kind == "sol_message":
        return _speaker_line(agent_name, entry.get("text"))
    if kind == "talent_spawned":
        return f"*[{entry['name']} spawned: {entry['task']}]*"
    if kind == "talent_finished":
        return f"*[{entry['name']} finished: {entry['summary']}]*"
    if kind == "talent_errored":
        return f"*[{entry['name']} errored: {entry['reason']}]*"
    if kind == "chat_error":
        return f"*[chat trouble: {entry['reason']}]*"
    if kind == KIND_SOL_CHAT_REQUEST:
        return _format_sol_request(entry)
    if kind in {
        KIND_SOL_CHAT_REQUEST_SUPERSEDED,
        KIND_OWNER_CHAT_OPEN,
        KIND_OWNER_CHAT_DISMISSED,
    }:
        return None
    raise ValueError(f"Unknown chat event kind for formatter: {kind}")


def _format_sol_request(entry: dict[str, Any]) -> str:
    text = f"[sol] {entry.get('summary') or ''}".strip()
    message = str(entry.get("message") or "").strip()
    if message:
        text = f"{text}\n{message}"
    return text


def _speaker_line(label: str, body: Any) -> str:
    text = str(body or "").strip()
    if not text:
        return f"**{label}**"
    return f"**{label}** {text}"
