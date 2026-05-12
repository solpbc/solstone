# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatReasonAction:
    label: str
    href: str


@dataclass(frozen=True)
class ChatReason:
    code: str
    template: str
    action: ChatReasonAction | None


DISPLAY_NAMES: dict[str, str] = {
    "google": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
}

CHAT_REASONS: dict[str, ChatReason] = {
    "provider_key_invalid": ChatReason(
        code="provider_key_invalid",
        template="your {provider} key didn't validate",
        action=ChatReasonAction(label="Open Settings", href="/app/settings/#providers"),
    ),
    "provider_quota_exceeded": ChatReason(
        code="provider_quota_exceeded",
        template="your {provider} quota is spent — try again later",
        action=None,
    ),
    "network_unreachable": ChatReason(
        code="network_unreachable",
        template="I couldn't reach the network",
        action=None,
    ),
    "provider_response_invalid": ChatReason(
        code="provider_response_invalid",
        template="{provider} sent something I couldn't read — try again",
        action=None,
    ),
    "provider_unavailable": ChatReason(
        code="provider_unavailable",
        template="{provider} is having trouble — try again",
        action=None,
    ),
    "chat_timeout": ChatReason(
        code="chat_timeout",
        template="chat took too long — try again",
        action=None,
    ),
    "unknown": ChatReason(
        code="unknown",
        template="chat had trouble — try again",
        action=None,
    ),
}


def render_chat_reason(code: str, provider: str) -> dict[str, Any]:
    reason = CHAT_REASONS.get(code)
    if reason is None:
        return {"code": code, "message": code, "action": None}

    if code == "unknown":
        display_name = DISPLAY_NAMES.get(provider)
        message = (
            f"something went wrong with {display_name}"
            if display_name
            else reason.template
        )
        return {"code": code, "message": message, "action": None}

    display_name = DISPLAY_NAMES.get(provider, provider)
    # Avoid str.format so future owner copy with braces cannot crash rendering.
    message = reason.template.replace("{provider}", display_name)
    action = (
        {"label": reason.action.label, "href": reason.action.href}
        if reason.action
        else None
    )
    return {"code": code, "message": message, "action": action}
