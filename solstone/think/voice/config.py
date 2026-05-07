# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Voice config readers."""

from __future__ import annotations

import os
from typing import Any

from solstone.think.utils import get_config

DEFAULT_VOICE_MODEL = "gpt-realtime"
DEFAULT_BRAIN_MODEL = "haiku"


def _voice_config() -> dict[str, Any]:
    config = get_config()
    voice = config.get("voice")
    return voice if isinstance(voice, dict) else {}


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def get_openai_api_key() -> str | None:
    configured = _clean_str(_voice_config().get("openai_api_key"))
    if configured:
        return configured
    return _clean_str(os.environ.get("OPENAI_API_KEY"))


def get_voice_model() -> str:
    return _clean_str(_voice_config().get("model")) or DEFAULT_VOICE_MODEL


def get_brain_model() -> str:
    return _clean_str(_voice_config().get("brain_model")) or DEFAULT_BRAIN_MODEL


__all__ = [
    "DEFAULT_BRAIN_MODEL",
    "DEFAULT_VOICE_MODEL",
    "get_brain_model",
    "get_openai_api_key",
    "get_voice_model",
]
