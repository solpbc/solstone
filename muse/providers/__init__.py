# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""AI provider backends for muse.

This package contains provider-specific implementations for LLM generation
and agent execution. Each provider module exposes:

- generate(): Sync text generation
- agenerate(): Async text generation
- run_agent(): Agent execution with MCP tools

Available providers:
- google: Google Gemini models
- openai: OpenAI GPT models
- anthropic: Anthropic Claude models
"""

from importlib import import_module
from types import ModuleType
from typing import Dict

# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------
# Central registry of supported providers and their module paths.
# All registered providers must implement:
#   - generate(contents, model, ...) -> str
#   - agenerate(contents, model, ...) -> str
#   - run_agent(config, on_event) -> str
#
# The "claude" provider (Claude Code SDK) is intentionally excluded from this
# registry as it uses a fundamentally different execution model (local CLI)
# and is handled as a special case in muse/agents.py.
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, str] = {
    "google": "muse.providers.google",
    "openai": "muse.providers.openai",
    "anthropic": "muse.providers.anthropic",
}


def get_provider_module(provider: str) -> ModuleType:
    """Get the provider module for the given provider name.

    Parameters
    ----------
    provider
        Provider name (e.g., "google", "openai", "anthropic").

    Returns
    -------
    ModuleType
        The provider module with generate, agenerate, and run_agent functions.

    Raises
    ------
    ValueError
        If the provider is not registered.
    """
    if provider not in PROVIDER_REGISTRY:
        valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider: {provider!r}. Valid providers: {valid}")

    return import_module(PROVIDER_REGISTRY[provider])


__all__ = [
    "PROVIDER_REGISTRY",
    "get_provider_module",
]
