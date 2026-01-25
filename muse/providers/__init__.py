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
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------
# Central registry of supported providers and their module paths.
# All registered providers must implement:
#   - generate(contents, model, ...) -> str
#   - agenerate(contents, model, ...) -> str
#   - run_agent(config, on_event) -> str
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, str] = {
    "google": "muse.providers.google",
    "openai": "muse.providers.openai",
    "anthropic": "muse.providers.anthropic",
}

# ---------------------------------------------------------------------------
# Provider Metadata
# ---------------------------------------------------------------------------
# Display labels and environment variable names for each provider.
# Used by settings UI to dynamically build provider dropdowns.
# ---------------------------------------------------------------------------

PROVIDER_METADATA: Dict[str, Dict[str, str]] = {
    "google": {"label": "Google (Gemini)", "env_key": "GOOGLE_API_KEY"},
    "openai": {"label": "OpenAI (GPT)", "env_key": "OPENAI_API_KEY"},
    "anthropic": {"label": "Anthropic (Claude)", "env_key": "ANTHROPIC_API_KEY"},
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


def get_provider_list() -> List[Dict[str, Any]]:
    """Get list of providers with metadata for UI display.

    Returns
    -------
    List[Dict[str, Any]]
        List of provider info dicts, each containing:
        - name: Provider identifier (e.g., "google")
        - label: Display label (e.g., "Google (Gemini)")
        - env_key: Environment variable for API key
    """
    return [
        {"name": name, **PROVIDER_METADATA.get(name, {"label": name, "env_key": ""})}
        for name in PROVIDER_REGISTRY
    ]


__all__ = [
    "PROVIDER_REGISTRY",
    "PROVIDER_METADATA",
    "get_provider_module",
    "get_provider_list",
]
