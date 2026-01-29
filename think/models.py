# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import fnmatch
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import frontmatter

from think.utils import get_config, get_journal

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

TIER_PRO = 1
TIER_FLASH = 2
TIER_LITE = 3

# ---------------------------------------------------------------------------
# Model constants
#
# IMPORTANT: When updating these models, verify pricing support:
#   1. Run: make test-only TEST=tests/test_models.py::test_all_default_models_have_pricing
#   2. If test fails, update genai-prices: make update-prices
#   3. If still failing, the model may be too new for genai-prices
#
# The genai-prices library provides token cost data. New models may not have
# pricing immediately after release. See: https://pypi.org/project/genai-prices/
# ---------------------------------------------------------------------------

GEMINI_PRO = "gemini-3-pro-preview"
GEMINI_FLASH = "gemini-3-flash-preview"
GEMINI_LITE = "gemini-2.5-flash-lite"

GPT_5 = "gpt-5.2"
GPT_5_MINI = "gpt-5-mini"
GPT_5_NANO = "gpt-5-nano"

CLAUDE_OPUS_4 = "claude-opus-4-5"
CLAUDE_SONNET_4 = "claude-sonnet-4-5"
CLAUDE_HAIKU_4 = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# System defaults: provider -> tier -> model
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: Dict[str, Dict[int, str]] = {
    "google": {
        TIER_PRO: GEMINI_PRO,
        TIER_FLASH: GEMINI_FLASH,
        TIER_LITE: GEMINI_LITE,
    },
    "openai": {
        TIER_PRO: GPT_5,
        TIER_FLASH: GPT_5_MINI,
        TIER_LITE: GPT_5_NANO,
    },
    "anthropic": {
        TIER_PRO: CLAUDE_OPUS_4,
        TIER_FLASH: CLAUDE_SONNET_4,
        TIER_LITE: CLAUDE_HAIKU_4,
    },
}

DEFAULT_PROVIDER = "google"
DEFAULT_TIER = TIER_FLASH


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IncompleteJSONError(ValueError):
    """Raised when JSON response is truncated due to token limits or other reasons.

    Attributes:
        reason: The finish/stop reason from the API (e.g., "MAX_TOKENS", "length").
        partial_text: The truncated response text, useful for debugging.
    """

    def __init__(self, reason: str, partial_text: str):
        self.reason = reason
        self.partial_text = partial_text
        super().__init__(f"JSON response incomplete (reason: {reason})")


# ---------------------------------------------------------------------------
# Context defaults: context pattern -> {tier, label, group}
#
# These define the default tier for each context when not overridden in config.
# Patterns support glob-style matching (fnmatch).
#
# Each entry contains:
#   - tier: Default tier (TIER_PRO, TIER_FLASH, TIER_LITE)
#   - label: Human-readable name for settings UI
#   - group: Category for grouping in settings UI
#
# NAMING CONVENTION:
#   {module}.{feature}[.{operation}]
#
# Examples:
#   - observe.describe.frame    -> observe module, describe feature, frame operation
#   - observe.enrich            -> observe module, enrich feature (no sub-operation)
#   - agent.*                   -> agent module, all features (wildcard)
#   - app.chat.title            -> apps module, chat app, title operation
#
# DYNAMIC DISCOVERY:
#   Categories (observe/categories/*.json) and agents (muse/*.md,
#   apps/*/muse/*.md) can express tier/label/group in their frontmatter.
#   These are discovered at runtime and merged with the static defaults below.
#
# When adding new contexts:
#   1. Use module prefix matching the package (observe, think, app)
#   2. Add specific operations as suffixes when granular control is needed
#   3. Use wildcards sparingly - prefer explicit entries for clarity
#   4. If not listed here, context falls back to DEFAULT_TIER (FLASH)
#   5. For categories/agents, prefer adding tier/label/group to JSON configs
# ---------------------------------------------------------------------------

# Static context defaults - non-discoverable contexts only
# Categories and agents express their own tier/label/group in JSON configs
CONTEXT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Observe pipeline - screen and audio capture processing
    "observe.describe.frame": {
        "tier": TIER_LITE,
        "label": "Screen Categorization",
        "group": "Observe",
    },
    # Fallback for categories without explicit tier in their JSON
    "observe.describe.*": {
        "tier": TIER_FLASH,
        "label": "Screen Extraction",
        "group": "Observe",
    },
    "observe.detect.segment": {
        "tier": TIER_FLASH,
        "label": "Segmentation",
        "group": "Import",
    },
    "observe.detect.json": {
        "tier": TIER_FLASH,
        "label": "Normalization",
        "group": "Import",
    },
    "observe.enrich": {
        "tier": TIER_FLASH,
        "label": "Audio Enrichment",
        "group": "Observe",
    },
    "observe.transcribe.gemini": {
        "tier": TIER_FLASH,
        "label": "Audio Transcription (Gemini)",
        "group": "Observe",
    },
    "observe.extract.selection": {
        "tier": TIER_FLASH,
        "label": "Frame Selection",
        "group": "Observe",
    },
    "observe.summarize": {
        "tier": TIER_FLASH,
        "label": "Summarization",
        "group": "Import",
    },
    # Generator pipeline - daily analysis and summaries
    "agent.entities.*": {
        "tier": TIER_LITE,
        "label": "Entity Extraction",
        "group": "Think",
    },
    "agent.daily_schedule.*": {
        "tier": TIER_LITE,
        "label": "Maintenance Window",
        "group": "Think",
    },
    "agent.*": {
        "tier": TIER_FLASH,
        "label": "Agent Outputs",
        "group": "Think",
    },
    # Utilities - miscellaneous processing tasks
    "detect.created": {
        "tier": TIER_LITE,
        "label": "Date Detection",
        "group": "Import",
    },
    "planner.generate": {
        "tier": TIER_FLASH,
        "label": "Agent Prompt Generation",
        "group": "Think",
    },
    # Apps - application-specific contexts
    "app.chat.title": {
        "tier": TIER_LITE,
        "label": "Chat Title Generation",
        "group": "Apps",
    },
}


# ---------------------------------------------------------------------------
# Dynamic context discovery
# ---------------------------------------------------------------------------

# Cached context registry (built lazily on first use)
_context_registry: Optional[Dict[str, Dict[str, Any]]] = None


def _discover_agent_contexts() -> Dict[str, Dict[str, Any]]:
    """Discover agent context defaults from JSON config files.

    Scans system agents (muse/*.md) and app agents (apps/*/muse/*.md)
    for tier/label/group metadata. This is a lightweight scan that only reads
    the JSON metadata, not the full agent configuration.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of context patterns to {tier, label, group} dicts.
        Context patterns are: agent.system.{name} or agent.{app}.{name}
    """
    contexts = {}

    # System agents from muse/ (agents have "tools" field, generators don't)
    muse_dir = Path(__file__).parent.parent / "muse"
    if muse_dir.exists():
        for md_path in muse_dir.glob("*.md"):
            agent_name = md_path.stem
            try:
                post = frontmatter.load(
                    md_path,
                )
                config = post.metadata if post.metadata else {}

                # Only include agents (they have "tools" field)
                if "tools" not in config:
                    continue

                context = f"agent.system.{agent_name}"
                contexts[context] = {
                    "tier": config.get("tier", TIER_FLASH),
                    "label": config.get("label", config.get("title", agent_name)),
                    "group": config.get("group", "Agents"),
                }
            except Exception:
                pass  # Skip agents that can't be loaded

    # App agents from apps/*/muse/
    apps_dir = Path(__file__).parent.parent / "apps"
    if apps_dir.is_dir():
        for app_path in apps_dir.iterdir():
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue
            muse_subdir = app_path / "muse"
            if not muse_subdir.is_dir():
                continue
            app_name = app_path.name
            for md_path in muse_subdir.glob("*.md"):
                agent_name = md_path.stem
                try:
                    post = frontmatter.load(
                        md_path,
                    )
                    config = post.metadata if post.metadata else {}

                    # Only include agents (they have "tools" field)
                    if "tools" not in config:
                        continue

                    context = f"agent.{app_name}.{agent_name}"
                    contexts[context] = {
                        "tier": config.get("tier", TIER_FLASH),
                        "label": config.get("label", config.get("title", agent_name)),
                        "group": config.get("group", "Agents"),
                    }
                except Exception:
                    pass  # Skip agents that can't be loaded

    return contexts


def _build_context_registry() -> Dict[str, Dict[str, Any]]:
    """Build complete context registry from static defaults and discovered configs.

    Merges:
    1. Static CONTEXT_DEFAULTS (non-discoverable contexts)
    2. Category contexts from observe/describe.py CATEGORIES
    3. Agent contexts from _discover_agent_contexts()

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Complete context registry mapping patterns to {tier, label, group}.
    """
    # Start with static defaults
    registry = dict(CONTEXT_DEFAULTS)

    # Merge category contexts (lazy import to avoid circular dependency)
    try:
        from observe.describe import CATEGORIES

        for category, metadata in CATEGORIES.items():
            context = metadata.get("context", f"observe.describe.{category}")
            registry[context] = {
                "tier": metadata.get("tier", TIER_FLASH),
                "label": metadata.get("label", category.replace("_", " ").title()),
                "group": metadata.get("group", "Screen Analysis"),
            }
    except ImportError:
        pass  # observe module not available

    # Merge agent contexts
    agent_contexts = _discover_agent_contexts()
    registry.update(agent_contexts)

    return registry


def get_context_registry() -> Dict[str, Dict[str, Any]]:
    """Get the complete context registry, building it lazily on first use.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Complete context registry mapping patterns to {tier, label, group}.
    """
    global _context_registry
    if _context_registry is None:
        _context_registry = _build_context_registry()
    return _context_registry


def _resolve_tier(context: str) -> int:
    """Resolve context to tier number.

    Checks journal config contexts first, then dynamic context registry with glob matching.

    Parameters
    ----------
    context
        Context string (e.g., "agent.system.default", "agent.meetings").

    Returns
    -------
    int
        Tier number (1=pro, 2=flash, 3=lite).
    """
    from think.utils import get_config

    journal_config = get_config()
    providers_config = journal_config.get("providers", {})
    contexts = providers_config.get("contexts", {})

    # Get dynamic context registry (includes static defaults + discovered categories/agents)
    registry = get_context_registry()

    # Check journal config contexts first (exact match)
    if context in contexts:
        return contexts[context].get("tier", DEFAULT_TIER)

    # Check context registry (exact match)
    if context in registry:
        return registry[context]["tier"]

    # Check glob patterns in both
    for pattern, ctx_config in contexts.items():
        if fnmatch.fnmatch(context, pattern):
            return ctx_config.get("tier", DEFAULT_TIER)

    for pattern, ctx_default in registry.items():
        if fnmatch.fnmatch(context, pattern):
            return ctx_default["tier"]

    return DEFAULT_TIER


def _resolve_model(provider: str, tier: int, config_models: Dict[str, Any]) -> str:
    """Resolve tier to model string for a given provider.

    Checks config overrides first, then falls back to system defaults.
    If requested tier is unavailable, falls back to more capable tiers
    (3→2→1, i.e., lite→flash→pro).

    Parameters
    ----------
    provider
        Provider name ("google", "openai", "anthropic").
    tier
        Tier number (1=pro, 2=flash, 3=lite).
    config_models
        The "models" section from providers config, mapping provider to tier overrides.

    Returns
    -------
    str
        Model identifier string.
    """
    # Check config overrides first
    provider_overrides = config_models.get(provider, {})

    # Try requested tier, then fall back to more capable tiers (lower numbers)
    for t in [tier, tier - 1, tier - 2] if tier > 1 else [tier]:
        if t < 1:
            continue

        # Check config override (tier as string key in JSON)
        tier_key = str(t)
        if tier_key in provider_overrides:
            return provider_overrides[tier_key]

        # Check system defaults
        provider_defaults = PROVIDER_DEFAULTS.get(provider, {})
        if t in provider_defaults:
            return provider_defaults[t]

    # Ultimate fallback: system default for provider at DEFAULT_TIER
    provider_defaults = PROVIDER_DEFAULTS.get(
        provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER]
    )
    return provider_defaults.get(DEFAULT_TIER, GEMINI_FLASH)


def resolve_model_for_provider(context: str, provider: str) -> str:
    """Resolve model for a specific provider based on context tier.

    Use this when provider is overridden from the default - resolves the
    appropriate model for the given provider at the context's tier.

    Parameters
    ----------
    context
        Context string (e.g., "agent.system.default").
    provider
        Provider name ("google", "openai", "anthropic").

    Returns
    -------
    str
        Model identifier string for the provider at the context's tier.
    """
    from think.utils import get_config

    tier = _resolve_tier(context)
    journal_config = get_config()
    providers_config = journal_config.get("providers", {})
    config_models = providers_config.get("models", {})

    return _resolve_model(provider, tier, config_models)


def resolve_provider(context: str) -> tuple[str, str]:
    """Resolve context to provider and model based on configuration.

    Matches context against configured contexts using exact match first,
    then glob patterns (via fnmatch), falling back to defaults.

    Supports both explicit model strings and tier-based routing:
    - {"provider": "google", "model": "gemini-3-flash-preview"} - explicit model
    - {"provider": "google", "tier": 2} - tier-based (2=flash)
    - {"tier": 1} - tier only, inherits provider from default

    The "models" section in providers config allows overriding which model
    is used for each tier per provider.

    Parameters
    ----------
    context
        Context string (e.g., "observe.describe.frame", "agent.meetings").

    Returns
    -------
    tuple[str, str]
        (provider_name, model) tuple. Provider is one of "google", "openai",
        "anthropic". Model is the full model identifier string.
    """
    config = get_config()
    providers = config.get("providers", {})
    config_models = providers.get("models", {})

    # Get defaults
    default = providers.get("default", {})
    default_provider = default.get("provider", DEFAULT_PROVIDER)
    default_tier = default.get("tier", DEFAULT_TIER)

    # Handle explicit "model" key in default (overrides tier-based resolution)
    if "model" in default and "tier" not in default:
        default_model = default["model"]
    else:
        default_model = _resolve_model(default_provider, default_tier, config_models)

    contexts = providers.get("contexts", {})

    # Find matching context config
    match_config: Optional[Dict[str, Any]] = None

    if context and contexts:
        # Check for exact match first
        if context in contexts:
            match_config = contexts[context]
        else:
            # Check glob patterns - most specific (longest non-wildcard prefix) wins
            matches = []
            for pattern, ctx_config in contexts.items():
                if fnmatch.fnmatch(context, pattern):
                    specificity = len(pattern.split("*")[0])
                    matches.append((specificity, pattern, ctx_config))

            if matches:
                matches.sort(key=lambda x: x[0], reverse=True)
                _, _, match_config = matches[0]

    # No context match - check dynamic context registry for this context
    if match_config is None:
        # Get dynamic context registry (includes static defaults + discovered categories/agents)
        registry = get_context_registry()

        # Check for matching context default (exact match first, then glob)
        context_tier = None
        if context:
            if context in registry:
                context_tier = registry[context]["tier"]
            else:
                # Check glob patterns
                matches = []
                for pattern, ctx_default in registry.items():
                    if fnmatch.fnmatch(context, pattern):
                        specificity = len(pattern.split("*")[0])
                        matches.append((specificity, ctx_default["tier"]))
                if matches:
                    matches.sort(key=lambda x: x[0], reverse=True)
                    context_tier = matches[0][1]

        if context_tier is not None:
            model = _resolve_model(default_provider, context_tier, config_models)
            return (default_provider, model)

        return (default_provider, default_model)

    # Resolve provider (from match or default)
    provider = match_config.get("provider", default_provider)

    # Resolve model: explicit model takes precedence over tier
    if "model" in match_config:
        model = match_config["model"]
    elif "tier" in match_config:
        tier = match_config["tier"]
        # Validate tier
        if not isinstance(tier, int) or tier < 1 or tier > 3:
            logging.getLogger(__name__).warning(
                "Invalid tier %r in context %r, using default", tier, context
            )
            tier = default_tier
        model = _resolve_model(provider, tier, config_models)
    else:
        # No model or tier specified - use default tier
        model = _resolve_model(provider, default_tier, config_models)

    return (provider, model)


def log_token_usage(
    model: str,
    usage: Union[Dict[str, Any], Any],
    context: Optional[str] = None,
    segment: Optional[str] = None,
) -> None:
    """Log token usage to journal with unified schema.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash")
    usage : dict or response object
        Usage data in provider-specific format, OR a Gemini response object.
        Dict formats supported:
        - OpenAI format: {input_tokens, output_tokens, total_tokens,
                         details: {input: {cached_tokens}, output: {reasoning_tokens}}}
        - Gemini format: {prompt_token_count, candidates_token_count,
                         cached_content_token_count, thoughts_token_count, total_token_count}
        - Unified format: {input_tokens, output_tokens, total_tokens,
                          cached_tokens, reasoning_tokens, requests}
        Response objects: Gemini GenerateContentResponse with usage_metadata attribute
    context : str, optional
        Context string (e.g., "module.function:123" or "agent.name.id").
        If None, auto-detects from call stack.
    segment : str, optional
        Segment key (e.g., "143022_300") for attribution.
        If None, falls back to SEGMENT_KEY environment variable.
    """
    try:
        journal = get_journal()

        # Extract from Gemini response object if needed
        if hasattr(usage, "usage_metadata"):
            try:
                metadata = usage.usage_metadata
                usage = {
                    "prompt_token_count": getattr(metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(
                        metadata, "candidates_token_count", 0
                    ),
                    "cached_content_token_count": getattr(
                        metadata, "cached_content_token_count", 0
                    ),
                    "thoughts_token_count": getattr(
                        metadata, "thoughts_token_count", 0
                    ),
                    "total_token_count": getattr(metadata, "total_token_count", 0),
                }
            except Exception:
                return  # Can't extract, fail silently

        # Auto-detect calling context if not provided
        if context is None:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None

            # Skip frames that contain "gemini" in function name
            while caller_frame and "gemini" in caller_frame.f_code.co_name.lower():
                caller_frame = caller_frame.f_back

            if caller_frame:
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                func_name = caller_frame.f_code.co_name
                line_num = caller_frame.f_lineno

                # Clean up module name
                for prefix in ["think.", "observe.", "convey."]:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]
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
                    normalized_usage["reasoning_tokens"] = output_details[
                        "reasoning_tokens"
                    ]

            # Optional requests field for OpenAI
            if "requests" in usage and usage["requests"] is not None:
                normalized_usage["requests"] = usage["requests"]

            # Pass through Anthropic cache fields if present
            if usage.get("cached_tokens"):
                normalized_usage["cached_tokens"] = usage["cached_tokens"]
            if usage.get("cache_creation_tokens"):
                normalized_usage["cache_creation_tokens"] = usage[
                    "cache_creation_tokens"
                ]

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

        # Add segment: prefer parameter, fallback to env (set by think/insight, observe handlers)
        segment_key = segment or os.getenv("SEGMENT_KEY")
        if segment_key:
            token_data["segment"] = segment_key

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


def get_model_provider(model: str) -> str:
    """Get the provider name from a model identifier.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-5", "gemini-2.5-flash", "claude-sonnet-4-5")

    Returns
    -------
    str
        Provider name: "openai", "google", "anthropic", or "unknown"
    """
    model_lower = model.lower()

    if model_lower.startswith("gpt"):
        return "openai"
    elif model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("claude"):
        return "anthropic"
    else:
        return "unknown"


def calc_token_cost(token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculate cost for a token usage record.

    Parameters
    ----------
    token_data : dict
        Token usage record from journal logs with structure:
        {
            "model": "gemini-2.5-flash",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 500,
                "cached_tokens": 800,
                "reasoning_tokens": 200,
                ...
            }
        }

    Returns
    -------
    dict or None
        Cost breakdown:
        {
            "total_cost": 0.00123,
            "input_cost": 0.00075,
            "output_cost": 0.00048,
            "currency": "USD"
        }
        Returns None if pricing unavailable or calculation fails.
    """
    try:
        from genai_prices import Usage, calc_price

        model = token_data.get("model")
        usage_data = token_data.get("usage", {})

        if not model or not usage_data:
            return None

        # Get provider ID
        provider_id = get_model_provider(model)
        if provider_id == "unknown":
            return None

        # Map our token fields to genai_prices Usage format
        # Note: Gemini reports reasoning_tokens separately, but they're billed at
        # output token rates. genai-prices doesn't have a separate field for reasoning,
        # so we add them to output_tokens for correct pricing.
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cached_tokens = usage_data.get("cached_tokens", 0)
        reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        # Add reasoning tokens to output for pricing (Gemini bills them as output)
        total_output_tokens = output_tokens + reasoning_tokens

        # Create Usage object
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=total_output_tokens,
            cache_read_tokens=cached_tokens if cached_tokens > 0 else None,
        )

        # Calculate price
        result = calc_price(
            usage=usage,
            model_ref=model,
            provider_id=provider_id,
        )

        # Return simplified cost breakdown
        return {
            "total_cost": float(result.total_price),
            "input_cost": float(result.input_price),
            "output_cost": float(result.output_price),
            "currency": "USD",
        }

    except Exception:
        # Silently fail if pricing unavailable
        return None


def calc_agent_cost(
    model: Optional[str], usage: Optional[Dict[str, Any]]
) -> Optional[float]:
    """Calculate total cost for an agent run from model and usage data.

    Convenience wrapper around calc_token_cost for agent cost lookups.

    Returns total cost in USD, or None if data is missing or pricing unavailable.
    """
    if not model or not usage:
        return None
    try:
        cost_data = calc_token_cost({"model": model, "usage": usage})
        if cost_data:
            return cost_data["total_cost"]
    except Exception:
        return None
    return None


def iter_token_log(day: str) -> Any:
    """Iterate over token log entries for a given day.

    Yields parsed JSON entries from the token log file, skipping empty lines
    and invalid JSON. This is a shared utility for code that processes token logs.

    Parameters
    ----------
    day : str
        Day in YYYYMMDD format.

    Yields
    ------
    dict
        Parsed token log entry with fields: timestamp, model, context, usage,
        and optionally segment.
    """
    journal = get_journal()
    log_path = Path(journal) / "tokens" / f"{day}.jsonl"

    if not log_path.exists():
        return

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_usage_cost(
    day: str,
    segment: Optional[str] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Get aggregated token usage and cost for a day, optionally filtered.

    This is a shared utility for apps that want to display cost information
    for segments, agent runs, or other contexts.

    Parameters
    ----------
    day : str
        Day in YYYYMMDD format.
    segment : str, optional
        Filter to entries with this exact segment key.
    context : str, optional
        Filter to entries where context starts with this prefix.
        For example, "agent.system" matches "agent.system.default".

    Returns
    -------
    dict
        Aggregated usage data:
        {
            "requests": int,
            "tokens": int,
            "cost": float,  # USD
        }
        Returns zeros if no matching entries or day file doesn't exist.
    """
    result = {"requests": 0, "tokens": 0, "cost": 0.0}

    for entry in iter_token_log(day):
        # Apply filters
        if segment is not None and entry.get("segment") != segment:
            continue
        if context is not None:
            entry_context = entry.get("context", "")
            if not entry_context.startswith(context):
                continue

        # Skip unknown providers (can't calculate cost)
        model = entry.get("model", "unknown")
        if get_model_provider(model) == "unknown":
            continue

        # Accumulate
        usage = entry.get("usage", {})
        result["requests"] += 1
        result["tokens"] += usage.get("total_tokens", 0) or 0

        cost_data = calc_token_cost(entry)
        if cost_data:
            result["cost"] += cost_data["total_cost"]

    return result


# ---------------------------------------------------------------------------
# Unified generate/agenerate with provider routing
# ---------------------------------------------------------------------------


def _validate_json_response(result: Dict[str, Any], json_output: bool) -> None:
    """Validate response for JSON output mode.

    Raises IncompleteJSONError if finish_reason indicates truncation.
    """
    if not json_output:
        return

    finish_reason = result.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        raise IncompleteJSONError(
            reason=finish_reason,
            partial_text=result.get("text", ""),
        )


def generate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """Generate text using the configured provider for the given context.

    Routes the request to the appropriate backend (Google, OpenAI, or Anthropic)
    based on the providers configuration in journal.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    context : str
        Context string for routing and token logging (e.g., "agent.meetings").
        This is required and determines which provider/model to use.
    temperature : float
        Temperature for generation (default: 0.3).
    max_output_tokens : int
        Maximum tokens for the model's response output.
    system_instruction : str, optional
        System instruction for the model.
    json_output : bool
        Whether to request JSON response format.
    thinking_budget : int, optional
        Token budget for model thinking (ignored by providers that don't support it).
    timeout_s : float, optional
        Request timeout in seconds.
    **kwargs
        Additional provider-specific options passed through to the backend.

    Returns
    -------
    str
        Response text from the model.

    Raises
    ------
    ValueError
        If the resolved provider is not supported.
    IncompleteJSONError
        If json_output=True and response was truncated.
    """
    from think.providers import get_provider_module

    # Allow model override via kwargs (used by callers with explicit model selection)
    model_override = kwargs.pop("model", None)

    provider, model = resolve_provider(context)
    if model_override:
        model = model_override

    # Get provider module via registry (raises ValueError for unknown providers)
    provider_mod = get_provider_module(provider)

    # Call provider's run_generate (returns GenerateResult)
    result = provider_mod.run_generate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        **kwargs,
    )

    # Validate JSON output if requested
    _validate_json_response(result, json_output)

    # Log token usage centrally
    if result.get("usage"):
        log_token_usage(model=model, usage=result["usage"], context=context)

    return result["text"]


def generate_with_result(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> dict:
    """Generate text and return full result with usage data.

    Same as generate() but returns the full GenerateResult dict instead of
    just the text. Used by cortex-managed generators that need usage data
    for event emission.

    Returns
    -------
    dict
        GenerateResult with: text, usage, finish_reason, thinking.
    """
    from think.providers import get_provider_module

    model_override = kwargs.pop("model", None)

    provider, model = resolve_provider(context)
    if model_override:
        model = model_override

    provider_mod = get_provider_module(provider)

    result = provider_mod.run_generate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        **kwargs,
    )

    _validate_json_response(result, json_output)

    if result.get("usage"):
        log_token_usage(model=model, usage=result["usage"], context=context)

    return result


async def agenerate(
    contents: Union[str, List[Any]],
    context: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: Optional[str] = None,
    json_output: bool = False,
    thinking_budget: Optional[int] = None,
    timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> str:
    """Async generate text using the configured provider for the given context.

    Routes the request to the appropriate backend (Google, OpenAI, or Anthropic)
    based on the providers configuration in journal.json.

    Parameters
    ----------
    contents : str or List
        The content to send to the model.
    context : str
        Context string for routing and token logging (e.g., "agent.meetings").
        This is required and determines which provider/model to use.
    temperature : float
        Temperature for generation (default: 0.3).
    max_output_tokens : int
        Maximum tokens for the model's response output.
    system_instruction : str, optional
        System instruction for the model.
    json_output : bool
        Whether to request JSON response format.
    thinking_budget : int, optional
        Token budget for model thinking (ignored by providers that don't support it).
    timeout_s : float, optional
        Request timeout in seconds.
    **kwargs
        Additional provider-specific options passed through to the backend.

    Returns
    -------
    str
        Response text from the model.

    Raises
    ------
    ValueError
        If the resolved provider is not supported.
    IncompleteJSONError
        If json_output=True and response was truncated.
    """
    from think.providers import get_provider_module

    # Allow model override via kwargs (used by Batch for explicit model selection)
    model_override = kwargs.pop("model", None)

    provider, model = resolve_provider(context)
    if model_override:
        model = model_override

    # Get provider module via registry (raises ValueError for unknown providers)
    provider_mod = get_provider_module(provider)

    # Call provider's run_agenerate (returns GenerateResult)
    result = await provider_mod.run_agenerate(
        contents=contents,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
        **kwargs,
    )

    # Validate JSON output if requested
    _validate_json_response(result, json_output)

    # Log token usage centrally
    if result.get("usage"):
        log_token_usage(model=model, usage=result["usage"], context=context)

    return result["text"]


__all__ = [
    # Provider configuration
    "DEFAULT_TIER",
    "DEFAULT_PROVIDER",
    "CONTEXT_DEFAULTS",
    "get_context_registry",
    # Model constants (used by provider backends for defaults)
    "GEMINI_FLASH",
    "GPT_5",
    "CLAUDE_SONNET_4",
    # Unified API
    "generate",
    "generate_with_result",
    "agenerate",
    "resolve_provider",
    # Utilities
    "log_token_usage",
    "calc_token_cost",
    "calc_agent_cost",
    "get_usage_cost",
    "iter_token_log",
    "get_model_provider",
]
