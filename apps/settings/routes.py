# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from apps.utils import log_app_action
from convey import state
from think.utils import get_config as get_journal_config

settings_bp = Blueprint(
    "app:settings",
    __name__,
    url_prefix="/app/settings",
)


# API keys that can be configured in the env section
# Used for system env checks and allowed env fields validation
API_KEY_ENV_VARS = [
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "REVAI_ACCESS_TOKEN",
]


@settings_bp.route("/api/config")
def get_config() -> Any:
    """Return the journal configuration.

    The env section is masked for security - returns boolean indicating
    whether each key is configured rather than the actual values.

    Also returns system_env with boolean status for keys available from
    the system environment (shell env + .env file).
    """
    try:
        config = get_journal_config()
        # Mask env values - return True/False for whether key is set in journal config
        if "env" in config:
            config["env"] = {k: bool(v) for k, v in config["env"].items()}

        # Add system_env - keys available from os.getenv (shell + .env)
        config["system_env"] = {k: bool(os.getenv(k)) for k in API_KEY_ENV_VARS}

        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/config", methods=["PUT"])
def update_config() -> Any:
    """Update the journal configuration.

    Accepts JSON with a 'section' key indicating which config section to update,
    and a 'data' key containing the fields to update. Supported sections:
    - identity: User profile (name, preferred, bio, pronouns, aliases, etc.)
    - transcribe: Transcription settings (device, model, compute_type)
    - convey: Web app settings (password)
    - env: API keys (GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, REVAI_ACCESS_TOKEN)

    Note: Model/provider configuration is done via the 'providers' section in
    journal.json. See docs/JOURNAL.md for the providers config format.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        section = request_data.get("section")
        data = request_data.get("data", {})

        # Backward compatibility: if no section specified but identity key exists
        if not section and "identity" in request_data:
            section = "identity"
            data = request_data["identity"]

        if not section:
            return jsonify({"error": "No section specified"}), 400

        # Define allowed fields per section
        # For transcribe, we have flat fields plus nested backend configs
        allowed_sections = {
            "identity": [
                "name",
                "preferred",
                "bio",
                "pronouns",
                "aliases",
                "email_addresses",
                "timezone",
            ],
            "transcribe": ["backend", "enrich", "preserve_all", "noise_upgrade"],
            "convey": ["password"],
            "env": API_KEY_ENV_VARS,
        }

        # Nested config schemas for transcribe backends - built from BACKEND_METADATA
        from observe.transcribe import BACKEND_METADATA

        transcribe_nested = {
            name: meta.get("settings", [])
            for name, meta in BACKEND_METADATA.items()
            if meta.get("settings")
        }

        if section not in allowed_sections:
            return jsonify({"error": f"Unknown section: {section}"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "journal.json"

        # Load existing config
        old_config = get_journal_config()
        config = get_journal_config()

        # Ensure section exists
        if section not in config:
            config[section] = {}

        # Track changes for logging
        changed_fields = {}
        old_section = old_config.get(section, {})

        # Update only allowed fields
        for key in allowed_sections[section]:
            if key in data:
                new_value = data[key]
                old_value = old_section.get(key)
                if old_value != new_value:
                    changed_fields[key] = {"old": old_value, "new": new_value}
                config[section][key] = new_value

        # Handle nested backend configs for transcribe section
        if section == "transcribe":
            for backend_key, allowed_keys in transcribe_nested.items():
                if backend_key in data and isinstance(data[backend_key], dict):
                    # Ensure nested dict exists
                    if backend_key not in config[section]:
                        config[section][backend_key] = {}
                    old_backend = old_section.get(backend_key, {})
                    # Update only allowed nested fields
                    for nested_key in allowed_keys:
                        if nested_key in data[backend_key]:
                            new_value = data[backend_key][nested_key]
                            old_value = old_backend.get(nested_key)
                            if old_value != new_value:
                                changed_fields[f"{backend_key}.{nested_key}"] = {
                                    "old": old_value,
                                    "new": new_value,
                                }
                            config[section][backend_key][nested_key] = new_value

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log if something changed (don't log sensitive values)
        if changed_fields:
            log_fields = changed_fields
            if section == "convey" and "password" in log_fields:
                # Don't log actual password values
                log_fields = {"password": {"old": "***", "new": "***"}}
            elif section == "env":
                # Don't log actual API key values
                log_fields = {k: {"old": "***", "new": "***"} for k in changed_fields}

            log_app_action(
                app="settings",
                facet=None,
                action=f"{section}_update",
                params={"changed_fields": log_fields},
            )

        # Mask env values in response
        if "env" in config:
            config["env"] = {k: bool(v) for k, v in config["env"].items()}

        return jsonify({"success": True, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Transcribe API
# ---------------------------------------------------------------------------


@settings_bp.route("/api/transcribe")
def get_transcribe() -> Any:
    """Return transcribe backend configuration.

    Returns:
        - backends: List of available backends with metadata
        - api_keys: Boolean status for each backend's API key
        - config: Current transcribe config from journal
    """
    try:
        from observe.transcribe import get_backend_list

        config = get_journal_config()
        transcribe_config = config.get("transcribe", {})

        # Get backends list from registry
        backends = get_backend_list()

        # Check API key status for each backend
        api_keys = {}
        for backend in backends:
            env_key = backend.get("env_key")
            if env_key:
                api_keys[backend["name"]] = bool(os.getenv(env_key))
            else:
                api_keys[backend["name"]] = True  # Local backends always available

        return jsonify(
            {
                "backends": backends,
                "api_keys": api_keys,
                "config": transcribe_config,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Providers API
# ---------------------------------------------------------------------------

VALID_TIERS = {1, 2, 3}


@settings_bp.route("/api/providers")
def get_providers() -> Any:
    """Return providers configuration with context defaults and API key status.

    Returns:
        - providers: List of available providers with labels
        - default: Current default provider and tier
        - contexts: Configured context overrides from journal.json
        - context_defaults: Context registry with labels/groups for UI
          (includes muse configs with has_tools, schedule, disabled, extract)
        - api_keys: Boolean status for each provider's API key
    """
    try:
        from think.models import (
            DEFAULT_PROVIDER,
            DEFAULT_TIER,
            get_context_registry,
        )
        from think.providers import get_provider_list
        from think.utils import get_muse_configs

        config = get_journal_config()
        providers_config = config.get("providers", {})

        # Get default settings
        default = providers_config.get("default", {})
        default_provider = default.get("provider", DEFAULT_PROVIDER)
        default_tier = default.get("tier", DEFAULT_TIER)

        # Get context overrides from config
        contexts = providers_config.get("contexts", {})

        # Build context defaults with metadata for UI (uses dynamic registry)
        context_defaults = {}
        for pattern, ctx_config in get_context_registry().items():
            context_defaults[pattern] = {
                "tier": ctx_config["tier"],
                "label": ctx_config["label"],
                "group": ctx_config["group"],
            }
            # Include has_tools for muse contexts
            if "has_tools" in ctx_config:
                context_defaults[pattern]["has_tools"] = ctx_config["has_tools"]

        # Enhance muse contexts with additional metadata from get_muse_configs
        from think.utils import key_to_context

        muse_configs = get_muse_configs(include_disabled=True)
        for key, info in muse_configs.items():
            context_key = key_to_context(key)

            if context_key in context_defaults:
                # Add muse-specific fields
                if "schedule" in info:
                    context_defaults[context_key]["schedule"] = info["schedule"]
                context_defaults[context_key]["disabled"] = info.get("disabled", False)
                # Include extract for generators with occurrence/anticipation hooks
                hook = info.get("hook")
                has_extraction = (
                    isinstance(hook, dict)
                    and hook.get("post") in ("occurrence", "anticipation")
                ) or hook in ("occurrence", "anticipation")
                if has_extraction:
                    context_defaults[context_key]["extract"] = info.get("extract", True)

        # Get providers list from registry
        providers_list = get_provider_list()

        # Check API key status for each provider using os.getenv()
        # This reflects the true runtime availability (shell env + .env + journal config)
        api_keys = {}
        for p in providers_list:
            env_key = p.get("env_key", "")
            api_keys[p["name"]] = bool(os.getenv(env_key)) if env_key else False

        return jsonify(
            {
                "providers": providers_list,
                "default": {
                    "provider": default_provider,
                    "tier": default_tier,
                },
                "contexts": contexts,
                "context_defaults": context_defaults,
                "api_keys": api_keys,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/providers", methods=["PUT"])
def update_providers() -> Any:
    """Update providers configuration.

    Accepts JSON with optional keys:
        - default: {provider, tier} - Set default provider and/or tier
        - contexts: {pattern: {provider?, tier?, disabled?, extract?} | null}
          Set or clear context overrides

    Setting a context to null removes the override.
    For muse contexts, disabled and extract can also be set.
    """
    try:
        from think.providers import PROVIDER_REGISTRY

        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "journal.json"

        # Load existing config
        config = get_journal_config()
        old_providers = copy.deepcopy(config.get("providers", {}))

        # Ensure providers section exists
        if "providers" not in config:
            config["providers"] = {}

        changed_fields = {}

        # Handle default updates
        if "default" in request_data:
            default_data = request_data["default"]
            if "default" not in config["providers"]:
                config["providers"]["default"] = {}

            old_default = old_providers.get("default", {})

            # Validate and update provider
            if "provider" in default_data:
                provider = default_data["provider"]
                if provider not in PROVIDER_REGISTRY:
                    return (
                        jsonify(
                            {
                                "error": f"Invalid provider: {provider}. "
                                f"Must be one of: {', '.join(sorted(PROVIDER_REGISTRY.keys()))}"
                            }
                        ),
                        400,
                    )
                if old_default.get("provider") != provider:
                    changed_fields["default.provider"] = {
                        "old": old_default.get("provider"),
                        "new": provider,
                    }
                config["providers"]["default"]["provider"] = provider

            # Validate and update tier
            if "tier" in default_data:
                tier = default_data["tier"]
                if tier not in VALID_TIERS:
                    return (
                        jsonify(
                            {"error": f"Invalid tier: {tier}. Must be 1, 2, or 3."}
                        ),
                        400,
                    )
                if old_default.get("tier") != tier:
                    changed_fields["default.tier"] = {
                        "old": old_default.get("tier"),
                        "new": tier,
                    }
                config["providers"]["default"]["tier"] = tier

        # Handle context overrides
        if "contexts" in request_data:
            contexts_data = request_data["contexts"]
            if "contexts" not in config["providers"]:
                config["providers"]["contexts"] = {}

            old_contexts = old_providers.get("contexts", {})

            for pattern, ctx_config in contexts_data.items():
                old_ctx = old_contexts.get(pattern)

                # null means remove the override
                if ctx_config is None:
                    if pattern in config["providers"]["contexts"]:
                        changed_fields[f"contexts.{pattern}"] = {
                            "old": old_ctx,
                            "new": None,
                        }
                        del config["providers"]["contexts"][pattern]
                    continue

                # Validate provider if specified
                if "provider" in ctx_config:
                    provider = ctx_config["provider"]
                    if provider not in PROVIDER_REGISTRY:
                        return (
                            jsonify(
                                {"error": f"Invalid provider for {pattern}: {provider}"}
                            ),
                            400,
                        )

                # Validate tier if specified
                if "tier" in ctx_config:
                    tier = ctx_config["tier"]
                    if tier not in VALID_TIERS:
                        return (
                            jsonify({"error": f"Invalid tier for {pattern}: {tier}"}),
                            400,
                        )

                # Validate disabled if specified (must be boolean)
                if "disabled" in ctx_config:
                    if not isinstance(ctx_config["disabled"], bool):
                        return (
                            jsonify(
                                {"error": f"disabled for {pattern} must be a boolean"}
                            ),
                            400,
                        )

                # Validate extract if specified (must be boolean)
                if "extract" in ctx_config:
                    if not isinstance(ctx_config["extract"], bool):
                        return (
                            jsonify(
                                {"error": f"extract for {pattern} must be a boolean"}
                            ),
                            400,
                        )

                # Only store if there's something to override
                if ctx_config:
                    if old_ctx != ctx_config:
                        changed_fields[f"contexts.{pattern}"] = {
                            "old": old_ctx,
                            "new": ctx_config,
                        }
                    config["providers"]["contexts"][pattern] = ctx_config

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log if something changed
        if changed_fields:
            log_app_action(
                app="settings",
                facet=None,
                action="providers_update",
                params={"changed_fields": changed_fields},
            )

        # Return updated providers config
        return get_providers()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Generators API (compatibility layer for Settings UI)
# ---------------------------------------------------------------------------


def _build_generator_info(key: str, info: dict) -> dict:
    """Build generator info dict from muse config for Settings UI.

    Transforms muse config metadata into the format expected by the
    Settings UI Insights section.
    """
    # Determine if extraction is supported (occurrence/anticipation hooks)
    hook = info.get("hook")
    has_extraction = (
        isinstance(hook, dict) and hook.get("post") in ("occurrence", "anticipation")
    ) or hook in ("occurrence", "anticipation")

    return {
        "key": key,
        "title": info.get("title", info.get("label", key)),
        "description": info.get("description", ""),
        "source": info.get("source", "system"),
        "app": info.get("app"),
        "disabled": info.get("disabled", False),
        "extract": info.get("extract", True) if has_extraction else None,
        "has_extraction": has_extraction,
    }


@settings_bp.route("/api/generators")
def get_generators() -> Any:
    """Return generators grouped by schedule for Settings UI.

    This is a compatibility layer that transforms the unified muse config
    into the format expected by the Settings UI Insights section.

    Returns:
        - segment: List of segment-schedule generators
        - daily: List of daily-schedule generators
    """
    try:
        from think.utils import get_muse_configs

        # Get all generators (has output but no tools)
        all_generators = get_muse_configs(
            has_tools=False, has_output=True, include_disabled=True
        )

        segment = []
        daily = []

        for key, info in all_generators.items():
            gen_info = _build_generator_info(key, info)
            schedule = info.get("schedule")

            if schedule == "segment":
                segment.append(gen_info)
            elif schedule == "daily":
                daily.append(gen_info)
            # Skip generators without valid schedule

        return jsonify({"segment": segment, "daily": daily})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/generators", methods=["PUT"])
def update_generators() -> Any:
    """Update generator settings via providers.contexts.

    This is a compatibility layer that accepts the old generators API
    format and stores settings in the unified providers.contexts location.

    Accepts JSON with generator keys mapping to {disabled?, extract?}.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "journal.json"

        # Load existing config
        config = get_journal_config()
        old_providers = copy.deepcopy(config.get("providers", {}))

        if "providers" not in config:
            config["providers"] = {}
        if "contexts" not in config["providers"]:
            config["providers"]["contexts"] = {}

        old_contexts = old_providers.get("contexts", {})
        changed_fields = {}

        from think.utils import key_to_context

        for key, updates in request_data.items():
            if not isinstance(updates, dict):
                continue

            context_key = key_to_context(key)

            # Get or create context config
            ctx_config = config["providers"]["contexts"].get(context_key, {})
            old_ctx = old_contexts.get(context_key, {})

            # Apply updates
            if "disabled" in updates:
                if not isinstance(updates["disabled"], bool):
                    return (
                        jsonify({"error": f"disabled must be boolean for {key}"}),
                        400,
                    )
                ctx_config["disabled"] = updates["disabled"]

            if "extract" in updates:
                if not isinstance(updates["extract"], bool):
                    return jsonify({"error": f"extract must be boolean for {key}"}), 400
                ctx_config["extract"] = updates["extract"]

            # Only store if there's something to override
            if ctx_config:
                if old_ctx != ctx_config:
                    changed_fields[f"contexts.{context_key}"] = {
                        "old": old_ctx if old_ctx else None,
                        "new": ctx_config,
                    }
                config["providers"]["contexts"][context_key] = ctx_config

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log if something changed
        if changed_fields:
            log_app_action(
                app="settings",
                facet=None,
                action="generators_update",
                params={"changed_fields": changed_fields},
            )

        # Return updated generators
        return get_generators()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------

VALID_IMPORTANCE = {"high", "normal", "low", "ignore"}


@settings_bp.route("/api/vision")
def get_vision() -> Any:
    """Return vision configuration with category defaults.

    Returns:
        - max_extractions: Current max extractions setting (default: 20)
        - categories: Dict of category overrides from config
        - category_defaults: Discovered categories with their defaults
    """
    try:
        from observe.describe import CATEGORIES
        from observe.extract import DEFAULT_MAX_EXTRACTIONS

        config = get_journal_config()
        describe_config = config.get("describe", {})

        # Build category defaults from discovered categories
        category_defaults = {}
        for name, meta in CATEGORIES.items():
            category_defaults[name] = {
                "label": meta.get("label", name.replace("_", " ").title()),
                "group": meta.get("group", "Screen Analysis"),
                "extraction": meta.get("extraction", ""),
                "importance": meta.get("importance", "normal"),
            }

        return jsonify(
            {
                "max_extractions": describe_config.get(
                    "max_extractions", DEFAULT_MAX_EXTRACTIONS
                ),
                "categories": describe_config.get("categories", {}),
                "category_defaults": category_defaults,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/vision", methods=["PUT"])
def update_vision() -> Any:
    """Update vision configuration.

    Accepts JSON with optional keys:
        - max_extractions: int (5-100) - Maximum frames to extract
        - categories: {name: {importance?, extraction?} | null} - Category overrides

    Setting a category to null removes its overrides.
    """
    try:
        from observe.describe import CATEGORIES

        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "journal.json"

        # Load existing config
        config = get_journal_config()
        old_describe = copy.deepcopy(config.get("describe", {}))

        # Ensure describe section exists
        if "describe" not in config:
            config["describe"] = {}

        changed_fields = {}

        # Handle max_extractions update
        if "max_extractions" in request_data:
            max_ext = request_data["max_extractions"]
            if not isinstance(max_ext, int) or max_ext < 5 or max_ext > 100:
                return (
                    jsonify(
                        {
                            "error": "max_extractions must be an integer between 5 and 100"
                        }
                    ),
                    400,
                )
            old_val = old_describe.get("max_extractions")
            if old_val != max_ext:
                changed_fields["max_extractions"] = {"old": old_val, "new": max_ext}
            config["describe"]["max_extractions"] = max_ext

        # Handle category overrides
        if "categories" in request_data:
            categories_data = request_data["categories"]
            if "categories" not in config["describe"]:
                config["describe"]["categories"] = {}

            old_categories = old_describe.get("categories", {})

            for name, cat_config in categories_data.items():
                # Validate category exists
                if name not in CATEGORIES:
                    return (
                        jsonify({"error": f"Unknown category: {name}"}),
                        400,
                    )

                old_cat = old_categories.get(name)

                # null means remove the override
                if cat_config is None:
                    if name in config["describe"]["categories"]:
                        changed_fields[f"categories.{name}"] = {
                            "old": old_cat,
                            "new": None,
                        }
                        del config["describe"]["categories"][name]
                    continue

                # Validate importance if specified
                if "importance" in cat_config:
                    importance = cat_config["importance"]
                    if importance not in VALID_IMPORTANCE:
                        return (
                            jsonify(
                                {
                                    "error": f"Invalid importance for {name}: {importance}. "
                                    f"Must be one of: {', '.join(sorted(VALID_IMPORTANCE))}"
                                }
                            ),
                            400,
                        )

                # Validate extraction if specified (must be string)
                if "extraction" in cat_config:
                    extraction = cat_config["extraction"]
                    if not isinstance(extraction, str):
                        return (
                            jsonify(
                                {"error": f"extraction for {name} must be a string"}
                            ),
                            400,
                        )

                # Only store if there's something to override
                if cat_config:
                    if old_cat != cat_config:
                        changed_fields[f"categories.{name}"] = {
                            "old": old_cat,
                            "new": cat_config,
                        }
                    config["describe"]["categories"][name] = cat_config

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log if something changed
        if changed_fields:
            log_app_action(
                app="settings",
                facet=None,
                action="vision_update",
                params={"changed_fields": changed_fields},
            )

        # Return updated vision config
        return get_vision()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Observe API
# ---------------------------------------------------------------------------

# Default observe configuration - single source of truth for all defaults
OBSERVE_TMUX_DEFAULTS = {
    "enabled": True,
    "capture_interval": 5,
    "capture_interval_min": 1,
    "capture_interval_max": 60,
}


@settings_bp.route("/api/observe")
def get_observe() -> Any:
    """Return observe configuration with defaults and validation bounds.

    Returns:
        - tmux: Tmux capture settings
            - enabled: Whether tmux capture is enabled
            - capture_interval: Seconds between terminal captures
        - defaults: Default values and validation bounds for UI
    """
    try:
        config = get_journal_config()
        observe_config = config.get("observe", {})
        tmux_config = observe_config.get("tmux", {})

        # Build result with user config merged over defaults
        result = {
            "tmux": {
                "enabled": tmux_config.get("enabled", OBSERVE_TMUX_DEFAULTS["enabled"]),
                "capture_interval": tmux_config.get(
                    "capture_interval", OBSERVE_TMUX_DEFAULTS["capture_interval"]
                ),
            },
            "defaults": {
                "tmux": OBSERVE_TMUX_DEFAULTS,
            },
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/observe", methods=["PUT"])
def update_observe() -> Any:
    """Update observe configuration.

    Accepts JSON with optional keys:
        - tmux: {enabled?: bool, capture_interval?: int}
            - enabled: Whether tmux capture is enabled
            - capture_interval: Seconds between terminal captures (1-60)
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "journal.json"

        # Load existing config
        config = get_journal_config()
        old_observe = copy.deepcopy(config.get("observe", {}))

        # Ensure observe section exists
        if "observe" not in config:
            config["observe"] = {}

        changed_fields = {}

        # Handle tmux settings
        if "tmux" in request_data:
            tmux_data = request_data["tmux"]
            if not isinstance(tmux_data, dict):
                return jsonify({"error": "tmux must be an object"}), 400

            if "tmux" not in config["observe"]:
                config["observe"]["tmux"] = {}

            old_tmux = old_observe.get("tmux", {})
            defaults = OBSERVE_TMUX_DEFAULTS

            # Validate and update enabled
            if "enabled" in tmux_data:
                enabled = tmux_data["enabled"]
                if not isinstance(enabled, bool):
                    return jsonify({"error": "tmux.enabled must be a boolean"}), 400
                if enabled != old_tmux.get("enabled", defaults["enabled"]):
                    config["observe"]["tmux"]["enabled"] = enabled
                    changed_fields["tmux.enabled"] = enabled

            # Validate and update capture_interval
            if "capture_interval" in tmux_data:
                capture_interval = tmux_data["capture_interval"]
                min_val = defaults["capture_interval_min"]
                max_val = defaults["capture_interval_max"]
                if (
                    not isinstance(capture_interval, int)
                    or capture_interval < min_val
                    or capture_interval > max_val
                ):
                    return (
                        jsonify(
                            {
                                "error": f"tmux.capture_interval must be an integer between {min_val} and {max_val}"
                            }
                        ),
                        400,
                    )
                if capture_interval != old_tmux.get(
                    "capture_interval", defaults["capture_interval"]
                ):
                    config["observe"]["tmux"]["capture_interval"] = capture_interval
                    changed_fields["tmux.capture_interval"] = capture_interval

        # Save config if changed
        if changed_fields:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")

            log_app_action(
                app="settings",
                facet=None,
                action="observe_update",
                params={"changed_fields": changed_fields},
            )

        return get_observe()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet", methods=["POST"])
def create_facet() -> Any:
    """Create a new facet.

    Accepts JSON with:
        title: Display title (required)
        emoji: Icon emoji (optional, default: "📦")
        color: Hex color (optional, default: "#667eea")

    The facet name (slug) is auto-generated from the title.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        title = data.get("title", "").strip()
        if not title:
            return jsonify({"error": "Title is required"}), 400

        # Optional fields with defaults
        emoji = data.get("emoji", "📦")
        color = data.get("color", "#667eea")

        # Generate slug from title: lowercase, replace spaces/special chars with hyphens
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
        slug = slug.strip("-")  # Remove leading/trailing hyphens

        if not slug:
            return (
                jsonify({"error": "Title must contain at least one letter or number"}),
                400,
            )

        # Check for conflicts with existing facets
        from think.facets import get_facets

        existing = get_facets()
        if slug in existing:
            return jsonify({"error": f"Facet '{slug}' already exists"}), 409

        # Create facet directory and config
        facet_path = Path(state.journal_root) / "facets" / slug
        facet_path.mkdir(parents=True, exist_ok=True)

        config = {
            "title": title,
            "description": "",
            "color": color,
            "emoji": emoji,
        }

        config_file = facet_path / "facet.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log the creation
        log_app_action(
            app="settings",
            facet=slug,
            action="facet_create",
            params={"title": title, "emoji": emoji, "color": color},
        )

        return jsonify({"success": True, "facet": slug, "config": config}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet/<facet_name>")
def get_facet_config(facet_name: str) -> Any:
    """Get configuration for a specific facet."""
    try:
        from think.facets import get_facets

        facets = get_facets()
        if facet_name not in facets:
            return jsonify({"error": "Facet not found"}), 404

        return jsonify({"facet": facet_name, "config": facets[facet_name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet/<facet_name>", methods=["PUT"])
def update_facet_config(facet_name: str) -> Any:
    """Update configuration for a specific facet."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Build path to facet config file
        facet_path = Path(state.journal_root) / "facets" / facet_name
        config_file = facet_path / "facet.json"

        if not facet_path.exists():
            return jsonify({"error": "Facet not found"}), 404

        # Read existing config or create new one
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        # Track changes for logging
        changed_fields = {}
        allowed_fields = ["title", "description", "color", "emoji", "muted"]
        for field in allowed_fields:
            if field in data:
                old_value = config.get(field)
                new_value = data[field]
                if old_value != new_value:
                    changed_fields[field] = {"old": old_value, "new": new_value}
                config[field] = new_value

        # Write back to file
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log only if something actually changed
        if changed_fields:
            log_app_action(
                app="settings",
                facet=facet_name,
                action="facet_update",
                params={"changed_fields": changed_fields},
            )

        return jsonify({"success": True, "facet": facet_name, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_logs_from_dir(logs_dir: Path, cursor: str | None) -> dict:
    """Load action logs from a directory, one day at a time.

    Args:
        logs_dir: Path to logs directory containing YYYYMMDD.jsonl files
        cursor: Optional YYYYMMDD - load the day before this date

    Returns:
        Dict with {day, entries, next_cursor}
    """
    if not logs_dir.exists():
        return {"day": None, "entries": [], "next_cursor": None}

    # Find all log files sorted newest first
    log_files = sorted(
        [f for f in logs_dir.iterdir() if re.fullmatch(r"\d{8}\.jsonl", f.name)],
        key=lambda f: f.stem,
        reverse=True,
    )

    if not log_files:
        return {"day": None, "entries": [], "next_cursor": None}

    # Apply cursor filter if provided
    if cursor:
        log_files = [f for f in log_files if f.stem < cursor]

    if not log_files:
        return {"day": None, "entries": [], "next_cursor": None}

    # Load the first (newest) day
    target_file = log_files[0]
    day = target_file.stem
    entries = []

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        pass

    # Reverse to show newest first within the day
    entries.reverse()

    # Determine next cursor
    next_cursor = log_files[1].stem if len(log_files) > 1 else None

    return {"day": day, "entries": entries, "next_cursor": next_cursor}


@settings_bp.route("/api/logs")
def get_journal_logs() -> Any:
    """Get journal-level action logs, one day at a time.

    These are actions not tied to a specific facet, such as settings changes,
    remote observer management, and other journal-wide operations.

    Query params:
        cursor: Optional YYYYMMDD - load the day before this date

    Returns:
        {day, entries, next_cursor} where next_cursor is null if no more days
    """
    logs_dir = Path(state.journal_root) / "config" / "actions"
    cursor = request.args.get("cursor")
    return jsonify(_get_logs_from_dir(logs_dir, cursor))


@settings_bp.route("/api/facet/<facet_name>/logs")
def get_facet_logs(facet_name: str) -> Any:
    """Get action logs for a facet, one day at a time.

    Query params:
        cursor: Optional YYYYMMDD - load the day before this date

    Returns:
        {day, entries, next_cursor} where next_cursor is null if no more days
    """
    logs_dir = Path(state.journal_root) / "facets" / facet_name / "logs"
    cursor = request.args.get("cursor")
    return jsonify(_get_logs_from_dir(logs_dir, cursor))
