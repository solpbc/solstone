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
            "transcribe": ["device", "model", "compute_type", "enrich", "preserve_all"],
            "convey": ["password"],
            "env": API_KEY_ENV_VARS,
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
# Providers API
# ---------------------------------------------------------------------------

VALID_PROVIDERS = {"google", "openai", "anthropic"}
VALID_TIERS = {1, 2, 3}

# Map env key names to provider names
PROVIDER_API_KEYS = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@settings_bp.route("/api/providers")
def get_providers() -> Any:
    """Return providers configuration with context defaults and API key status.

    Returns:
        - default: Current default provider and tier
        - contexts: Configured context overrides from journal.json
        - context_defaults: CONTEXT_DEFAULTS with labels/groups for UI
        - api_keys: Boolean status for each provider's API key
    """
    try:
        from muse.models import (
            CONTEXT_DEFAULTS,
            DEFAULT_PROVIDER,
            DEFAULT_TIER,
        )

        config = get_journal_config()
        providers_config = config.get("providers", {})

        # Get default settings
        default = providers_config.get("default", {})
        default_provider = default.get("provider", DEFAULT_PROVIDER)
        default_tier = default.get("tier", DEFAULT_TIER)

        # Get context overrides from config
        contexts = providers_config.get("contexts", {})

        # Build context defaults with metadata for UI
        context_defaults = {}
        for pattern, ctx_config in CONTEXT_DEFAULTS.items():
            context_defaults[pattern] = {
                "tier": ctx_config["tier"],
                "label": ctx_config["label"],
                "group": ctx_config["group"],
            }

        # Check API key status for each provider using os.getenv()
        # This reflects the true runtime availability (shell env + .env + journal config)
        api_keys = {}
        for provider, env_key in PROVIDER_API_KEYS.items():
            api_keys[provider] = bool(os.getenv(env_key))

        return jsonify(
            {
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
        - contexts: {pattern: {provider?, tier?} | null} - Set or clear context overrides

    Setting a context to null removes the override.
    """
    try:
        from muse.models import CONTEXT_DEFAULTS

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
                if provider not in VALID_PROVIDERS:
                    return (
                        jsonify(
                            {
                                "error": f"Invalid provider: {provider}. "
                                f"Must be one of: {', '.join(sorted(VALID_PROVIDERS))}"
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
                # Validate pattern exists in CONTEXT_DEFAULTS
                if pattern not in CONTEXT_DEFAULTS:
                    return (
                        jsonify({"error": f"Unknown context pattern: {pattern}"}),
                        400,
                    )

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
                    if provider not in VALID_PROVIDERS:
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
