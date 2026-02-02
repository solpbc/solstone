# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Muse prompt loading and configuration utilities.

This module provides all functionality for loading, parsing, and configuring
muse prompts (agents and generators) from muse/*.md and apps/*/muse/*.md.

Key functions:
- load_prompt(): Load and parse .md prompt files with template substitution
- get_muse_configs(): Discover all muse configs with filtering
- get_agent(): Load complete agent configuration by name
- compose_instructions(): Build system/user prompts from instruction config
- Hook loading: load_pre_hook(), load_post_hook()
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from string import Template
from typing import Any, Callable, NamedTuple

import frontmatter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUSE_DIR = Path(__file__).parent.parent / "muse"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Cached raw template content loaded from think/templates/*.md
_templates_cache: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Template Loading
# ---------------------------------------------------------------------------


def _load_raw_templates() -> dict[str, str]:
    """Load raw template files from think/templates/ directory.

    Templates are cached on first load. Each .md file becomes a template
    variable named after its stem (e.g., daily_preamble.md -> $daily_preamble).

    Returns
    -------
    dict[str, str]
        Mapping of template variable names to their raw content (no substitution).
    """
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache

    _templates_cache = {}
    if TEMPLATES_DIR.is_dir():
        for md_path in TEMPLATES_DIR.glob("*.md"):
            var_name = md_path.stem
            try:
                post = frontmatter.load(
                    md_path,
                )
                _templates_cache[var_name] = post.content.strip()
            except Exception as exc:
                logging.debug("Failed to load template %s: %s", md_path, exc)

    return _templates_cache


def _load_templates(template_vars: dict[str, str] | None = None) -> dict[str, str]:
    """Load and substitute template files from think/templates/ directory.

    Raw templates are cached, but substitution is performed on each call
    to support context-dependent variables like $date and $segment_start.

    Parameters
    ----------
    template_vars:
        Optional variables to substitute into templates. Templates can use
        identity vars ($name, $preferred), context vars ($day, $date,
        $segment_start, $segment_end), and other template vars.

    Returns
    -------
    dict[str, str]
        Mapping of template variable names to their substituted content.
    """
    raw_templates = _load_raw_templates()

    if not template_vars:
        return dict(raw_templates)

    # Substitute variables into each template
    substituted = {}
    for var_name, content in raw_templates.items():
        try:
            template = Template(content)
            substituted[var_name] = template.safe_substitute(template_vars)
        except Exception as exc:
            logging.debug("Template substitution failed for %s: %s", var_name, exc)
            substituted[var_name] = content

    return substituted


# ---------------------------------------------------------------------------
# Prompt Loading
# ---------------------------------------------------------------------------


class PromptContent(NamedTuple):
    """Container for prompt text, metadata, and its resolved path."""

    text: str
    path: Path
    metadata: dict[str, Any] = {}


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt file cannot be located."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Prompt file not found: {path}")


def _flatten_identity_to_template_vars(identity: dict[str, Any]) -> dict[str, str]:
    """Flatten identity config into template variables with uppercase-first versions.

    Parameters
    ----------
    identity:
        Identity configuration dictionary from get_config()['identity'].

    Returns
    -------
    dict[str, str]
        Template variables including flattened nested objects and uppercase-first versions.
        For example:
        - 'name' → identity['name']
        - 'pronouns_possessive' → identity['pronouns']['possessive']
        - 'Pronouns_possessive' → identity['pronouns']['possessive'].capitalize()
        - 'bio' → identity['bio']
    """
    template_vars: dict[str, str] = {}

    # Flatten top-level and nested values
    for key, value in identity.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries with underscore separator
            for subkey, subvalue in value.items():
                var_name = f"{key}_{subkey}"
                template_vars[var_name] = str(subvalue)
                # Create uppercase-first version
                template_vars[var_name.capitalize()] = str(subvalue).capitalize()
        elif isinstance(value, (str, int, float, bool)):
            # Top-level scalar values
            template_vars[key] = str(value)
            # Create uppercase-first version
            template_vars[key.capitalize()] = str(value).capitalize()

    return template_vars


def load_prompt(
    name: str,
    base_dir: str | Path | None = None,
    *,
    include_journal: bool = False,
    context: dict[str, Any] | None = None,
) -> PromptContent:
    """Return the text contents, metadata, and path for a ``.md`` prompt file.

    Prompt files use JSON frontmatter for metadata. Supports Python
    string.Template variable substitution using:
    - Identity config from get_config()['identity']:
      - Top-level fields: $name, $preferred, $bio, $timezone
      - Nested fields with underscores: $pronouns_possessive, $pronouns_subject
      - Uppercase-first versions: $Pronouns_possessive, $Name, $Bio
    - Templates from think/templates/*.md:
      - Each file becomes a variable named after its stem
      - Example: daily_preamble.md -> $daily_preamble
      - Templates are pre-processed with identity and context vars, so templates
        can use $date, $preferred, etc. before being substituted into prompts

    Callers can provide additional context variables via the ``context`` parameter.
    Context variables override identity and template variables if there's a collision.
    Uppercase-first versions are automatically created for context variables.

    Parameters
    ----------
    name:
        Base filename of the prompt without the ``.md`` suffix. If the suffix is
        included, it will not be duplicated.
    base_dir:
        Optional directory containing the prompt file. Defaults to the directory
        of this module when not provided.
    include_journal:
        If True, prepends the content of ``think/journal.md`` to the requested
        prompt. Defaults to False. Context variables are passed through to the
        journal template as well.
    context:
        Optional dictionary of additional template variables. Values are converted
        to strings. For each key, an uppercase-first version is also created
        (e.g., ``{"day": "20250110"}`` adds both ``$day`` and ``$Day``).

    Returns
    -------
    PromptContent
        The prompt text (with surrounding whitespace removed and template variables
        substituted), the resolved path to the ``.md`` file, and metadata from
        the JSON frontmatter.
    """
    from think.utils import get_config

    if not name:
        raise ValueError("Prompt name must be provided")

    if name.endswith(".md"):
        filename = name
    else:
        filename = f"{name}.md"

    prompt_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
    prompt_path = prompt_dir / filename
    try:
        post = frontmatter.load(
            prompt_path,
        )
        text = post.content.strip()
        metadata = dict(post.metadata)
    except FileNotFoundError as exc:  # pragma: no cover - caller handles missing prompt
        raise PromptNotFoundError(prompt_path) from exc

    # Perform template substitution
    try:
        config = get_config()
        identity = config.get("identity", {})
        template_vars = _flatten_identity_to_template_vars(identity)

        # Merge caller-provided context (overrides identity vars if collision)
        if context:
            for key, value in context.items():
                str_value = str(value)
                template_vars[key] = str_value
                # Add uppercase-first version
                template_vars[key.capitalize()] = str_value.capitalize()

        # Load templates with identity and context vars so templates can use them
        templates = _load_templates(template_vars)
        template_vars.update(templates)

        # Use safe_substitute to avoid errors for undefined variables
        template = Template(text)
        text = template.safe_substitute(template_vars)
    except Exception as exc:
        # Log but don't fail - return original text if substitution fails
        logging.debug("Template substitution failed for %s: %s", prompt_path, exc)

    # Prepend journal content if requested
    if include_journal and name != "journal":
        journal_content = load_prompt("journal", context=context)
        text = f"{journal_content.text}\n\n{text}"

    return PromptContent(text=text, path=prompt_path, metadata=metadata)


# ---------------------------------------------------------------------------
# Prompt Metadata Loading
# ---------------------------------------------------------------------------


def _load_prompt_metadata(md_path: Path) -> dict[str, object]:
    """Load prompt metadata from .md file with JSON frontmatter.

    Parameters
    ----------
    md_path:
        Path to the .md prompt file with JSON frontmatter.

    Returns
    -------
    dict
        Metadata dict with path, mtime, color, and frontmatter fields.
    """
    mtime = int(md_path.stat().st_mtime)
    info: dict[str, object] = {
        "path": str(md_path),
        "mtime": mtime,
    }

    try:
        post = frontmatter.load(
            md_path,
        )
        if post.metadata:
            info.update(post.metadata)
    except Exception as exc:  # pragma: no cover - metadata optional
        logging.debug("Error reading frontmatter from %s: %s", md_path, exc)

    # Apply default color if not specified
    if "color" not in info:
        info["color"] = "#6c757d"

    return info


# ---------------------------------------------------------------------------
# Muse Config Discovery
# ---------------------------------------------------------------------------


def key_to_context(key: str) -> str:
    """Convert muse config key to context pattern.

    Parameters
    ----------
    key:
        Muse config key in format "name" (system) or "app:name" (app).

    Returns
    -------
    str
        Context pattern: "muse.system.{name}" or "muse.{app}.{name}".

    Examples
    --------
    >>> key_to_context("meetings")
    'muse.system.meetings'
    >>> key_to_context("entities:observer")
    'muse.entities.observer'
    """
    if ":" in key:
        app, name = key.split(":", 1)
        return f"muse.{app}.{name}"
    return f"muse.system.{key}"


def get_output_topic(key: str) -> str:
    """Convert agent/generator key to filesystem-safe basename (no extension).

    Parameters
    ----------
    key:
        Generator key in format "topic" (system) or "app:topic" (app).

    Returns
    -------
    str
        Filesystem-safe name: "topic" or "_app_topic".

    Examples
    --------
    >>> get_output_topic("activity")
    'activity'
    >>> get_output_topic("chat:sentiment")
    '_chat_sentiment'
    """
    if ":" in key:
        app, topic = key.split(":", 1)
        return f"_{app}_{topic}"
    return key


def get_output_path(
    day_dir: "os.PathLike[str]",
    key: str,
    segment: str | None = None,
    output_format: str | None = None,
    facet: str | None = None,
) -> Path:
    """Return output path for generator agent output.

    Shared utility for determining where to write generator results.
    Used by think/agents.py and think/cortex.py.

    Parameters
    ----------
    day_dir:
        Day directory path (YYYYMMDD).
    key:
        Generator key or agent name (e.g., "activity", "chat:sentiment",
        "decisionalizer", "entities:observer").
    segment:
        Optional segment key (HHMMSS_LEN) for segment-level output.
    output_format:
        Output format - "json" for JSON, anything else for markdown.
    facet:
        Optional facet name for multi-facet agents. When provided, the facet
        is appended to the filename (e.g., "newsletter_work.md").

    Returns
    -------
    Path
        Output file path:
        - With segment: YYYYMMDD/{segment}/{topic}.{ext}
        - Without segment: YYYYMMDD/agents/{topic}.{ext}
        - With facet: {topic}_{facet}.{ext} instead of {topic}.{ext}
        Where topic is derived from key and ext is "json" or "md".
    """
    day = Path(day_dir)
    topic = get_output_topic(key)
    ext = "json" if output_format == "json" else "md"

    # Append facet suffix for multi-facet agent outputs
    if facet:
        filename = f"{topic}_{facet}.{ext}"
    else:
        filename = f"{topic}.{ext}"

    if segment:
        # Segment output goes directly in segment directory
        return day / segment / filename
    else:
        # Daily output goes in agents/ subdirectory
        return day / "agents" / filename


def get_muse_configs(
    *,
    has_tools: bool | None = None,
    has_output: bool | None = None,
    schedule: str | None = None,
    include_disabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load muse configs from system and app directories.

    Unified function for loading both tool-using agents and generators from
    muse/*.md files. Filters based on presence of tools/output fields.

    Args:
        has_tools: If True, only configs with "tools" field (agents).
            If False, only configs without "tools" field.
            If None, no filtering on tools presence.
        has_output: If True, only configs with "output" field (generators).
            If False, only configs without "output" field.
            If None, no filtering on output presence.
        schedule: If provided, only configs where schedule matches this value
            (e.g., "segment", "daily").
        include_disabled: If True, include configs with disabled=True.
            Default False (for processing pipelines).

    Returns:
        Dictionary mapping config keys to their metadata including:
        - path: Path to the .md file
        - source: "system" or "app"
        - app: App name (only for app configs)
        - All fields from frontmatter
    """
    from think.utils import get_config

    configs: dict[str, dict[str, Any]] = {}

    def matches_filter(info: dict) -> bool:
        """Check if config matches the filter criteria."""
        # Check has_tools filter
        if has_tools is True and "tools" not in info:
            return False
        if has_tools is False and "tools" in info:
            return False

        # Check has_output filter
        if has_output is True and "output" not in info:
            return False
        if has_output is False and "output" in info:
            return False

        # Check specific schedule value
        if schedule is not None and info.get("schedule") != schedule:
            return False

        # Check disabled status
        if not include_disabled and info.get("disabled", False):
            return False

        return True

    # System configs from muse/
    if MUSE_DIR.is_dir():
        for md_path in sorted(MUSE_DIR.glob("*.md")):
            name = md_path.stem
            info = _load_prompt_metadata(md_path)

            if not matches_filter(info):
                continue

            info["source"] = "system"
            configs[name] = info

    # App configs from apps/*/muse/
    apps_dir = Path(__file__).parent.parent / "apps"
    if apps_dir.is_dir():
        for app_path in sorted(apps_dir.iterdir()):
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue
            app_muse_dir = app_path / "muse"
            if not app_muse_dir.is_dir():
                continue
            app_name = app_path.name
            for md_path in sorted(app_muse_dir.glob("*.md")):
                item_name = md_path.stem
                info = _load_prompt_metadata(md_path)

                if not matches_filter(info):
                    continue

                key = f"{app_name}:{item_name}"
                info["source"] = "app"
                info["app"] = app_name
                configs[key] = info

    # Merge journal config overrides from providers.contexts
    providers_config = get_config().get("providers", {})
    contexts = providers_config.get("contexts", {})

    for key, info in configs.items():
        context_key = key_to_context(key)

        # Check for exact match in contexts
        override = contexts.get(context_key)
        if override and isinstance(override, dict):
            # Merge supported override fields
            if "disabled" in override:
                info["disabled"] = override["disabled"]
            if "extract" in override:
                info["extract"] = override["extract"]
            if "tier" in override:
                info["tier"] = override["tier"]
            if "provider" in override:
                info["provider"] = override["provider"]

    # Validate: scheduled prompts must have explicit priority
    for key, info in configs.items():
        if info.get("schedule") and "priority" not in info:
            raise ValueError(
                f"Scheduled prompt '{key}' is missing required 'priority' field. "
                f"All prompts with 'schedule' must declare an explicit priority."
            )

    return configs


# ---------------------------------------------------------------------------
# Agent Resolution
# ---------------------------------------------------------------------------


def _resolve_agent_path(name: str) -> tuple[Path, str]:
    """Resolve agent name to directory path and agent filename.

    Parameters
    ----------
    name:
        Agent name - either system agent (e.g., "default") or
        app-namespaced agent (e.g., "chat:helper").

    Returns
    -------
    tuple[Path, str]
        (agent_directory, agent_name) tuple.
    """
    if ":" in name:
        # App agent: "chat:helper" -> apps/chat/muse/helper
        app, agent_name = name.split(":", 1)
        agent_dir = Path(__file__).parent.parent / "apps" / app / "muse"
    else:
        # System agent: "default" -> muse/default
        agent_dir = MUSE_DIR
        agent_name = name
    return agent_dir, agent_name


# ---------------------------------------------------------------------------
# Instructions Composition
# ---------------------------------------------------------------------------

# Default instruction configuration - all false, agents must explicitly opt-in
_DEFAULT_INSTRUCTIONS = {
    "system": None,
    "facets": False,
    "sources": {
        "audio": False,
        "screen": False,
        "agents": False,
    },
}


def _merge_instructions_config(defaults: dict, overrides: dict | None) -> dict:
    """Merge instruction config overrides into defaults.

    Handles nested "sources" dict specially.

    Parameters
    ----------
    defaults:
        Default instruction configuration.
    overrides:
        Optional overrides from .json "instructions" key.

    Returns
    -------
    dict
        Merged configuration.
    """
    if not overrides:
        return defaults.copy()

    result = defaults.copy()

    # Merge top-level keys
    for key in ("system", "facets"):
        if key in overrides:
            result[key] = overrides[key]

    # Merge sources dict if present
    if "sources" in overrides and isinstance(overrides["sources"], dict):
        result["sources"] = {**defaults.get("sources", {}), **overrides["sources"]}

    return result


def compose_instructions(
    *,
    user_prompt: str | None = None,
    user_prompt_dir: Path | None = None,
    facet: str | None = None,
    include_datetime: bool = True,
    config_overrides: dict | None = None,
) -> dict:
    """Compose instruction components for agents or generators.

    This is the shared function for building system_instruction, user_instruction,
    extra_context, and sources configuration. Both agents and generators use this
    to ensure consistent prompt composition.

    Parameters
    ----------
    user_prompt:
        Name of the user instruction prompt to load (e.g., "default" for agents).
        If None, no user_instruction is included (typical for generators).
    user_prompt_dir:
        Directory to load user_prompt from. If None, uses think/ directory.
    facet:
        Optional facet name to focus on. When provided, extra_context includes
        only this facet's info (detail level controlled by "facets" setting).
    include_datetime:
        Whether to include current date/time in extra_context. Default True
        for agents (real-time chat), typically False for generators (past analysis).
    config_overrides:
        Optional dict from .json "instructions" key. Supported keys:
        - "system": prompt name for system instruction (default: "journal")
        - "facets": false | true | "full" (default: true)
          false = skip facet context
          true = include facet context with names only
          "full" = include facet context with full descriptions
          For faceted generators, shows focused facet; for unfaceted, shows all facets.
        - "sources": {"audio": bool, "screen": bool, "agents": bool|dict}
          The "agents" source can be:
          - bool: True (all agents), False (no agents)
          - "required": all agents, fail if none found
          - dict: selective filtering, e.g., {"entities": true, "meetings": "required"}

    Returns
    -------
    dict
        Composed instruction configuration:
        - system_instruction: str - loaded from "system" prompt
        - system_prompt_name: str - name of system prompt (for cache keys)
        - user_instruction: str | None - loaded from user_prompt if provided
        - extra_context: str | None - facets + datetime
        - sources: dict - {"audio": bool, "screen": bool, "agents": bool|dict}
    """
    from datetime import datetime

    # Merge defaults with overrides
    cfg = _merge_instructions_config(_DEFAULT_INSTRUCTIONS, config_overrides)

    result: dict = {}

    # Load system instruction (None means no system prompt)
    system_name = cfg.get("system")
    if system_name:
        system_prompt = load_prompt(system_name)
        result["system_instruction"] = system_prompt.text
        result["system_prompt_name"] = system_name
    else:
        result["system_instruction"] = ""
        result["system_prompt_name"] = ""

    # Load user instruction if specified
    if user_prompt:
        base_dir = user_prompt_dir if user_prompt_dir else Path(__file__).parent
        user_prompt_obj = load_prompt(user_prompt, base_dir=base_dir)
        result["user_instruction"] = user_prompt_obj.text
    else:
        result["user_instruction"] = None

    # Build extra_context based on facets setting
    # Values: false (skip), true (names only), "full" (with descriptions)
    extra_parts = []
    facets_setting = cfg.get("facets", False)
    facets_full = facets_setting == "full"

    if facets_setting:
        if facet:
            # Focused facet mode: include only this facet's context
            try:
                from think.facets import facet_summary

                summary = facet_summary(facet, detailed=facets_full)
                extra_parts.append(f"## Facet Focus\n{summary}")
            except Exception:
                pass  # Ignore if facet can't be loaded
        else:
            # General mode: all facets
            try:
                from think.facets import facet_summaries

                summary = facet_summaries(detailed=facets_full)
                if summary and summary != "No facets found.":
                    extra_parts.append(summary)
            except Exception:
                pass  # Ignore if facets can't be loaded

    # Add current date/time if requested
    if include_datetime:
        now = datetime.now()
        try:
            import tzlocal

            local_tz = tzlocal.get_localzone()
            now_local = now.astimezone(local_tz)
            time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        except Exception:
            time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        extra_parts.append(f"## Current Date and Time\nToday is {time_str}")

    result["extra_context"] = "\n\n".join(extra_parts).strip() if extra_parts else None

    # Include sources config
    result["sources"] = cfg.get("sources", _DEFAULT_INSTRUCTIONS["sources"])

    return result


# ---------------------------------------------------------------------------
# Source Configuration Helpers
# ---------------------------------------------------------------------------


def source_is_enabled(value: bool | str | dict) -> bool:
    """Check if a source should be loaded based on its config value.

    Sources can be configured as:
    - False: don't load
    - True: load if available
    - "required": load (and generation will fail if none found)
    - dict: for agents source, selective loading (e.g., {"entities": true})

    Both True and "required" mean the source should be loaded.
    A non-empty dict means the source should be loaded (with filtering).

    Args:
        value: The source config value (bool, "required" string, or dict for agents)

    Returns:
        True if the source should be loaded, False otherwise.
    """
    if isinstance(value, dict):
        # Dict means selective loading - enabled if any agent is enabled
        return any(v is True or v == "required" for v in value.values())
    return value is True or value == "required"


def source_is_required(value: bool | str | dict) -> bool:
    """Check if a source must have content for generation to proceed.

    Args:
        value: The source config value (bool, "required" string, or dict for agents)

    Returns:
        True if the source is required (generation should skip if no content).
        For dict values, returns True if any agent is marked "required".
    """
    if isinstance(value, dict):
        return any(v == "required" for v in value.values())
    return value == "required"


def get_agent_filter(value: bool | str | dict) -> dict[str, bool | str] | None:
    """Extract agent filter from sources config.

    When agents source is a dict, returns it as filter mapping agent names
    to their enabled/required status. When agents source is bool or "required",
    returns None to indicate all agents should be loaded.

    Args:
        value: The agents source config value

    Returns:
        Dict mapping agent names to bool/"required", or None for all agents.
        Returns empty dict if value is False (no agents).

    Examples:
        >>> get_agent_filter(True)
        None  # All agents
        >>> get_agent_filter(False)
        {}  # No agents
        >>> get_agent_filter({"entities": True, "meetings": "required"})
        {"entities": True, "meetings": "required"}
    """
    if isinstance(value, dict):
        return value
    if value is False:
        return {}  # No agents
    return None  # All agents (True or "required")


# ---------------------------------------------------------------------------
# Agent Loading
# ---------------------------------------------------------------------------


def get_agent(name: str = "default", facet: str | None = None) -> dict:
    """Return complete agent configuration by name.

    Loads configuration from .md file with JSON frontmatter and instruction text,
    merges with runtime context.

    Parameters
    ----------
    name:
        Agent name to load. Can be a system agent (e.g., "default")
        or an app-namespaced agent (e.g., "chat:helper" for apps/chat/muse/helper).
    facet:
        Optional facet name to focus on. When provided, includes detailed
        information for just this facet (with full entity details) instead
        of summaries of all facets.

    Returns
    -------
    dict
        Complete agent configuration including system_instruction, user_instruction,
        extra_context, model, backend, etc.
    """
    # Resolve agent path based on namespace
    agent_dir, agent_name = _resolve_agent_path(name)

    # Verify agent prompt file exists
    md_path = agent_dir / f"{agent_name}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Agent not found: {name}")

    # Load config from frontmatter
    post = frontmatter.load(
        md_path,
    )
    config = dict(post.metadata) if post.metadata else {}

    # Extract instructions config if present
    instructions_config = config.pop("instructions", None)

    # Use compose_instructions for consistent prompt composition
    instructions = compose_instructions(
        user_prompt=agent_name,
        user_prompt_dir=agent_dir,
        facet=facet,
        include_datetime=True,
        config_overrides=instructions_config,
    )

    # Merge instruction results into config
    config["system_instruction"] = instructions["system_instruction"]
    config["user_instruction"] = instructions["user_instruction"]
    if instructions["extra_context"]:
        config["extra_context"] = instructions["extra_context"]

    # Set agent name
    config["name"] = name

    return config


# ---------------------------------------------------------------------------
# Hook Loading
# ---------------------------------------------------------------------------


def _resolve_hook_path(hook_name: str) -> Path:
    """Resolve hook name to file path.

    Resolution:
    - Named: "name" -> muse/{name}.py
    - App-qualified: "app:name" -> apps/{app}/muse/{name}.py
    - Explicit path: "path/to/hook.py" -> direct path
    """
    if "/" in hook_name or hook_name.endswith(".py"):
        return Path(hook_name)
    elif ":" in hook_name:
        app, name = hook_name.split(":", 1)
        return Path(__file__).parent.parent / "apps" / app / "muse" / f"{name}.py"
    else:
        return MUSE_DIR / f"{hook_name}.py"


def _load_hook_function(config: dict, key: str, func_name: str) -> Callable | None:
    """Load a hook function from config.

    Args:
        config: Agent/generator config dict
        key: Hook key in config ("pre" or "post")
        func_name: Function name to load ("pre_process" or "post_process")

    Returns:
        The hook function, or None if no hook configured.

    Raises:
        ValueError: If hook file doesn't define the required function.
        ImportError: If hook file cannot be loaded.
    """
    hook_config = config.get("hook")
    if not hook_config or not isinstance(hook_config, dict):
        return None

    hook_name = hook_config.get(key)
    if not hook_name:
        return None

    hook_path = _resolve_hook_path(hook_name)

    if not hook_path.exists():
        raise ImportError(f"Hook file not found: {hook_path}")

    spec = importlib.util.spec_from_file_location(
        f"{key}_hook_{hook_path.stem}", hook_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hook from {hook_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, func_name):
        raise ValueError(f"Hook {hook_path} must define a '{func_name}' function")

    process_func = getattr(module, func_name)
    if not callable(process_func):
        raise ValueError(f"Hook {hook_path} '{func_name}' must be callable")

    return process_func


def load_post_hook(config: dict) -> Callable[[str, "HookContext"], str | None] | None:
    """Load post-processing hook from config if defined.

    Hook config format: {"hook": {"post": "name"}}

    Returns:
        Post-processing function or None if no hook configured.
        Function signature: (result: str, context: HookContext) -> str | None
    """
    return _load_hook_function(config, "post", "post_process")


def load_pre_hook(config: dict) -> Callable[["PreHookContext"], dict | None] | None:
    """Load pre-processing hook from config if defined.

    Hook config format: {"hook": {"pre": "name"}}

    Returns:
        Pre-processing function or None if no hook configured.
        Function signature: (context: PreHookContext) -> dict | None
    """
    return _load_hook_function(config, "pre", "pre_process")


# Type aliases for hook context (actual TypedDicts defined in agents.py)
HookContext = dict
PreHookContext = dict
