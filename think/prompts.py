# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Core prompt loading utilities.

This module provides the foundational prompt loading functionality used by both
standalone prompts (observe/, think/*.md) and the full muse agent orchestration.

Key functions:
- load_prompt(): Load and parse .md prompt files with template substitution
- PromptContent: Named tuple for prompt text, path, and metadata

For full agent/generator orchestration (scheduling, hooks, instruction composition),
use think.muse instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import Any, NamedTuple

import frontmatter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    to support context-dependent variables like $day and $segment_start.

    Parameters
    ----------
    template_vars:
        Optional variables to substitute into templates. Templates can use
        identity vars ($name, $preferred), context vars ($day, $day_YYYYMMDD,
        $segment_start, $segment_end, $now), and other template vars.

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


def format_current_datetime() -> str:
    """Format current datetime with timezone for display.

    Returns a human-readable string like "Monday, February 3, 2025 at 10:30 AM PST".
    Falls back to timezone-naive format if tzlocal is unavailable.

    This is the single source of truth for $now template variable and
    instructions.now context formatting.
    """
    from datetime import datetime

    now = datetime.now()
    try:
        import tzlocal

        local_tz = tzlocal.get_localzone()
        now_local = now.astimezone(local_tz)
        return now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception:
        return now.strftime("%A, %B %d, %Y at %I:%M %p")


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
        can use $day, $preferred, $now, etc. before being substituted into prompts

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

        # Add $now template variable (current datetime)
        template_vars["now"] = format_current_datetime()

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
