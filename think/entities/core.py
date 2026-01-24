# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Core entity types, constants, and validation utilities.

This module provides the foundational types and functions used throughout
the entity system:
- Type aliases for better documentation
- Constants for default values and validation
- Identity and slug generation
- Type validation
"""

import hashlib
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from slugify import slugify

from think.utils import get_config

# Type alias for entity dictionaries
EntityDict = dict[str, Any]

# Default timestamp for entities without activity data (Jan 1 2026 00:00:00 UTC)
# Used as fallback in entity_last_active_ts() to ensure all entities have a sortable value
DEFAULT_ACTIVITY_TS = 1767225600000

# Standard entity types - used for UI suggestions and documentation.
# Custom types are still allowed (validated by is_valid_entity_type regex).
ENTITY_TYPES = [
    {"name": "Person"},
    {"name": "Company"},
    {"name": "Project"},
    {"name": "Tool"},
]

# Maximum length for entity slug before truncation
MAX_ENTITY_SLUG_LENGTH = 200


def get_identity_names() -> list[str]:
    """Get all names/aliases for the journal principal from identity config.

    Returns a list of names to match against entities, in display priority order:
    1. identity.preferred (nickname/preferred name) - best for display
    2. identity.name (full name)
    3. identity.aliases (list of alternative names)

    The first element (if any) is the best name for display purposes.
    Returns empty list if identity is not configured.
    """
    config = get_config()
    identity = config.get("identity", {})

    names: list[str] = []

    # Preferred name first (best for display)
    preferred = identity.get("preferred", "").strip()
    if preferred:
        names.append(preferred)

    # Full name
    name = identity.get("name", "").strip()
    if name and name not in names:
        names.append(name)

    # Aliases
    aliases = identity.get("aliases", [])
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str):
                alias = alias.strip()
                if alias and alias not in names:
                    names.append(alias)

    return names


def entity_last_active_ts(entity: EntityDict) -> int:
    """Get entity's last activity timestamp with fallback chain.

    Returns a Unix timestamp (milliseconds) representing when the entity was
    last active, using the following priority:
    1. last_seen (YYYYMMDD string, converted to local midnight)
    2. updated_at (Unix ms)
    3. attached_at (Unix ms)
    4. DEFAULT_ACTIVITY_TS (Jan 1 2026)

    This ensures all entities have a sortable timestamp value.

    Args:
        entity: Entity dictionary with optional last_seen, updated_at, attached_at fields

    Returns:
        Unix timestamp in milliseconds

    Examples:
        >>> entity_last_active_ts({"last_seen": "20260115"})  # Jan 15 2026 local midnight
        >>> entity_last_active_ts({"updated_at": 1700000000000})
        1700000000000
        >>> entity_last_active_ts({})
        1767225600000  # DEFAULT_ACTIVITY_TS (Jan 1 2026 UTC)
    """
    # Priority 1: last_seen (YYYYMMDD string)
    last_seen = entity.get("last_seen")
    if last_seen and isinstance(last_seen, str) and len(last_seen) == 8:
        try:
            dt = datetime.strptime(last_seen, "%Y%m%d")
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass  # Malformed, fall through

    # Priority 2: updated_at
    updated_at = entity.get("updated_at")
    if updated_at and isinstance(updated_at, int) and updated_at > 0:
        return updated_at

    # Priority 3: attached_at
    attached_at = entity.get("attached_at")
    if attached_at and isinstance(attached_at, int) and attached_at > 0:
        return attached_at

    # Priority 4: Default
    return DEFAULT_ACTIVITY_TS


def is_valid_entity_type(etype: str) -> bool:
    """Validate entity type: alphanumeric and spaces only, at least 3 characters."""
    if not etype or len(etype.strip()) < 3:
        return False
    # Must contain only alphanumeric and spaces, and at least one alphanumeric character
    return bool(
        re.match(r"^[A-Za-z0-9 ]+$", etype) and re.search(r"[A-Za-z0-9]", etype)
    )


def entity_slug(name: str) -> str:
    """Generate a stable slug identifier for an entity name.

    The slug is used as:
    - The `id` field stored in entity records
    - Folder names for entity memory storage
    - URL-safe programmatic references

    Uses python-slugify to convert names to lowercase with underscores.
    Long names are truncated with a hash suffix to ensure uniqueness.

    Args:
        name: Entity name (e.g., "Alice Johnson", "Acme Corp")

    Returns:
        Slug identifier (e.g., "alice_johnson", "acme_corp")

    Examples:
        >>> entity_slug("Alice Johnson")
        'alice_johnson'
        >>> entity_slug("O'Brien")
        'o_brien'
        >>> entity_slug("AT&T")
        'at_t'
        >>> entity_slug("José García")
        'jose_garcia'
    """
    if not name or not name.strip():
        return ""

    # Use slugify with underscore separator
    slug = slugify(name, separator="_")

    # Handle very long names - truncate and add hash suffix
    if len(slug) > MAX_ENTITY_SLUG_LENGTH:
        # Create hash of full name for uniqueness
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        # Truncate and append hash
        slug = slug[: MAX_ENTITY_SLUG_LENGTH - 9] + "_" + name_hash

    return slug


def atomic_write(path: Path, content: str, prefix: str = ".tmp_") -> None:
    """Write content to a file atomically using tempfile + rename.

    Creates a temporary file in the same directory, writes content,
    then atomically renames to the target path. This ensures the
    target file is never in a partial state.

    Args:
        path: Target file path
        content: String content to write
        prefix: Prefix for the temporary file (default: ".tmp_")

    Raises:
        OSError: If write or rename fails
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise
