# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity observations management.

Observations are durable factoids about entities stored in:
    facets/<facet>/entities/<id>/observations.jsonl

They capture useful information like preferences, expertise, relationships,
and biographical facts that help with future interactions.
"""

import fcntl
import json
import random
import time
from pathlib import Path
from typing import Any

from think.entities.core import atomic_write
from think.entities.relationships import entity_memory_path
from think.utils import now_ms

# Global cache for entity observations: {(facet, entity_slug): list[dict]}
_OBSERVATION_CACHE: dict[tuple[str, str], list[dict[str, Any]]] | None = None
# Global cache for observation counts: {path: count}
_OBSERVATION_COUNT_CACHE: dict[Path, int] | None = None


def clear_observation_cache() -> None:
    """Clear the entity observation cache."""
    global _OBSERVATION_CACHE
    _OBSERVATION_CACHE = None


def clear_observation_count_cache() -> None:
    """Clear the entity observation count cache."""
    global _OBSERVATION_COUNT_CACHE
    _OBSERVATION_COUNT_CACHE = None


def observations_file_path(facet: str, name: str) -> Path:
    """Return path to observations file for an entity.

    Observations are stored in the entity's memory folder:
    facets/{facet}/entities/{entity_slug}/observations.jsonl

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name (will be slugified)

    Returns:
        Path to observations.jsonl file

    Raises:
        ValueError: If name slugifies to empty string
    """
    folder = entity_memory_path(facet, name)
    return folder / "observations.jsonl"


def load_observations(facet: str, name: str) -> list[dict[str, Any]]:
    """Load observations for an entity.

    Args:
        facet: Facet name
        name: Entity name

    Returns:
        List of observation dictionaries with content, observed_at, source_day keys.
        Returns empty list if file doesn't exist.

    Example:
        >>> load_observations("work", "Alice Johnson")
        [{"content": "Prefers async communication", "observed_at": 1736784000000, "source_day": "20250113"}]
    """
    global _OBSERVATION_CACHE
    from think.entities.core import entity_slug

    slug = entity_slug(name)
    if _OBSERVATION_CACHE is not None:
        cached = _OBSERVATION_CACHE.get((facet, slug))
        if cached is not None:
            return cached

    path = observations_file_path(facet, name)

    if not path.exists():
        return []

    observations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                observations.append(data)
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    # Update cache if initialized
    if _OBSERVATION_CACHE is not None:
        _OBSERVATION_CACHE[(facet, slug)] = observations

    return observations


def count_observations(facet: str, name: str) -> int:
    """Count observations for an entity."""
    global _OBSERVATION_COUNT_CACHE
    try:
        obs_file = entity_memory_path(facet, name) / "observations.jsonl"
    except ValueError:
        return 0

    if not obs_file.exists():
        return 0

    if _OBSERVATION_COUNT_CACHE is None:
        _OBSERVATION_COUNT_CACHE = {}

    cached = _OBSERVATION_COUNT_CACHE.get(obs_file)
    if cached is not None:
        return cached

    try:
        with open(obs_file, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
    except OSError:
        return 0

    _OBSERVATION_COUNT_CACHE[obs_file] = count
    return count


def save_observations(
    facet: str, name: str, observations: list[dict[str, Any]]
) -> None:
    """Save observations to entity's observations file using atomic write.

    Args:
        facet: Facet name
        name: Entity name
        observations: List of observation dictionaries
    """
    # Clear cache on modification
    clear_observation_cache()
    clear_observation_count_cache()

    path = observations_file_path(facet, name)

    # Format observations as JSONL
    content = "".join(
        json.dumps(obs, ensure_ascii=False) + "\n" for obs in observations
    )
    atomic_write(path, content, prefix=".observations_")


def add_observation(
    facet: str,
    name: str,
    content: str,
    source_day: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Add an observation to an entity with file locking.

    Acquires an exclusive file lock to serialize concurrent writes to the
    same entity's observations file.

    Args:
        facet: Facet name
        name: Entity name
        content: The observation text
        source_day: Optional day (YYYYMMDD) when observation was made
        max_retries: Maximum attempts on transient OS errors (default 3)

    Returns:
        Dictionary with updated observations list and count

    Raises:
        ValueError: If content is empty
        OSError: If all retries exhausted

    Example:
        >>> add_observation("work", "Alice", "Prefers morning meetings", "20250113")
        {"observations": [...], "count": 1}
    """
    content = content.strip()
    if not content:
        raise ValueError("Observation content cannot be empty")

    path = observations_file_path(facet, name)
    lock_path = path.parent / f"{path.name}.lock"

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    observations = load_observations(facet, name)

                    observation: dict[str, Any] = {
                        "content": content,
                        "observed_at": now_ms(),
                    }
                    if source_day:
                        observation["source_day"] = source_day

                    observations.append(observation)
                    save_observations(facet, name, observations)

                    return {
                        "observations": observations,
                        "count": len(observations),
                    }
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except ValueError:
            raise  # Logical errors — don't retry
        except OSError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))

    raise last_error  # type: ignore[misc]
