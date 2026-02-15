# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Syncable importer backend framework."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SyncableBackend(Protocol):
    """Protocol for importer backends that support syncing."""

    name: str

    def sync(self, journal_root: Path, *, dry_run: bool = True) -> dict[str, Any]: ...


SYNCABLE_REGISTRY: dict[str, str] = {
    "plaud": "think.importers.plaud",
}


def load_sync_state(journal_root: Path, backend: str) -> dict[str, Any] | None:
    """Load sync state for a backend."""
    state_path = journal_root / "imports" / f"{backend}.json"
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load sync state for %s: %s", backend, exc)
        return None


def save_sync_state(journal_root: Path, backend: str, state: dict[str, Any]) -> None:
    """Save sync state for a backend with an atomic write."""
    imports_dir = journal_root / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    state_path = imports_dir / f"{backend}.json"

    fd, tmp_path = tempfile.mkstemp(
        dir=imports_dir, suffix=".tmp", prefix=f".{backend}_"
    )
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(state_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def get_syncable_backends() -> list[SyncableBackend]:
    """Discover and instantiate all registered syncable backends."""
    import importlib

    backends: list[SyncableBackend] = []
    for name, module_path in SYNCABLE_REGISTRY.items():
        try:
            module = importlib.import_module(module_path)
            backend = getattr(module, "backend", None)
            if isinstance(backend, SyncableBackend):
                backends.append(backend)
            else:
                logger.warning(
                    "Backend %s from %s does not conform to SyncableBackend",
                    name,
                    module_path,
                )
        except Exception as exc:
            logger.warning("Failed to load syncable backend %s: %s", name, exc)
    return backends
