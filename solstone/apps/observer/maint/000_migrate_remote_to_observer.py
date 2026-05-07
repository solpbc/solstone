# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate remote observer data and config to observer naming."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from solstone.think.utils import get_journal, setup_cli

logger = logging.getLogger(__name__)


def _move_remote_storage(journal_path: Path) -> int:
    """Move legacy remote storage files into the observer app directory."""
    source_root = journal_path / "apps" / "remote" / "remotes"
    target_root = journal_path / "apps" / "observer" / "observers"

    if not source_root.exists():
        return 0

    moved = 0
    for source_path in sorted(source_root.rglob("*")):
        if source_path.is_dir():
            continue

        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            if source_path.read_bytes() == target_path.read_bytes():
                source_path.unlink()
            else:
                warning = (
                    f"Conflict: {target_path} already exists with different content; "
                    f"leaving legacy file in place at {source_path}"
                )
                logger.warning(warning)
                print(f"WARNING: {warning}")
            continue

        source_path.rename(target_path)
        moved += 1

    for directory in sorted(source_root.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass

    try:
        source_root.rmdir()
    except OSError:
        pass

    legacy_app_dir = journal_path / "apps" / "remote"
    try:
        legacy_app_dir.rmdir()
    except OSError:
        pass

    return moved


def _migrate_config(journal_path: Path) -> bool:
    """Move observe.remote config into observe.observer."""
    config_path = journal_path / "config" / "journal.json"
    if not config_path.exists():
        return False

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read config %s: %s", config_path, exc)
        return False

    observe_config = config.get("observe")
    if not isinstance(observe_config, dict):
        return False

    remote_config = observe_config.get("remote")
    if not isinstance(remote_config, dict):
        return False

    observer_config = observe_config.setdefault("observer", {})
    if not isinstance(observer_config, dict):
        observer_config = {}
        observe_config["observer"] = observer_config

    for key, value in remote_config.items():
        observer_config.setdefault(key, value)

    del observe_config["remote"]

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    setup_cli(parser)

    journal_path = Path(get_journal())

    print("Migrating remote observer data to observer naming...")
    moved_files = _move_remote_storage(journal_path)
    config_updated = _migrate_config(journal_path)

    print(f"  Storage files moved: {moved_files}")
    print(f"  Config updated:      {'yes' if config_updated else 'no'}")


if __name__ == "__main__":
    main()
