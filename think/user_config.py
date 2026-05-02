# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Read/write the per-user solstone config at ~/.config/solstone/config.toml.

Single TOML key today: ``journal``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import tomllib


def default_journal() -> str:
    return str(Path.home() / "Documents" / "journal")


def config_path() -> Path:
    return Path.home() / ".config" / "solstone" / "config.toml"


def read_user_config() -> dict[str, str]:
    try:
        data: dict[str, Any] = tomllib.loads(config_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {}

    return {k: v for k, v in data.items() if isinstance(v, str)}


def write_user_config(*, journal: str) -> Path:
    cfg = config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)

    escaped = journal.replace("\\", "\\\\").replace('"', '\\"')
    content = f'journal = "{escaped}"\n'

    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_config",
        suffix=".toml",
        dir=cfg.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, cfg)
    except Exception:
        os.unlink(tmp_path)
        raise

    return cfg
