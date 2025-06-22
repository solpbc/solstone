"""Utilities for writing `.crumbs` dependency files."""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


class CrumbBuilder:
    """Builder for collecting metadata and writing `.crumbs` files."""

    def __init__(self, generator: str | None = None) -> None:
        self.generator = generator or sys.argv[0] or "unknown"
        self._deps: List[Dict[str, Any]] = []

    def add_file(self, path: str) -> "CrumbBuilder":
        """Record a single file dependency."""
        mtime = int(os.path.getmtime(path))
        self._deps.append({"type": "file", "path": path, "mtime": mtime})
        return self

    def add_files(self, paths: Iterable[str]) -> "CrumbBuilder":
        for p in paths:
            self.add_file(p)
        return self

    def add_glob(self, pattern: str) -> "CrumbBuilder":
        """Record a glob pattern and the files matched."""
        matches = glob.glob(pattern)
        files = {m: int(os.path.getmtime(m)) for m in matches}
        self._deps.append({"type": "glob", "pattern": pattern, "files": files})
        return self

    def add_model(self, name: str) -> "CrumbBuilder":
        self._deps.append({"type": "model", "name": name})
        return self

    def commit(self, output: str) -> str:
        crumb_path = output + ".crumb"

        crumb = {
            "generator": self.generator,
            "output": output,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dependencies": self._deps,
        }

        directory = os.path.dirname(crumb_path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(crumb_path, "w", encoding="utf-8") as f:
            json.dump(crumb, f, indent=2)
        return crumb_path
