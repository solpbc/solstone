"""Utilities for writing `.crumbs` dependency files."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List


class CrumbBuilder:
    """Builder for collecting metadata and writing `.crumbs` files."""

    def __init__(self, generator: str, output: str) -> None:
        self.generator = generator
        self.output = output
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

    def commit(self, crumb_path: str | None = None) -> str:
        if crumb_path is None:
            base, _ = os.path.splitext(self.output)
            crumb_path = base + ".crumbs"

        crumb = {
            "generator": self.generator,
            "output": self.output,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "dependencies": self._deps,
        }

        os.makedirs(os.path.dirname(crumb_path), exist_ok=True)
        with open(crumb_path, "w", encoding="utf-8") as f:
            json.dump(crumb, f, indent=2)
        return crumb_path
