"""Utilities for writing `.crumbs` dependency files."""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List


class CrumbBuilder:
    """Builder for collecting metadata and writing `.crumbs` files."""

    def __init__(self, generator: str | None = None) -> None:
        self.generator = generator or sys.argv[0] or "unknown"
        self._deps: List[Dict[str, Any]] = []

    def add_file(self, path: str | Path) -> "CrumbBuilder":
        """Record a single file dependency."""
        path_str = str(path)
        mtime = int(os.path.getmtime(path_str))
        self._deps.append({"type": "file", "path": path_str, "mtime": mtime})
        return self

    def add_files(self, paths: Iterable[str | Path]) -> "CrumbBuilder":
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


class CrumbState(str, Enum):
    """Result of :func:`validate_crumb`."""

    MISSING = "missing"
    STALE = "stale"
    OK = "ok"


def validate_crumb(output: str | Path) -> CrumbState:
    """Return status for ``output`` based on its ``.crumb`` file."""

    output_path = Path(output)
    crumb_path = Path(str(output) + ".crumb")

    if not output_path.exists() or not crumb_path.exists():
        return CrumbState.MISSING

    try:
        with open(crumb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return CrumbState.STALE

    for dep in data.get("dependencies", []):
        dep_type = dep.get("type")
        if dep_type == "file":
            path = dep.get("path")
            mtime = dep.get("mtime")
            if (
                not path
                or not os.path.exists(path)
                or int(os.path.getmtime(path)) != mtime
            ):
                return CrumbState.STALE
        elif dep_type == "glob":
            pattern = dep.get("pattern", "")
            recorded = dep.get("files", {})
            matches = glob.glob(pattern)
            if set(matches) != set(recorded):
                return CrumbState.STALE
            for m in matches:
                if int(os.path.getmtime(m)) != recorded[m]:
                    return CrumbState.STALE
        elif dep_type == "model":
            continue
        else:
            return CrumbState.STALE

    return CrumbState.OK
