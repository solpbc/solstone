# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "install_parakeet_model.py"
_PROBE = """
import json
import runpy
import sys
from pathlib import Path

ns = runpy.run_path(sys.argv[1])
action = sys.argv[2]

if action == "constants":
    print(
        json.dumps(
            {
                "repo_name": ns["MAC_FLUIDAUDIO_REPO_NAME"],
                "model_files": ns["MAC_MODEL_FILES"],
            }
        )
    )
elif action == "verify":
    print(json.dumps({"ok": ns["_verify_mac_cache"](Path(sys.argv[3]))}))
else:
    raise AssertionError(f"unknown action: {action}")
"""


def _probe(action: str, cache_dir: Path | None = None) -> dict[str, object]:
    argv = [sys.executable, "-c", _PROBE, str(SCRIPT), action]
    if cache_dir is not None:
        argv.append(str(cache_dir))
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _load_constants() -> tuple[str, tuple[str, ...]]:
    data = _probe("constants")
    return data["repo_name"], tuple(data["model_files"])


def _verify_mac_cache(cache_dir: Path) -> bool:
    data = _probe("verify", cache_dir)
    return bool(data["ok"])


def _write_model_files(base_dir: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        target = base_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ok")


def test_verify_returns_true_when_files_at_fluidaudio_sibling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_name, model_files = _load_constants()
        tmp_path = Path(tmp_dir)
        cache_dir = tmp_path / "models"
        repo_dir = tmp_path / repo_name
        _write_model_files(repo_dir, model_files)

        assert _verify_mac_cache(cache_dir) is True


def test_verify_returns_false_when_sibling_empty():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_name, _ = _load_constants()
        tmp_path = Path(tmp_dir)
        cache_dir = tmp_path / "models"
        cache_dir.mkdir()
        (tmp_path / repo_name).mkdir()

        assert _verify_mac_cache(cache_dir) is False


def test_verify_returns_false_when_files_at_literal_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, model_files = _load_constants()
        tmp_path = Path(tmp_dir)
        cache_dir = tmp_path / "models"
        _write_model_files(cache_dir, model_files)

        assert _verify_mac_cache(cache_dir) is False
