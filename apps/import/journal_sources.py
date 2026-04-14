# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from functools import wraps
from pathlib import Path

from flask import abort, g, request

from apps.utils import get_app_storage_path
from convey import state

logger = logging.getLogger(__name__)

KEY_BYTES = 32
STATE_AREAS = ("segments", "entities", "facets", "imports", "config")


def is_valid_journal_source_name(name: str) -> bool:
    return (
        bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name
    )


def generate_key() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode().rstrip("=")


def get_journal_sources_dir() -> Path:
    return get_app_storage_path("import", "journal_sources", ensure_exists=True)


def load_journal_source(key: str) -> dict | None:
    sources_dir = get_journal_sources_dir()
    for source_path in sources_dir.glob("*.json"):
        try:
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("key") == key:
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load journal source %s: %s", source_path, e)
    return None


def save_journal_source(data: dict) -> bool:
    name = data.get("name")
    if not is_valid_journal_source_name(name):
        return False
    source_path = get_journal_sources_dir() / f"{name}.json"
    try:
        with open(source_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(source_path, 0o600)
        return True
    except OSError:
        return False


def list_journal_sources() -> list[dict]:
    sources_dir = get_journal_sources_dir()
    sources = []
    for source_path in sources_dir.glob("*.json"):
        try:
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)
            sources.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    sources.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return sources


def find_journal_source_by_name(name: str) -> dict | None:
    if not is_valid_journal_source_name(name):
        return None
    source_path = get_journal_sources_dir() / f"{name}.json"
    if not source_path.exists():
        return None
    try:
        with open(source_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def create_state_directory(journal_root: Path, key_prefix: str) -> Path:
    state_dir = journal_root / "imports" / key_prefix
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "source.json").write_text("{}", encoding="utf-8")
    for area in STATE_AREAS:
        area_dir = state_dir / area
        area_dir.mkdir(parents=True, exist_ok=True)
        (area_dir / "state.json").write_text("{}", encoding="utf-8")
    return state_dir


def get_state_directory(key_prefix: str) -> Path:
    return Path(state.journal_root) / "imports" / key_prefix


def require_journal_source(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = None
        if auth.startswith("Bearer "):
            bearer = auth[7:].strip()
            if bearer:
                token = bearer

        if not token:
            abort(401, description="Missing or invalid authentication")

        source = load_journal_source(token)
        if not source:
            abort(401, description="Invalid API key")
        if source.get("revoked"):
            abort(403, description="API key has been revoked")

        g.journal_source = source
        return f(*args, **kwargs)

    return wrapped
