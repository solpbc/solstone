# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pairing key validation and session-key helpers."""

from __future__ import annotations

import hashlib
import secrets

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MAX_PUBLIC_KEY_LENGTH = 2048


def validate_public_key(public_key: str) -> str:
    candidate = str(public_key or "").strip()
    if not candidate:
        raise ValueError("public key is required")
    if len(candidate) > MAX_PUBLIC_KEY_LENGTH:
        raise ValueError("public key is too long")
    try:
        parsed = serialization.load_ssh_public_key(candidate.encode("utf-8"))
    except Exception as exc:
        raise ValueError("public key is invalid") from exc
    if not isinstance(parsed, Ed25519PublicKey):
        raise ValueError("public key must be ssh-ed25519")
    return parsed.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("utf-8")


def generate_session_key() -> str:
    return f"dsk_{secrets.token_urlsafe(32)}"


def hash_session_key(session_key: str) -> str:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def mask_session_key(session_key: str) -> str:
    value = str(session_key or "")
    return f"...{value[-4:]} (len={len(value)})"


__all__ = [
    "MAX_PUBLIC_KEY_LENGTH",
    "generate_session_key",
    "hash_session_key",
    "mask_session_key",
    "validate_public_key",
]
