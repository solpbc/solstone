# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Centralized credential, authorization header, and token redactor."""

import re
from typing import Optional

# Regex patterns for sensitive tokens and credentials
RE_AWS_AUTH_HEADER = re.compile(r"AWS4-HMAC-SHA256\s+Credential=[^,\s]+(?:\s*,\s*SignedHeaders=[^,\s]+)?(?:\s*,\s*Signature=[a-f0-9]+)?", re.IGNORECASE)
RE_AWS_SESSION_TOKEN = re.compile(r"(?:x-amz-security-token|session_token)\s*[:=]\s*['\"]?([A-Za-z0-9+/=]{20,})['\"]?", re.IGNORECASE)
RE_AWS_ACCESS_KEY = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AROA)[A-Z0-9]{16}(?![A-Z0-9])")
RE_MINISIGN_SEC = re.compile(r"untrusted comment: minisign secret key[^\n]*\n[A-Za-z0-9+/=]+(?:\n[^\n]*\n[A-Za-z0-9+/=]+)?", re.IGNORECASE)
RE_MINISIGN_SEC_KEY = re.compile(r"(?<![A-Za-z0-9+/=])RWR[A-Za-z0-9+/=]{60,}(?![A-Za-z0-9+/=])")
RE_SECRET_KEY = re.compile(r"(\b(?:secret_key|secret|password|passphrase|token)\s*[:=]\s*['\"]?)(?!\[REDACTED)[^'\"\s,\n]+(['\"]?)", re.IGNORECASE)


def redact_sensitive_text(text: Optional[str]) -> str:
    """Redact secrets, authorization headers, access keys, tokens, and passphrases from text."""
    if not text:
        return ""

    s = text
    # Mask Authorization header
    s = RE_AWS_AUTH_HEADER.sub("AWS4-HMAC-SHA256 Credential=[REDACTED], SignedHeaders=[REDACTED], Signature=[REDACTED]", s)
    # Mask AWS session tokens first before generic token
    s = RE_AWS_SESSION_TOKEN.sub("x-amz-security-token: [REDACTED_SESSION_TOKEN]", s)
    # Mask Minisign secret block
    s = RE_MINISIGN_SEC.sub("[REDACTED_MINISIGN_SECRET_KEY]", s)
    # Mask raw Minisign secret key base64
    s = RE_MINISIGN_SEC_KEY.sub("[REDACTED_MINISIGN_SECRET_KEY]", s)
    # Mask AWS Access Key IDs
    s = RE_AWS_ACCESS_KEY.sub("[REDACTED_AWS_ACCESS_KEY]", s)
    # Mask named secrets/passwords/tokens
    s = RE_SECRET_KEY.sub(r"\1[REDACTED_SECRET]\2", s)

    return s
