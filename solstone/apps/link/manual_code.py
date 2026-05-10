# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Manual pairing-code generation and normalization helpers."""

from __future__ import annotations

import secrets

from solstone.apps.link.copy import (
    MANUAL_CODE_ALPHABET,
    MANUAL_CODE_GROUP,
    MANUAL_CODE_LEN,
)


def generate() -> str:
    raw = "".join(secrets.choice(MANUAL_CODE_ALPHABET) for _ in range(MANUAL_CODE_LEN))
    return "-".join(
        raw[i : i + MANUAL_CODE_GROUP]
        for i in range(0, MANUAL_CODE_LEN, MANUAL_CODE_GROUP)
    )


def normalize(s: str) -> str:
    return s.upper().replace("-", "").replace(" ", "").strip()
