# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Owner-facing copy and locked constants for the link pairing flow."""

from __future__ import annotations

from solstone.think.link.nonces import NONCE_TTL_SECONDS

PAIR_LINK_HOST = "link.solpbc.org"
PAIR_LINK_PATH = "/p"
MANUAL_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
MANUAL_CODE_LEN = 8
MANUAL_CODE_GROUP = 4
PAIR_CODE_TTL_SECONDS = NONCE_TTL_SECONDS
MANUAL_CODE_LABEL = "manual code"
