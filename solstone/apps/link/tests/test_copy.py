# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Regression tests for locked link pairing copy constants."""

from __future__ import annotations

from solstone.apps.link import copy
from solstone.think.link.nonces import NONCE_TTL_SECONDS


def test_copy_constants_are_locked() -> None:
    assert copy.PAIR_LINK_HOST == "link.solpbc.org"
    assert copy.PAIR_LINK_PATH == "/p"
    assert copy.MANUAL_CODE_ALPHABET == "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    assert len(copy.MANUAL_CODE_ALPHABET) == 32
    assert not {"0", "O", "1", "I"} & set(copy.MANUAL_CODE_ALPHABET)
    assert copy.MANUAL_CODE_LEN == 8
    assert copy.MANUAL_CODE_GROUP == 4
    assert copy.PAIR_CODE_TTL_SECONDS == NONCE_TTL_SECONDS
    assert copy.PAIR_CODE_TTL_SECONDS == 300
    assert copy.MANUAL_CODE_LABEL == "manual code"
