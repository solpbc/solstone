# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Request identity stamped before Convey dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class ConveyIdentity:
    mode: Literal["dl", "pl-direct", "pl-via-spl"]
    fingerprint: Optional[str]
    device_label: Optional[str]
    paired_at: Optional[str]
    session_id: Optional[str]
