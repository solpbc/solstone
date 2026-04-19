# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivitySourceRef:
    facet: str
    day: str
    activity_id: str
    field: str
    created_at: int


@dataclass(frozen=True)
class LedgerItem:
    id: str
    state: str
    owner: str
    owner_entity_id: str | None
    counterparty: str | None
    counterparty_entity_id: str | None
    action: str
    summary: str  # summary == action verbatim; CLI composes "owner: action → counterparty" at render time if it wants.
    when: str | None
    context: str
    opened_at: int
    closed_at: int | None
    age_days: int
    sources: tuple[ActivitySourceRef, ...]


@dataclass(frozen=True)
class Decision:
    id: str
    owner: str
    owner_entity_id: str | None
    action: str
    context: str
    day: str
    created_at: int
    source: ActivitySourceRef
