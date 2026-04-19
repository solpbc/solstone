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


@dataclass(frozen=True)
class Cadence:
    recent_interactions_count_30d: int
    last_seen: str | None
    avg_interval_days: float | None
    gone_quiet_since: int | None


@dataclass(frozen=True)
class ProfileBrief:
    entity_id: str
    name: str
    type: str
    description: str | None
    last_seen: str | None
    open_loop_count: int
    decisions_count_30d: int


@dataclass(frozen=True)
class Profile:
    entity_id: str
    name: str
    type: str
    aka: tuple[str, ...]
    is_self: bool
    facets: tuple[str, ...]
    description: str | None
    cadence: Cadence
    open_with_them: tuple[LedgerItem, ...]
    closed_with_them_30d: tuple[LedgerItem, ...]
    decisions_involving_them: tuple[Decision, ...]
    sources: tuple[ActivitySourceRef, ...]
    generated_at: int
