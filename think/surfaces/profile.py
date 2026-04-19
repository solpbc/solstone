# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Profile consumer surface.

Cadence math uses distinct interaction days for ``avg_interval_days`` rather
than raw record count. ``gone_quiet_since`` uses a strict ``> avg * 2``
threshold, not ``>=``. When fewer than two distinct days exist, cadence still
returns ``last_seen`` when available but leaves ``avg_interval_days`` and
``gone_quiet_since`` unset. Known failure: if an entity is renamed mid-journal,
matching stays on the resolved ``entity_id`` while historical
``participation[].entity_id`` values keep the old slug, so cadence silently
splits across the rename boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterator

from think.activities import load_activity_records
from think.entities.journal import load_all_journal_entities
from think.entities.matching import find_matching_entity
from think.entities.relationships import (
    enrich_relationship_with_journal,
    load_facet_relationship,
)
from think.facets import get_facets
from think.surfaces import ledger
from think.surfaces.types import ActivitySourceRef, Cadence, Profile, ProfileBrief


@dataclass(frozen=True)
class _ResolvedTarget:
    entity_id: str
    name: str
    type: str
    aka: tuple[str, ...]
    is_principal: bool
    facets_with_desc: dict[str, str]

    def description_for(self, facets: tuple[str, ...] | None) -> str | None:
        selected_facets = (
            tuple(self.facets_with_desc.keys())
            if facets is None
            else tuple(facet for facet in facets if facet in self.facets_with_desc)
        )
        descriptions = [
            self.facets_with_desc[facet]
            for facet in selected_facets
            if self.facets_with_desc[facet]
        ]
        return " | ".join(descriptions) if descriptions else None


def _today_day() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _day_minus(days: int) -> str:
    today = datetime.strptime(_today_day(), "%Y%m%d").replace(tzinfo=UTC)
    return (today - timedelta(days=days)).strftime("%Y%m%d")


def _day_to_ordinal(day: str) -> int:
    return datetime.strptime(day, "%Y%m%d").toordinal()


def _iter_activity_days_window(window_days: int) -> Iterator[tuple[str, str]]:
    if window_days <= 0:
        return

    facets = tuple(get_facets().keys())
    today = datetime.strptime(_today_day(), "%Y%m%d").replace(tzinfo=UTC)
    for offset in range(window_days - 1, -1, -1):
        day = (today - timedelta(days=offset)).strftime("%Y%m%d")
        for facet in facets:
            yield facet, day


def _scan_attendee_entity_ids(window_days: int, roles: frozenset[str]) -> set[str]:
    entity_ids: set[str] = set()
    for facet, day in _iter_activity_days_window(window_days):
        for record in load_activity_records(facet, day):
            participation = record.get("participation")
            if not isinstance(participation, list):
                continue
            for entry in participation:
                if not isinstance(entry, dict) or entry.get("role") not in roles:
                    continue
                entity_id = entry.get("entity_id")
                if isinstance(entity_id, str) and entity_id:
                    entity_ids.add(entity_id)
    return entity_ids


def _resolve_target(name: str) -> _ResolvedTarget | None:
    journal_entities = load_all_journal_entities()
    if not journal_entities:
        return None

    match = find_matching_entity(name, list(journal_entities.values()))
    if match is None:
        return None

    entity_id = str(match.get("id") or "").strip()
    if not entity_id:
        return None

    journal_entity = journal_entities.get(entity_id) or dict(match)
    aka = journal_entity.get("aka")
    facets_with_desc: dict[str, str] = {}
    for facet in get_facets().keys():
        relationship = load_facet_relationship(facet, entity_id)
        if relationship is None:
            continue
        enriched = enrich_relationship_with_journal(relationship, journal_entity)
        description = enriched.get("description")
        facets_with_desc[facet] = description if isinstance(description, str) else ""

    return _ResolvedTarget(
        entity_id=entity_id,
        name=str(journal_entity.get("name") or match.get("name") or entity_id),
        type=str(journal_entity.get("type") or match.get("type") or ""),
        aka=tuple(item for item in aka if isinstance(item, str))
        if isinstance(aka, list)
        else (),
        is_principal=bool(
            journal_entity.get("is_principal") or match.get("is_principal")
        ),
        facets_with_desc=facets_with_desc,
    )


def _source_sort_key(source: ActivitySourceRef) -> tuple[str, str, str]:
    return source.day, source.facet, source.activity_id


def _compute_cadence(
    entity_id: str, *, include_mentions: bool
) -> tuple[Cadence, tuple[ActivitySourceRef, ...]]:
    active_roles = (
        frozenset({"attendee", "mentioned"})
        if include_mentions
        else frozenset({"attendee"})
    )
    interaction_days: list[str] = []
    sources: list[ActivitySourceRef] = []

    for facet, day in _iter_activity_days_window(90):
        for record in load_activity_records(facet, day):
            record_id = str(record.get("id") or "").strip()
            if not record_id:
                continue
            participation = record.get("participation")
            if not isinstance(participation, list):
                continue

            matched = False
            for entry in participation:
                if not isinstance(entry, dict):
                    continue
                if entry.get("entity_id") != entity_id:
                    continue
                if entry.get("role") not in active_roles:
                    continue
                matched = True
                break

            if not matched:
                continue

            interaction_days.append(day)
            sources.append(
                ActivitySourceRef(
                    facet=facet,
                    day=day,
                    activity_id=record_id,
                    field="participation",
                    created_at=int(record.get("created_at", 0) or 0),
                )
            )

    if not interaction_days:
        return Cadence(0, None, None, None), ()

    distinct_days = sorted(set(interaction_days))
    last_seen = distinct_days[-1]
    avg_interval_days: float | None = None
    gone_quiet_since: int | None = None
    recent_since = _day_minus(30)
    recent_count = sum(1 for day in interaction_days if day >= recent_since)

    if len(distinct_days) >= 2:
        first_ordinal = _day_to_ordinal(distinct_days[0])
        last_ordinal = _day_to_ordinal(last_seen)
        avg_interval_days = (last_ordinal - first_ordinal) / (len(distinct_days) - 1)
        quiet_gap_days = _day_to_ordinal(_today_day()) - last_ordinal
        if quiet_gap_days > avg_interval_days * 2:
            gone_quiet_since = quiet_gap_days

    cadence = Cadence(recent_count, last_seen, avg_interval_days, gone_quiet_since)
    return cadence, tuple(sorted(sources, key=_source_sort_key))


def full(
    name: str, *, facets: list[str] | None = None, include_mentions: bool = False
) -> Profile | None:
    target = _resolve_target(name)
    if target is None:
        return None

    cadence, sources = _compute_cadence(
        target.entity_id, include_mentions=include_mentions
    )
    if facets is None:
        selected_facets = tuple(target.facets_with_desc.keys())
    else:
        selected_facets = tuple(
            facet for facet in facets if facet in target.facets_with_desc
        )
    closed_since = _day_minus(30)

    return Profile(
        entity_id=target.entity_id,
        name=target.name,
        type=target.type,
        aka=target.aka,
        is_self=target.is_principal,
        facets=selected_facets,
        description=target.description_for(None if facets is None else tuple(facets)),
        cadence=cadence,
        open_with_them=tuple(ledger.list(state="open", counterparty=target.entity_id)),
        closed_with_them_30d=tuple(
            ledger.list(
                state="closed",
                closed_since=closed_since,
                counterparty=target.entity_id,
            )
        ),
        decisions_involving_them=tuple(ledger.decisions(involving=target.entity_id)),
        sources=sources,
        generated_at=int(datetime.now(UTC).timestamp() * 1000),
    )


def brief(name: str) -> ProfileBrief | None:
    target = _resolve_target(name)
    if target is None:
        return None

    cadence, _ = _compute_cadence(target.entity_id, include_mentions=False)
    decision_since = _day_minus(30)
    return ProfileBrief(
        entity_id=target.entity_id,
        name=target.name,
        type=target.type,
        description=target.description_for(None),
        last_seen=cadence.last_seen,
        open_loop_count=len(ledger.list(state="open", counterparty=target.entity_id)),
        decisions_count_30d=len(
            ledger.decisions(involving=target.entity_id, since=decision_since)
        ),
    )


def cadence(name: str, *, include_mentions: bool = False) -> Cadence | None:
    target = _resolve_target(name)
    if target is None:
        return None
    return _compute_cadence(target.entity_id, include_mentions=include_mentions)[0]


def list_active(*, window_days: int = 30) -> list[str]:
    return sorted(_scan_attendee_entity_ids(window_days, roles=frozenset({"attendee"})))
