# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Ledger surface for commitments, closures, and decisions.

Dropped/deferred talent resolutions: any matched closure -> state="closed"
regardless of its `resolution` field. CLI `--as dropped` is the only path to
state="dropped".
"""

from __future__ import annotations

import builtins
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from think.activities import append_ledger_close_edit, load_activity_records
from think.entities.matching import find_matching_entity
from think.facets import get_enabled_facets, get_facets
from think.surfaces.types import ActivitySourceRef, Decision, LedgerItem
from think.utils import get_journal

ACTION_MATCH_THRESHOLD = 78
# 78 < entity-name default 90: action phrases are longer/more variable (tense, articles) and need looser matching without opening to semantically unrelated strings.
_DAY_RE = re.compile(r"^\d{8}$")
_FIELD_ORDER = {"commitments": 0, "closures": 1, "decisions": 2, "edits": 3}
_ONE_DAY_MS = 86_400_000


def _normalize_action(s: str) -> str:
    # Lowercase, trim, and collapse whitespace so dedup/fuzzy matching stay stable without stripping semantic tokens.
    return " ".join(str(s).strip().lower().split())


def _normalize_text(s: str | None) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _dedup_key(
    owner_entity_id: str | None,
    action_normalized: str,
    counterparty_entity_id: str | None,
) -> str:
    digest = hashlib.sha256(
        f"{owner_entity_id or ''}|{action_normalized}|{counterparty_entity_id or ''}".encode(
            "utf-8"
        )
    )
    return digest.hexdigest()[:16]


def _decision_key(owner_entity_id: str | None, action_normalized: str, day: str) -> str:
    digest = hashlib.sha256(
        f"{owner_entity_id or ''}|{action_normalized}|{day}".encode("utf-8")
    )
    return digest.hexdigest()[:16]


def _record_created_at(record: dict[str, Any]) -> int:
    return int(record.get("created_at", 0) or 0)


def _activity_days(facet: str) -> builtins.list[str]:
    activities_dir = Path(get_journal()) / "facets" / facet / "activities"
    if not activities_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in activities_dir.glob("*.jsonl")
        if path.is_file() and _DAY_RE.fullmatch(path.stem)
    )


def _scan_records(facets: Iterable[str]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for facet in facets:
        for day in _activity_days(facet):
            for record in load_activity_records(facet, day):
                yield facet, day, record


def _source_ref(
    *, facet: str, day: str, activity_id: str, field: str, created_at: int
) -> ActivitySourceRef:
    return ActivitySourceRef(
        facet=facet,
        day=day,
        activity_id=activity_id,
        field=field,
        created_at=created_at,
    )


def _source_sort_key(source: ActivitySourceRef) -> tuple[int, str, str, str, int]:
    return (
        source.created_at,
        source.facet,
        source.day,
        source.activity_id,
        _FIELD_ORDER.get(source.field, 99),
    )


def _chronological_key(
    created_at: int, facet: str, day: str, activity_id: str
) -> tuple[int, str, str, str]:
    return created_at, facet, day, activity_id


def _edit_timestamp_ms(edit: dict[str, Any], fallback: int) -> int:
    timestamp = edit.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return fallback
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    return int(parsed.timestamp() * 1000)


def _actions_match(commitment_action: str, candidate_action: str) -> bool:
    if not commitment_action or not candidate_action:
        return False
    # Reuse the entity matcher with throwaway action "entities" so ledger pairing stays on the same fuzzy matching surface as entity resolution.
    match = find_matching_entity(
        commitment_action,
        [{"name": candidate_action}],
        fuzzy_threshold=ACTION_MATCH_THRESHOLD,
    )
    return match is not None


def _entity_pair_matches(
    left_id: str | None, right_id: str | None, *, allow_both_missing: bool
) -> bool:
    if left_id and right_id:
        return left_id == right_id
    if left_id or right_id:
        return False
    return allow_both_missing


def _counterparty_matches(item: dict[str, Any], closure: dict[str, Any]) -> bool:
    item_id = item["counterparty_entity_id"]
    closure_id = closure["counterparty_entity_id"]
    if item_id and closure_id:
        return item_id == closure_id
    if item_id or closure_id:
        return False
    return item["counterparty_normalized"] == closure["counterparty_normalized"]


def _story_closure_matches(item: dict[str, Any], closure: dict[str, Any]) -> bool:
    if not _entity_pair_matches(
        item["owner_entity_id"], closure["owner_entity_id"], allow_both_missing=True
    ):
        return False
    if not _counterparty_matches(item, closure):
        return False
    return _actions_match(item["action_normalized"], closure["action_normalized"])


def _resolve_sort(state: str, sort: str | None) -> str:
    valid = {"age_days_desc", "opened_at_desc", "closed_at_desc"}
    if sort is not None and sort not in valid:
        raise ValueError(f"unknown sort: {sort}")
    if sort is not None:
        return sort
    if state in {"closed", "dropped"}:
        return "closed_at_desc"
    return "age_days_desc"


def _validate_state(state: str) -> str:
    if state not in {"open", "closed", "dropped", "all"}:
        raise ValueError(f"unknown state: {state}")
    return state


def _party_matches(query: str, name: str | None, entity_id: str | None) -> bool:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return True
    candidates = [name or "", entity_id or "", (entity_id or "").replace("_", " ")]
    return any(
        normalized_query in _normalize_text(candidate) for candidate in candidates
    )


def _parse_day_ms(day: str, *, field_name: str) -> int:
    try:
        parsed = datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{field_name} must match YYYYMMDD") from exc
    return int(parsed.timestamp() * 1000)


def _list_all_facets() -> builtins.list[str]:
    return builtins.list(get_facets().keys())


def _build_ledger_items(
    records: Iterable[tuple[str, str, dict[str, Any]]],
) -> builtins.list[LedgerItem]:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    commitments: dict[str, dict[str, Any]] = {}
    story_closures: builtins.list[dict[str, Any]] = []
    manual_closes: dict[str, builtins.list[dict[str, Any]]] = {}

    for facet, day, record in records:
        record_id = str(record.get("id") or "")
        if not record_id:
            continue
        created_at = _record_created_at(record)

        for raw_commitment in record.get("commitments", []):
            if not isinstance(raw_commitment, dict):
                continue
            owner = str(raw_commitment.get("owner") or "").strip()
            action = str(raw_commitment.get("action") or "").strip()
            # Skip malformed story data rather than minting unusable ledger items with blank actions.
            if not owner or not action:
                continue

            owner_entity_id = raw_commitment.get("owner_entity_id")
            if not isinstance(owner_entity_id, str):
                owner_entity_id = None
            counterparty = str(raw_commitment.get("counterparty") or "").strip() or None
            counterparty_entity_id = raw_commitment.get("counterparty_entity_id")
            if not isinstance(counterparty_entity_id, str):
                counterparty_entity_id = None
            action_normalized = _normalize_action(action)
            item_id = _dedup_key(
                owner_entity_id, action_normalized, counterparty_entity_id
            )
            source = _source_ref(
                facet=facet,
                day=day,
                activity_id=record_id,
                field="commitments",
                created_at=created_at,
            )
            opening_key = _chronological_key(created_at, facet, day, record_id)
            entry = commitments.get(item_id)
            if entry is None:
                commitments[item_id] = {
                    "id": item_id,
                    "owner": owner,
                    "owner_entity_id": owner_entity_id,
                    "counterparty": counterparty,
                    "counterparty_entity_id": counterparty_entity_id,
                    "counterparty_normalized": _normalize_text(counterparty),
                    "action": action,
                    "action_normalized": action_normalized,
                    "when": str(raw_commitment.get("when") or "").strip() or None,
                    "context": str(raw_commitment.get("context") or ""),
                    "opened_at": created_at,
                    "opening_key": opening_key,
                    "sources": [source],
                }
                continue

            entry["sources"].append(source)
            if opening_key < entry["opening_key"]:
                entry.update(
                    {
                        "owner": owner,
                        "owner_entity_id": owner_entity_id,
                        "counterparty": counterparty,
                        "counterparty_entity_id": counterparty_entity_id,
                        "counterparty_normalized": _normalize_text(counterparty),
                        "action": action,
                        "action_normalized": action_normalized,
                        "when": str(raw_commitment.get("when") or "").strip() or None,
                        "context": str(raw_commitment.get("context") or ""),
                        "opened_at": created_at,
                        "opening_key": opening_key,
                    }
                )

        for raw_closure in record.get("closures", []):
            if not isinstance(raw_closure, dict):
                continue
            action = str(raw_closure.get("action") or "").strip()
            # Skip malformed story data rather than pairing closures with an empty fuzzy-match target.
            if not action:
                continue
            owner_entity_id = raw_closure.get("owner_entity_id")
            if not isinstance(owner_entity_id, str):
                owner_entity_id = None
            counterparty_entity_id = raw_closure.get("counterparty_entity_id")
            if not isinstance(counterparty_entity_id, str):
                counterparty_entity_id = None
            story_closures.append(
                {
                    "owner_entity_id": owner_entity_id,
                    "counterparty_entity_id": counterparty_entity_id,
                    "counterparty_normalized": _normalize_text(
                        str(raw_closure.get("counterparty") or "").strip() or None
                    ),
                    "action_normalized": _normalize_action(action),
                    "closed_at": created_at,
                    "state": "closed",
                    "sort_key": _chronological_key(created_at, facet, day, record_id),
                    "source": _source_ref(
                        facet=facet,
                        day=day,
                        activity_id=record_id,
                        field="closures",
                        created_at=created_at,
                    ),
                }
            )

        for raw_edit in record.get("edits", []):
            if not isinstance(raw_edit, dict):
                continue
            if raw_edit.get("fields") != ["ledger_close"]:
                continue
            ledger_close = raw_edit.get("ledger_close")
            if not isinstance(ledger_close, dict):
                continue
            item_id = ledger_close.get("item_id")
            as_state = ledger_close.get("as_state")
            if not isinstance(item_id, str) or not item_id:
                continue
            if as_state not in {"closed", "dropped"}:
                continue
            closed_at = _edit_timestamp_ms(raw_edit, created_at)
            manual_closes.setdefault(item_id, []).append(
                {
                    "closed_at": closed_at,
                    "state": as_state,
                    "sort_key": _chronological_key(closed_at, facet, day, record_id),
                    "source": _source_ref(
                        facet=facet,
                        day=day,
                        activity_id=record_id,
                        field="edits",
                        created_at=created_at,
                    ),
                }
            )

    story_closures.sort(key=lambda candidate: candidate["sort_key"])
    consumed_story_closures: set[int] = set()
    items: builtins.list[LedgerItem] = []

    for entry in sorted(commitments.values(), key=lambda item: item["opening_key"]):
        matched_story_sources: builtins.list[dict[str, Any]] = []
        for index, candidate in enumerate(story_closures):
            if index in consumed_story_closures:
                continue
            if not _story_closure_matches(entry, candidate):
                continue
            matched_story_sources.append(candidate)
            consumed_story_closures.add(index)

        closure_sources = matched_story_sources + manual_closes.get(entry["id"], [])
        # State/closed_at follow the earliest close across story and manual sources; later manual edits remain visible in sources but do not rewrite the first close.
        closure_sources.sort(key=lambda candidate: candidate["sort_key"])
        first_close = closure_sources[0] if closure_sources else None
        state = first_close["state"] if first_close is not None else "open"
        closed_at = first_close["closed_at"] if first_close is not None else None
        sources = builtins.list(entry["sources"])
        sources.extend(candidate["source"] for candidate in closure_sources)
        sources.sort(key=_source_sort_key)

        items.append(
            LedgerItem(
                id=entry["id"],
                state=state,
                owner=entry["owner"],
                owner_entity_id=entry["owner_entity_id"],
                counterparty=entry["counterparty"],
                counterparty_entity_id=entry["counterparty_entity_id"],
                action=entry["action"],
                summary=entry["action"],
                when=entry["when"],
                context=entry["context"],
                opened_at=entry["opened_at"],
                closed_at=closed_at,
                age_days=(now_ms - entry["opened_at"]) // _ONE_DAY_MS,
                sources=tuple(sources),
            )
        )

    return items


def _sort_items(
    items: builtins.list[LedgerItem], sort: str
) -> builtins.list[LedgerItem]:
    if sort == "age_days_desc":
        return sorted(
            items,
            key=lambda item: (item.age_days, item.opened_at, item.id),
            reverse=True,
        )
    if sort == "opened_at_desc":
        return sorted(items, key=lambda item: (item.opened_at, item.id), reverse=True)
    if sort == "closed_at_desc":
        return sorted(
            items,
            key=lambda item: (
                item.closed_at is not None,
                item.closed_at or -1,
                item.id,
            ),
            reverse=True,
        )
    raise ValueError(f"unknown sort: {sort}")


def list(
    *,
    state: str = "open",
    owner: str | None = None,
    counterparty: str | None = None,
    age_days_gte: int | None = None,
    closed_since: str | None = None,
    top: int | None = None,
    sort: str | None = None,
    facets: Iterable[str] | None = None,
) -> builtins.list[LedgerItem]:
    state = _validate_state(state)
    resolved_sort = _resolve_sort(state, sort)
    if facets is None:
        facets = get_enabled_facets().keys()  # Mirror enabled-facets convention: muted facets are opted out of downstream surfaces.
    items = _build_ledger_items(_scan_records(facets))

    if state != "all":
        items = [item for item in items if item.state == state]
    if owner:
        items = [
            item
            for item in items
            if _party_matches(owner, item.owner, item.owner_entity_id)
        ]
    if counterparty:
        items = [
            item
            for item in items
            if _party_matches(
                counterparty, item.counterparty, item.counterparty_entity_id
            )
        ]
    if age_days_gte is not None:
        items = [item for item in items if item.age_days >= age_days_gte]
    if closed_since is not None:
        threshold_ms = _parse_day_ms(closed_since, field_name="closed_since")
        items = [
            item
            for item in items
            if item.closed_at is not None and item.closed_at >= threshold_ms
        ]

    items = _sort_items(items, resolved_sort)
    if top is not None:
        items = items[:top]
    return items


def get(item_id: str) -> LedgerItem | None:
    for item in _build_ledger_items(_scan_records(_list_all_facets())):
        if item.id == item_id:
            return item
    return None


def close(item_id: str, *, note: str, as_state: str = "closed") -> LedgerItem:
    if as_state not in {"closed", "dropped"}:
        raise ValueError("as_state must be 'closed' or 'dropped'")
    if not note.strip():
        raise ValueError("note must be non-empty")

    item = get(item_id)
    if item is None:
        raise KeyError(item_id)

    commitment_sources = sorted(
        (source for source in item.sources if source.field == "commitments"),
        key=_source_sort_key,
    )
    if not commitment_sources:
        raise KeyError(item_id)

    source = commitment_sources[0]
    updated = append_ledger_close_edit(
        source.facet,
        source.day,
        source.activity_id,
        item_id=item_id,
        note=note.strip(),
        as_state=as_state,
    )
    if updated is None:
        raise KeyError(item_id)

    refreshed = get(item_id)
    if refreshed is None:
        raise KeyError(item_id)
    return refreshed


def decisions(
    *,
    owner: str | None = None,
    since: str | None = None,
    involving: str | None = None,
    top: int | None = None,
    facets: Iterable[str] | None = None,
) -> builtins.list[Decision]:
    if facets is None:
        facets = get_enabled_facets().keys()  # Mirror enabled-facets convention: muted facets are opted out of downstream surfaces.
    if since is not None:
        _parse_day_ms(since, field_name="since")

    deduped: dict[str, Decision] = {}
    for facet, day, record in _scan_records(facets):
        record_id = str(record.get("id") or "")
        if not record_id:
            continue
        created_at = _record_created_at(record)
        source = _source_ref(
            facet=facet,
            day=day,
            activity_id=record_id,
            field="decisions",
            created_at=created_at,
        )
        for raw_decision in record.get("decisions", []):
            if not isinstance(raw_decision, dict):
                continue
            owner_name = str(raw_decision.get("owner") or "").strip()
            action = str(raw_decision.get("action") or "").strip()
            if not owner_name or not action:
                continue
            owner_entity_id = raw_decision.get("owner_entity_id")
            if not isinstance(owner_entity_id, str):
                owner_entity_id = None
            decision_id = _decision_key(owner_entity_id, _normalize_action(action), day)
            candidate = Decision(
                id=decision_id,
                owner=owner_name,
                owner_entity_id=owner_entity_id,
                action=action,
                context=str(raw_decision.get("context") or ""),
                day=day,
                created_at=created_at,
                source=source,
            )
            current = deduped.get(decision_id)
            if current is None or _chronological_key(
                candidate.created_at,
                candidate.source.facet,
                candidate.day,
                candidate.source.activity_id,
            ) < _chronological_key(
                current.created_at,
                current.source.facet,
                current.day,
                current.source.activity_id,
            ):
                deduped[decision_id] = candidate

    results = builtins.list(deduped.values())
    if owner:
        results = [
            decision
            for decision in results
            if _party_matches(owner, decision.owner, decision.owner_entity_id)
        ]
    if involving:
        results = [
            decision
            for decision in results
            if _party_matches(involving, decision.owner, decision.owner_entity_id)
        ]
    if since:
        results = [decision for decision in results if decision.day >= since]

    results.sort(
        key=lambda decision: (
            decision.created_at,
            decision.source.facet,
            decision.day,
            decision.source.activity_id,
        ),
        reverse=True,
    )
    if top is not None:
        results = results[:top]
    return results
