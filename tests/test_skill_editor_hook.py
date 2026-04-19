# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.skills.talent.skill_editor import (
    NO_PENDING_SKILL_WORK,
    WATCHING_AND_LEARNING,
    post_process,
    pre_process,
)
from think import skills as think_skills


@pytest.fixture
def skill_editor_env(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    return Path(tmp_path)


def _pattern(
    *,
    slug: str = "alpha-skill",
    name: str = "Alpha Skill",
    status: str = "emerging",
    day: str = "2026-04-19",
    facet: str = "work",
    activity_ids: list[str] | None = None,
    notes: str = "Observed recurring work",
    needs_profile: bool = False,
    needs_refresh: bool = False,
    profile_generated_at: str | None = None,
    created_at: str = "2026-04-19T09:00:00Z",
    updated_at: str = "2026-04-19T10:00:00Z",
    observations: list[dict] | None = None,
) -> dict:
    if observations is None:
        ids = ["act-1"] if activity_ids is None else activity_ids
        observations = [
            {
                "day": day,
                "facet": facet,
                "activity_ids": ids,
                "notes": notes,
                "recorded_at": created_at,
            }
        ]
    facets = sorted({obs["facet"] for obs in observations if obs.get("facet")})
    days = sorted(obs["day"] for obs in observations if obs.get("day"))
    return {
        "slug": slug,
        "name": name,
        "status": status,
        "observations": observations,
        "facets_touched": facets,
        "first_seen": days[0] if days else "",
        "last_seen": days[-1] if days else "",
        "needs_profile": needs_profile,
        "needs_refresh": needs_refresh,
        "profile_generated_at": profile_generated_at,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _request(
    *,
    request_id: str = "req-1",
    slug: str = "alpha-skill",
    instructions: str = "Refine the description",
    requested_at: str = "2026-04-19T08:00:00Z",
    requested_by: str = "chat",
    processed_at: str | None = None,
) -> dict:
    return {
        "id": request_id,
        "slug": slug,
        "instructions": instructions,
        "requested_at": requested_at,
        "requested_by": requested_by,
        "processed_at": processed_at,
    }


def _activity_record(
    *,
    activity_id: str = "act-1",
    description: str = "Investigated a performance regression",
    segments: list[str] | None = None,
    facet: str = "work",
    day: str = "2026-04-19",
) -> dict:
    return {
        "id": activity_id,
        "activity": "coding",
        "title": "Performance investigation",
        "description": description,
        "segments": ["090000_300"] if segments is None else segments,
        "active_entities": ["indexer", "cprofile"],
        "created_at": "2026-04-19T09:05:00Z",
        "facet": facet,
        "day": day,
    }


def _profile_markdown(
    *,
    slug: str = "alpha-skill",
    display_name: str = "Alpha Skill",
    description: str = "A grounded description.",
    category: str = "engineering",
    confidence: float = 0.6,
    body: str = "## Description\n\nProfile body.\n",
    aliases: list[str] | None = None,
    updated_at: str | None = None,
) -> str:
    lines = [
        "---",
        f'name: "{slug}"',
        f'display_name: "{display_name}"',
        f'description: "{description}"',
        f'category: "{category}"',
        f"confidence: {confidence}",
    ]
    if aliases is not None:
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f'  - "{alias}"')
    if updated_at is not None:
        lines.append(f'updated_at: "{updated_at}"')
    lines.extend(["---", "", body.rstrip(), ""])
    return "\n".join(lines)


def _seed_fixture(
    root: Path,
    *,
    patterns: list[dict] | None = None,
    edit_requests: list[dict] | None = None,
    profiles: dict[str, str] | None = None,
    activities: dict[tuple[str, str], list[dict]] | None = None,
    narratives: dict[tuple[str, str, str], str] | None = None,
    spans: dict[tuple[str, str], list[dict]] | None = None,
    agency_observations: str = WATCHING_AND_LEARNING,
) -> None:
    skills_dir = root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if patterns is not None:
        content = "\n".join(json.dumps(row) for row in patterns)
        if content:
            content += "\n"
        (skills_dir / "patterns.jsonl").write_text(content, encoding="utf-8")

    if edit_requests is not None:
        content = "\n".join(json.dumps(row) for row in edit_requests)
        if content:
            content += "\n"
        (skills_dir / "edit_requests.jsonl").write_text(content, encoding="utf-8")

    for slug, markdown in (profiles or {}).items():
        (skills_dir / f"{slug}.md").write_text(markdown, encoding="utf-8")

    identity_dir = root / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / "agency.md").write_text(
        "# agency\n\n## observations\n"
        + agency_observations
        + "\n\n## next\n\n[nothing yet]\n",
        encoding="utf-8",
    )

    for (facet, day), records in (activities or {}).items():
        activities_dir = root / "facets" / facet / "activities"
        activities_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(json.dumps(record) for record in records)
        if content:
            content += "\n"
        (activities_dir / f"{day}.jsonl").write_text(content, encoding="utf-8")

    for (facet, day, activity_id), content in (narratives or {}).items():
        narrative_dir = root / "facets" / facet / "activities" / day / activity_id
        narrative_dir.mkdir(parents=True, exist_ok=True)
        (narrative_dir / "narrative.md").write_text(content, encoding="utf-8")

    for (facet, day), rows in (spans or {}).items():
        spans_dir = root / "facets" / facet / "spans"
        spans_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(json.dumps(row) for row in rows)
        if content:
            content += "\n"
        (spans_dir / f"{day}.jsonl").write_text(content, encoding="utf-8")


def test_pre_picks_oldest_edit_request(skill_editor_env):
    pattern = _pattern(needs_profile=True)
    older = _request(request_id="req-old", requested_at="2026-04-19T07:00:00Z")
    newer = _request(
        request_id="req-new",
        requested_at="2026-04-19T09:00:00Z",
        instructions="Newer request",
    )
    _seed_fixture(
        skill_editor_env,
        patterns=[pattern],
        edit_requests=[newer, older],
        profiles={"alpha-skill": _profile_markdown()},
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert result["meta"]["mode"] == "edit_request"
    assert result["meta"]["request_id"] == "req-old"
    assert result["template_vars"]["owner_instructions"] == "Refine the description"


def test_pre_picks_needs_profile_when_no_edit_request(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[
            _pattern(
                slug="first", needs_profile=True, updated_at="2026-04-19T07:00:00Z"
            ),
            _pattern(
                slug="second",
                needs_profile=True,
                updated_at="2026-04-19T08:00:00Z",
            ),
        ],
        activities={
            ("work", "2026-04-19"): [_activity_record(activity_id="act-1")],
        },
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert result["meta"]["mode"] == "create"
    assert result["meta"]["slug"] == "first"
    assert "## Observation ledger" in result["template_vars"]["skill_context"]


def test_pre_picks_needs_refresh_when_no_new_or_edit(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[_pattern(status="mature", needs_refresh=True)],
        profiles={"alpha-skill": _profile_markdown()},
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert result["meta"]["mode"] == "refresh"
    assert result["template_vars"]["existing_profile"].startswith("---")


def test_pre_returns_skip_when_nothing_pending(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(status="mature")])

    result = pre_process({"day": "2026-04-19"})

    assert result == {"skip_reason": NO_PENDING_SKILL_WORK}


def test_pre_skips_zero_observation_pattern(skill_editor_env):
    empty = _pattern(slug="empty", needs_profile=True, observations=[])
    ready = _pattern(
        slug="ready", needs_profile=True, updated_at="2026-04-19T11:00:00Z"
    )
    _seed_fixture(skill_editor_env, patterns=[empty, ready])

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert result["meta"]["slug"] == "ready"


def test_pre_handles_missing_slug_defensively(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        edit_requests=[_request(slug="missing-skill")],
    )

    result = pre_process({"day": "2026-04-19"})

    assert result == {"skip_reason": "edit-request target slug missing"}
    rows = think_skills.load_edit_requests()
    assert rows[0]["processed_at"] is not None
    assert rows[0]["processing_error"] == "slug missing"


def test_pre_refresh_missing_profile_falls_back_to_create(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[_pattern(status="mature", needs_refresh=True)],
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert result["meta"]["mode"] == "create"
    assert result["template_vars"]["existing_profile"] == ""


def test_pre_includes_span_bodies_from_spans_jsonl(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[_pattern(needs_profile=True)],
        spans={
            ("work", "2026-04-19"): [
                {"span_id": "act-1", "talent": "story", "body": "Span body evidence"}
            ]
        },
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert "story: Span body evidence" in result["template_vars"]["skill_context"]


def test_pre_emits_spans_unavailable_when_file_missing(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[_pattern(needs_profile=True)],
    )

    result = pre_process({"day": "2026-04-19"})

    assert result is not None
    assert "[spans unavailable]" in result["template_vars"]["skill_context"]


def test_post_creates_new_profile_and_fires_nudge(skill_editor_env):
    pattern = _pattern(
        needs_profile=True,
        observations=[
            _pattern()["observations"][0],
            _pattern(day="2026-04-20")["observations"][0],
        ],
    )
    _seed_fixture(skill_editor_env, patterns=[pattern])
    result = _profile_markdown(
        slug="alpha-skill",
        display_name="Alpha Skill",
        description="A grounded recurring capability.",
        confidence=0.8,
        body="## Description\n\nNew profile.\n",
    )

    post_process(
        result, {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}}
    )

    saved = think_skills.load_profile("alpha-skill")
    assert saved is not None and "display_name: Alpha Skill" in saved
    updated = think_skills.find_pattern("alpha-skill")
    assert updated["status"] == "mature"
    assert updated["needs_profile"] is False
    agency = (skill_editor_env / "identity" / "agency.md").read_text(encoding="utf-8")
    assert "- Noticed recurring skill: Alpha Skill — observed 2 times" in agency


def test_post_refreshes_existing_profile_no_nudge(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[
            _pattern(
                status="mature",
                needs_refresh=True,
                profile_generated_at="2026-04-18T09:00:00Z",
            )
        ],
        profiles={"alpha-skill": _profile_markdown(display_name="Old Name")},
        agency_observations="existing observation",
    )
    result = _profile_markdown(
        slug="alpha-skill",
        display_name="Refreshed Skill",
        description="Updated grounded profile.",
        confidence=0.7,
    )

    post_process(
        result, {"meta": {"slug": "alpha-skill", "mode": "refresh", "request_id": None}}
    )

    saved = think_skills.load_profile("alpha-skill")
    assert saved is not None and "Refreshed Skill" in saved
    updated = think_skills.find_pattern("alpha-skill")
    assert updated["needs_refresh"] is False
    agency = (skill_editor_env / "identity" / "agency.md").read_text(encoding="utf-8")
    assert "Refreshed Skill" not in agency


def test_post_processes_edit_request_clears_both_flags(skill_editor_env):
    _seed_fixture(
        skill_editor_env,
        patterns=[_pattern(needs_profile=True, needs_refresh=True)],
        edit_requests=[_request()],
        profiles={"alpha-skill": _profile_markdown()},
    )
    result = _profile_markdown(
        slug="alpha-skill",
        display_name="Edited Skill",
        description="Edited grounded profile.",
        confidence=0.75,
    )

    post_process(
        result,
        {
            "meta": {
                "slug": "alpha-skill",
                "mode": "edit_request",
                "request_id": "req-1",
            }
        },
    )

    updated = think_skills.find_pattern("alpha-skill")
    assert updated["needs_profile"] is False
    assert updated["needs_refresh"] is False
    requests = think_skills.load_edit_requests()
    assert requests[0]["processed_at"] is not None


def test_post_rejects_slug_mismatch(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])

    result = post_process(
        _profile_markdown(slug="wrong-slug"),
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    assert result is None
    assert think_skills.load_profile("alpha-skill") is None


def test_post_rejects_missing_required_field(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])
    invalid = """---
name: "alpha-skill"
display_name: "Alpha Skill"
description: "desc"
confidence: 0.5
---

## Description

Missing category.
"""

    result = post_process(
        invalid,
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    assert result is None
    assert think_skills.find_pattern("alpha-skill")["needs_profile"] is True


def test_post_rejects_description_too_long(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])

    result = post_process(
        _profile_markdown(description="x" * 1025),
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    assert result is None


def test_post_rejects_description_empty(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])

    result = post_process(
        _profile_markdown(description=""),
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    assert result is None


def test_post_rejects_non_numeric_confidence(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])
    invalid = """---
name: "alpha-skill"
display_name: "Alpha Skill"
description: "Grounded description"
category: "engineering"
confidence: "high"
---

## Description

Invalid confidence.
"""

    result = post_process(
        invalid,
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    assert result is None


def test_post_is_idempotent_on_double_call(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])
    result = _profile_markdown(
        description="A description that is grounded.",
        display_name="Stable Skill",
        confidence=0.9,
    )
    context = {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}}

    post_process(result, context)
    first_pattern = think_skills.find_pattern("alpha-skill").copy()
    first_agency = (skill_editor_env / "identity" / "agency.md").read_text(
        encoding="utf-8"
    )

    post_process(result, context)

    second_pattern = think_skills.find_pattern("alpha-skill").copy()
    second_agency = (skill_editor_env / "identity" / "agency.md").read_text(
        encoding="utf-8"
    )
    assert first_pattern == second_pattern
    assert first_agency == second_agency


def test_post_preserves_flags_when_output_invalid(skill_editor_env):
    _seed_fixture(skill_editor_env, patterns=[_pattern(needs_profile=True)])

    post_process(
        "not markdown",
        {"meta": {"slug": "alpha-skill", "mode": "create", "request_id": None}},
    )

    updated = think_skills.find_pattern("alpha-skill")
    assert updated["needs_profile"] is True
    assert think_skills.load_profile("alpha-skill") is None
