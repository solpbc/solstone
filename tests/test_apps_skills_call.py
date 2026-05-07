# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from solstone.apps.skills.call import app as skills_app
from solstone.think.call import call_app
from solstone.think.skills import (
    load_edit_requests,
    load_patterns,
    locked_modify_patterns,
    profile_path,
    save_patterns,
    save_profile,
)

runner = CliRunner()


@pytest.fixture
def skill_cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    return Path(tmp_path)


def _make_pattern(
    *,
    slug: str = "alpha-skill",
    name: str = "Alpha Skill",
    status: str = "emerging",
    day: str = "2026-04-19",
    facet: str = "work",
    activity_ids: list[str] | None = None,
    notes: str = "",
    needs_profile: bool = False,
    needs_refresh: bool = False,
    profile_generated_at: str | None = None,
    created_at: str = "2026-04-19T14:22:00Z",
    updated_at: str = "2026-04-19T14:22:00Z",
) -> dict:
    ids = ["act_abc"] if activity_ids is None else activity_ids
    observations = [
        {
            "day": day,
            "facet": facet,
            "activity_ids": ids,
            "notes": notes,
            "recorded_at": created_at,
        }
    ]
    return {
        "slug": slug,
        "name": name,
        "status": status,
        "observations": observations,
        "facets_touched": [facet],
        "first_seen": day,
        "last_seen": day,
        "needs_profile": needs_profile,
        "needs_refresh": needs_refresh,
        "profile_generated_at": profile_generated_at,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _seed_patterns(*rows: dict) -> None:
    save_patterns(list(rows))


def _invoke(*args: str):
    return runner.invoke(call_app, ["skills", *args])


def test_skills_app_has_ten_registered_commands(skill_cli_env):
    command_names = {command.name for command in skills_app.registered_commands}

    assert len(skills_app.registered_commands) == 10
    assert command_names == {
        "list",
        "show",
        "observe",
        "seed",
        "promote",
        "refresh",
        "mark-dormant",
        "retire",
        "edit-request",
        "rename",
    }


def test_list_empty_text_output(skill_cli_env):
    result = _invoke("list")

    assert result.exit_code == 0
    assert result.output == ""


def test_list_empty_json_output(skill_cli_env):
    result = _invoke("list", "--json")

    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_list_filters_by_status(skill_cli_env):
    _seed_patterns(
        _make_pattern(slug="alpha-skill"),
        _make_pattern(slug="beta-skill", status="dormant"),
    )

    result = _invoke("list", "--status", "dormant")

    assert result.exit_code == 0
    assert "beta-skill" in result.output
    assert "alpha-skill" not in result.output


def test_list_filters_by_status_comma_separated(skill_cli_env):
    _seed_patterns(
        _make_pattern(slug="alpha-skill", status="mature"),
        _make_pattern(slug="beta-skill", status="dormant"),
        _make_pattern(slug="gamma-skill", status="retired"),
    )

    result = _invoke("list", "--status", "mature,dormant")

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "beta-skill" in result.output
    assert "gamma-skill" not in result.output


def test_show_missing_slug(skill_cli_env):
    result = _invoke("show", "missing-skill")

    assert result.exit_code == 1
    assert "no such skill" in result.stderr


def test_show_renders_pattern(skill_cli_env):
    _seed_patterns(
        _make_pattern(
            slug="alpha-skill",
            name="Alpha Skill",
            activity_ids=["act_abc", "act_def"],
            notes="Observed in review",
        )
    )
    save_profile("alpha-skill", "# Alpha Skill\n")

    result = _invoke("show", "alpha-skill")

    assert result.exit_code == 0
    assert "name: Alpha Skill" in result.output
    assert "slug: alpha-skill" in result.output
    assert (
        "- 2026-04-19 [work] activity_ids=act_abc,act_def notes=Observed in review"
        in result.output
    )
    assert "# Alpha Skill" in result.output


def test_show_json_shape(skill_cli_env):
    _seed_patterns(_make_pattern())
    save_profile("alpha-skill", "# Alpha Skill\n")

    result = _invoke("show", "alpha-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert set(payload) == {"pattern", "profile"}
    assert payload["pattern"]["slug"] == "alpha-skill"
    assert payload["profile"] == "# Alpha Skill\n"


def test_show_sorts_observations_for_text_output(skill_cli_env):
    _seed_patterns(_make_pattern(day="2026-04-20", created_at="2026-04-20T10:00:00Z"))

    def mutate(rows):
        rows = list(rows)
        rows[0]["observations"].append(
            {
                "day": "2026-04-19",
                "facet": "solpbc",
                "activity_ids": ["act_older"],
                "notes": "Earlier observation",
                "recorded_at": "2026-04-19T09:00:00Z",
            }
        )
        rows[0]["first_seen"] = "2026-04-19"
        rows[0]["last_seen"] = "2026-04-20"
        return rows

    locked_modify_patterns(mutate)

    result = _invoke("show", "alpha-skill")

    assert result.exit_code == 0
    first_index = result.output.index(
        "- 2026-04-19 [solpbc] activity_ids=act_older notes=Earlier observation"
    )
    second_index = result.output.index(
        "- 2026-04-20 [work] activity_ids=act_abc notes="
    )
    assert first_index < second_index


def test_observe_missing_slug_errors(skill_cli_env):
    result = _invoke(
        "observe",
        "missing-skill",
        "--day",
        "2026-04-20",
        "--facet",
        "work",
        "--activity-ids",
        "act_new",
    )

    assert result.exit_code == 1
    assert "no such skill" in result.stderr


def test_observe_appends_and_updates_derived_fields(skill_cli_env):
    _seed_patterns(_make_pattern())

    result = _invoke(
        "observe",
        "alpha-skill",
        "--day",
        "2026-04-20",
        "--facet",
        "personal",
        "--activity-ids",
        "act_new",
        "--notes",
        "Later observation",
        "--json",
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["facets_touched"] == ["personal", "work"]
    assert payload["first_seen"] == "2026-04-19"
    assert payload["last_seen"] == "2026-04-20"
    assert len(payload["observations"]) == 2
    assert payload["updated_at"].endswith("Z")


def test_observe_resurrects_dormant(skill_cli_env):
    _seed_patterns(_make_pattern(status="dormant"))

    result = _invoke(
        "observe",
        "alpha-skill",
        "--day",
        "2026-04-20",
        "--facet",
        "work",
        "--activity-ids",
        "act_new",
        "--json",
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "mature"


def test_observe_idempotent_exits_0_with_already_recorded(skill_cli_env):
    _seed_patterns(_make_pattern(activity_ids=["act_a", "act_b"]))

    result = _invoke(
        "observe",
        "alpha-skill",
        "--day",
        "2026-04-19",
        "--facet",
        "work",
        "--activity-ids",
        "act_b,act_a",
    )

    assert result.exit_code == 0
    assert "already observed" in result.stderr
    rows = load_patterns()
    assert len(rows[0]["observations"]) == 1


def test_seed_creates_pattern_with_initial_observation(skill_cli_env):
    result = _invoke(
        "seed",
        "alpha-skill",
        "--name",
        "Alpha Skill",
        "--day",
        "2026-04-19",
        "--facet",
        "work",
        "--activity-ids",
        "act_abc,act_def",
        "--notes",
        "Initial seed",
        "--json",
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["slug"] == "alpha-skill"
    assert payload["status"] == "emerging"
    assert payload["needs_profile"] is False
    assert payload["needs_refresh"] is False
    assert payload["profile_generated_at"] is None
    assert payload["facets_touched"] == ["work"]
    assert payload["first_seen"] == "2026-04-19"
    assert payload["last_seen"] == "2026-04-19"
    assert payload["observations"][0]["activity_ids"] == ["act_abc", "act_def"]
    assert payload["observations"][0]["notes"] == "Initial seed"


def test_seed_collision_errors_with_slug_already_exists(skill_cli_env):
    _seed_patterns(_make_pattern())

    result = _invoke(
        "seed",
        "alpha-skill",
        "--name",
        "Alpha Skill",
        "--day",
        "2026-04-19",
        "--facet",
        "work",
        "--activity-ids",
        "act_abc",
    )

    assert result.exit_code == 1
    assert "slug already exists" in result.stderr


def test_promote_missing_slug_errors(skill_cli_env):
    result = _invoke("promote", "missing-skill")

    assert result.exit_code == 1
    assert "no such skill" in result.stderr


def test_promote_sets_needs_profile(skill_cli_env):
    _seed_patterns(_make_pattern())

    result = _invoke("promote", "alpha-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["needs_profile"] is True


def test_promote_already_flagged_exits_0(skill_cli_env):
    _seed_patterns(_make_pattern(needs_profile=True))

    result = _invoke("promote", "alpha-skill")

    assert result.exit_code == 0
    assert "already flagged" in result.stderr


def test_promote_already_mature_exits_0(skill_cli_env):
    _seed_patterns(_make_pattern(status="mature"))

    result = _invoke("promote", "alpha-skill")

    assert result.exit_code == 0
    assert "already mature" in result.stderr


def test_refresh_not_mature_exits_1(skill_cli_env):
    _seed_patterns(_make_pattern(status="emerging"))

    result = _invoke("refresh", "alpha-skill")

    assert result.exit_code == 1
    assert "not mature" in result.stderr


def test_refresh_sets_needs_refresh_on_mature(skill_cli_env):
    _seed_patterns(_make_pattern(status="mature"))

    result = _invoke("refresh", "alpha-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["needs_refresh"] is True


def test_refresh_already_flagged_exits_0(skill_cli_env):
    _seed_patterns(_make_pattern(status="mature", needs_refresh=True))

    result = _invoke("refresh", "alpha-skill")

    assert result.exit_code == 0
    assert "already flagged" in result.stderr


def test_mark_dormant_sets_status(skill_cli_env):
    _seed_patterns(_make_pattern())

    result = _invoke("mark-dormant", "alpha-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "dormant"


def test_mark_dormant_already_flagged_exits_0(skill_cli_env):
    _seed_patterns(_make_pattern(status="dormant"))

    result = _invoke("mark-dormant", "alpha-skill")

    assert result.exit_code == 0
    assert "already flagged" in result.stderr


def test_retire_sets_status(skill_cli_env):
    _seed_patterns(_make_pattern())

    result = _invoke("retire", "alpha-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "retired"


def test_retire_already_flagged_exits_0(skill_cli_env):
    _seed_patterns(_make_pattern(status="retired"))

    result = _invoke("retire", "alpha-skill")

    assert result.exit_code == 0
    assert "already flagged" in result.stderr


def test_edit_request_appends_with_unique_id(skill_cli_env):
    _seed_patterns(_make_pattern())

    first = _invoke("edit-request", "alpha-skill", "--instructions", "revise opening")
    second = _invoke("edit-request", "alpha-skill", "--instructions", "expand examples")

    assert first.exit_code == 0
    assert second.exit_code == 0
    rows = load_edit_requests()
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]


def test_edit_request_on_retired_skill_allowed(skill_cli_env):
    _seed_patterns(_make_pattern(status="retired"))

    result = _invoke(
        "edit-request",
        "alpha-skill",
        "--instructions",
        "still worth polishing",
        "--json",
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["slug"] == "alpha-skill"


def test_edit_request_missing_slug_errors(skill_cli_env):
    result = _invoke(
        "edit-request",
        "missing-skill",
        "--instructions",
        "revise this",
    )

    assert result.exit_code == 1
    assert "no such skill" in result.stderr


def test_rename_moves_profile_and_updates_slug(skill_cli_env):
    _seed_patterns(_make_pattern())
    save_profile("alpha-skill", "# Alpha Skill\n")

    result = _invoke("rename", "alpha-skill", "renamed-skill", "--json")

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["slug"] == "renamed-skill"
    assert not profile_path("alpha-skill").exists()
    assert (
        profile_path("renamed-skill").read_text(encoding="utf-8") == "# Alpha Skill\n"
    )
    assert load_patterns()[0]["slug"] == "renamed-skill"


def test_rename_target_exists_errors(skill_cli_env):
    _seed_patterns(_make_pattern(slug="alpha-skill"), _make_pattern(slug="beta-skill"))

    result = _invoke("rename", "alpha-skill", "beta-skill")

    assert result.exit_code == 1
    assert "new slug already exists" in result.stderr


def test_rename_orphan_profile_target_exists_errors(skill_cli_env):
    _seed_patterns(_make_pattern(slug="alpha-skill"))
    save_profile("orphan-target", "# Orphan Target\n")

    result = _invoke("rename", "alpha-skill", "orphan-target")

    assert result.exit_code == 1
    assert "new slug already exists" in result.stderr
    assert (
        profile_path("orphan-target").read_text(encoding="utf-8") == "# Orphan Target\n"
    )
    assert load_patterns()[0]["slug"] == "alpha-skill"


def test_rename_missing_source_errors(skill_cli_env):
    _seed_patterns(_make_pattern(slug="alpha-skill"))

    result = _invoke("rename", "missing-skill", "beta-skill")

    assert result.exit_code == 1
    assert "no such skill" in result.stderr
