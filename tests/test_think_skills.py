# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from think.skills import (
    edit_requests_lock_path,
    edit_requests_path,
    find_pattern,
    load_edit_requests,
    load_patterns,
    load_profile,
    locked_modify_edit_requests,
    locked_modify_patterns,
    make_request_id,
    observation_key,
    patterns_lock_path,
    patterns_path,
    profile_path,
    rename_profile,
    save_edit_requests,
    save_patterns,
    save_profile,
    skills_dir,
    touch_updated,
    utc_now_iso,
)


@pytest.fixture
def skill_journal(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    return Path(tmp_path)


def test_skills_dir_creates_dir(skill_journal):
    path = skills_dir()

    assert path == skill_journal / "skills"
    assert path.exists()
    assert path.is_dir()


def test_path_helpers_return_expected_names(skill_journal):
    assert patterns_path() == skill_journal / "skills" / "patterns.jsonl"
    assert edit_requests_path() == skill_journal / "skills" / "edit_requests.jsonl"
    assert profile_path("alpha-skill") == skill_journal / "skills" / "alpha-skill.md"
    assert patterns_lock_path() == skill_journal / "skills" / ".patterns.lock"
    assert edit_requests_lock_path() == skill_journal / "skills" / ".edit_requests.lock"


def test_load_patterns_missing_file_returns_empty(skill_journal):
    assert load_patterns() == []


def test_save_and_load_patterns_roundtrip(skill_journal):
    rows = [
        {"slug": "alpha-skill", "status": "emerging"},
        {"slug": "beta-skill", "status": "mature"},
    ]

    save_patterns(rows)

    assert load_patterns() == rows


def test_save_patterns_empty_list_writes_empty_file(skill_journal):
    save_patterns([])

    assert patterns_path().read_text(encoding="utf-8") == ""


def test_load_patterns_skips_malformed_line(skill_journal, caplog):
    patterns_path().write_text('{"slug": "alpha-skill"}\nnot-json\n', encoding="utf-8")

    rows = load_patterns()

    assert rows == [{"slug": "alpha-skill"}]
    assert "malformed JSONL line 2" in caplog.text


def test_load_patterns_warns_on_non_dict_line(skill_journal, caplog):
    patterns_path().write_text('{"slug": "alpha-skill"}\n[1, 2]\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        rows = load_patterns()

    assert rows == [{"slug": "alpha-skill"}]
    assert "non-object JSONL line 2" in caplog.text
    assert "list" in caplog.text


def test_load_edit_requests_missing_file_returns_empty(skill_journal):
    assert load_edit_requests() == []


def test_save_and_load_edit_requests_roundtrip(skill_journal):
    rows = [
        {"id": "req_1", "slug": "alpha-skill"},
        {"id": "req_2", "slug": "beta-skill"},
    ]

    save_edit_requests(rows)

    assert load_edit_requests() == rows


def test_load_profile_missing_returns_none(skill_journal):
    assert load_profile("missing-skill") is None


def test_save_profile_writes_and_load_reads_back(skill_journal):
    save_profile("alpha-skill", "# Alpha Skill\n")

    assert load_profile("alpha-skill") == "# Alpha Skill\n"


def test_rename_profile_renames_and_returns_true(skill_journal):
    save_profile("old-skill", "# Old\n")

    renamed = rename_profile("old-skill", "new-skill")

    assert renamed is True
    assert not profile_path("old-skill").exists()
    assert load_profile("new-skill") == "# Old\n"


def test_rename_profile_missing_returns_false(skill_journal):
    assert rename_profile("missing-skill", "new-skill") is False


def test_rename_profile_target_exists_raises(skill_journal):
    save_profile("old-skill", "# Old\n")
    save_profile("new-skill", "# New\n")

    with pytest.raises(FileExistsError):
        rename_profile("old-skill", "new-skill")


def test_find_pattern_returns_row_or_none(skill_journal):
    rows = [{"slug": "alpha-skill"}, {"slug": "beta-skill"}]

    assert find_pattern("beta-skill", rows) == {"slug": "beta-skill"}
    assert find_pattern("missing-skill", rows) is None


def test_observation_key_is_deterministic_and_sort_invariant(skill_journal):
    assert observation_key("alpha", "2026-04-19", ["x", "y"]) == observation_key(
        "alpha", "2026-04-19", ["y", "x"]
    )


def test_make_request_id_unique_across_100_calls(skill_journal):
    request_ids = {make_request_id() for _ in range(100)}

    assert len(request_ids) == 100


def test_utc_now_iso_ends_with_z(skill_journal):
    assert utc_now_iso().endswith("Z")


def test_touch_updated_sets_updated_at(skill_journal):
    row = {}

    touch_updated(row)

    assert row["updated_at"].endswith("Z")


def test_locked_modify_patterns_applies_fn_and_persists(skill_journal):
    def mutate(rows):
        return list(rows) + [{"slug": "alpha-skill"}]

    updated = locked_modify_patterns(mutate)

    assert updated == [{"slug": "alpha-skill"}]
    assert load_patterns() == [{"slug": "alpha-skill"}]


def test_locked_modify_edit_requests_applies_fn_and_persists(skill_journal):
    def mutate(rows):
        return list(rows) + [{"id": "req_1", "slug": "alpha-skill"}]

    updated = locked_modify_edit_requests(mutate)

    assert updated == [{"id": "req_1", "slug": "alpha-skill"}]
    assert load_edit_requests() == [{"id": "req_1", "slug": "alpha-skill"}]


def test_locked_modify_patterns_serializes_threads(skill_journal):
    barrier = threading.Barrier(4)
    exceptions: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait()

            def mutate(rows):
                next_rows = list(rows)
                next_rows.append({"slug": f"s{i}"})
                return next_rows

            locked_modify_patterns(mutate)
        except BaseException as exc:  # pragma: no cover - assertion surface
            exceptions.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert exceptions == []
    rows = load_patterns()
    assert sorted(row["slug"] for row in rows) == ["s0", "s1", "s2", "s3"]
