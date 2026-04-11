# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import os
from pathlib import Path
from unittest.mock import patch

from talent.skills import (
    MATCH_THRESHOLD,
    _extract_keywords,
    _find_best_match,
    _jaccard,
    _load_patterns,
    _make_slug,
    _normalize_entities,
    _pattern_score,
    _write_patterns,
    post_process,
    pre_process,
)

os.environ.setdefault("_SOLSTONE_JOURNAL_OVERRIDE", "tests/fixtures/journal")


_DEFAULT_ENTITIES = ["Alice", "Bob"]


def _make_context(
    facet="test-facet",
    day="20260410",
    activity_type="meeting",
    activity_id="act-001",
    description="Weekly standup with engineering team",
    entities=None,
):
    return {
        "facet": facet,
        "day": day,
        "activity": {
            "activity": activity_type,
            "id": activity_id,
            "description": description,
            "active_entities": _DEFAULT_ENTITIES if entities is None else entities,
        },
    }


SAMPLE_LLM_RESULT = json.dumps(
    {
        "skill_name": "Engineering Standup Facilitation",
        "slug": "meeting-alice-bob-standup-engineering",
        "category": "communication",
        "description": "Facilitates regular engineering standup meetings.",
        "how": "Runs structured standups covering blockers and progress.",
        "why": "Keeps the engineering team aligned and unblocked.",
        "tools": ["Zoom", "Jira"],
        "collaborators": ["Alice", "Bob"],
        "confidence": 0.85,
    }
)


class TestKeywordExtraction:
    def test_basic_extraction(self):
        result = _extract_keywords("Weekly standup with engineering team")
        assert "weekly" in result
        assert "standup" in result
        assert "engineering" in result
        assert "team" in result

    def test_stopword_removal(self):
        result = _extract_keywords("the quick brown fox")
        assert "the" not in result

    def test_short_words_removed(self):
        assert _extract_keywords("a an is") == []

    def test_numeric_removed(self):
        result = _extract_keywords("meeting 123 items")
        assert "123" not in result

    def test_empty_string(self):
        assert _extract_keywords("") == []


class TestNormalizeEntities:
    def test_basic(self):
        assert _normalize_entities(["Alice", "Bob"]) == ["alice", "bob"]

    def test_dedup(self):
        assert _normalize_entities(["Alice", "alice"]) == ["alice"]

    def test_non_string(self):
        assert _normalize_entities([123]) == ["123"]


class TestJaccard:
    def test_identical(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        assert _jaccard({1, 2, 3}, {2, 3, 4}) == 0.5

    def test_empty(self):
        assert _jaccard(set(), set()) == 0.0


class TestMakeSlug:
    def test_basic(self):
        assert _make_slug("meeting", ["alice"], ["standup"]) == "meeting-alice-standup"

    def test_special_chars(self):
        slug = _make_slug("Team Meeting!", ["Alice Smith"], ["Q2/Planning"])
        assert slug == "team-meeting-alice-smith-q2-planning"

    def test_truncation(self):
        slug = _make_slug(
            "meeting",
            ["a" * 40, "b" * 40],
            ["c" * 40, "d" * 40, "e" * 40],
        )
        assert len(slug) <= 80

    def test_empty_fallback(self):
        assert _make_slug("meeting", [], []) == "meeting-pattern"


class TestPatternScore:
    def test_matching_entities(self):
        pattern = {
            "activity_type": "meeting",
            "entities": ["alice", "bob"],
            "keywords": ["engineering", "standup"],
        }
        score = _pattern_score(
            {"alice", "carol"},
            {"engineering", "weekly"},
            pattern,
        )
        assert score is not None
        assert score >= MATCH_THRESHOLD

    def test_no_overlap(self):
        pattern = {
            "activity_type": "meeting",
            "entities": ["alice"],
            "keywords": ["standup"],
        }
        assert _pattern_score({"carol"}, {"planning"}, pattern) is None

    def test_exact_type_required(self):
        patterns = [
            {
                "id": "meeting-alice",
                "activity_type": "meeting",
                "entities": ["alice"],
                "keywords": ["standup"],
            }
        ]
        match = _find_best_match("email", {"alice"}, {"standup"}, patterns)
        assert match is None

    def test_entity_only_signal(self):
        pattern = {"activity_type": "meeting", "entities": ["alice"], "keywords": []}
        score = _pattern_score({"alice"}, set(), pattern)
        assert score == 1.0


class TestPreProcess:
    def test_novel_activity_seeds_pattern_and_skips(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            result = pre_process(_make_context())
            patterns = _load_patterns("test-facet")

        assert result == {"skip_reason": "first observation, seeded pattern"}
        assert len(patterns) == 1
        assert patterns[0]["activity_type"] == "meeting"
        assert len(patterns[0]["observations"]) == 1

    def test_second_observation_returns_template_vars(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            pre_process(_make_context())
            result = pre_process(
                _make_context(
                    day="20260411",
                    activity_id="act-002",
                    description="Weekly engineering standup with Alice and Bob",
                )
            )

        assert "template_vars" in result
        assert "meta" in result
        assert result["meta"]["mode"] == "comparison"
        assert {
            "skill_instruction",
            "pattern_context",
            "previous_outputs",
        } <= set(result["template_vars"])

    def test_third_observation_triggers_generation(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            pre_process(_make_context())
            pre_process(
                _make_context(
                    day="20260411",
                    activity_id="act-002",
                    description="Weekly engineering standup with Alice and Bob",
                )
            )
            result = pre_process(
                _make_context(
                    day="20260412",
                    activity_id="act-003",
                    description="Weekly engineering standup and blocker review",
                )
            )

        assert result["meta"]["mode"] == "generate"

    def test_no_signal_activity_skips(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            result = pre_process(_make_context(entities=[], description=""))
            patterns = _load_patterns("test-facet")

        assert result == {"skip_reason": "no signal to seed pattern"}
        assert len(patterns) == 0

    def test_entities_and_keywords_merged(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            pre_process(
                _make_context(
                    entities=["Alice", "Bob"],
                    description="Weekly standup with engineering team",
                )
            )
            pre_process(
                _make_context(
                    day="20260411",
                    activity_id="act-002",
                    entities=["Alice", "Carol"],
                    description="Weekly standup with product team",
                )
            )
            patterns = _load_patterns("test-facet")

        assert patterns[0]["entities"] == ["alice", "bob", "carol"]
        assert "engineering" in patterns[0]["keywords"]
        assert "product" in patterns[0]["keywords"]


class TestPostProcess:
    def _seed_pattern(self, tmp_path, skill_generated=False):
        pattern = {
            "id": "meeting-alice-bob-standup-engineering",
            "activity_type": "meeting",
            "keywords": ["engineering", "standup", "weekly"],
            "entities": ["alice", "bob"],
            "observations": [
                {
                    "day": "20260410",
                    "activity_id": "act-001",
                    "description": "Weekly standup with engineering team",
                    "entities": ["alice", "bob"],
                    "keywords": ["engineering", "standup", "weekly"],
                },
                {
                    "day": "20260411",
                    "activity_id": "act-002",
                    "description": "Weekly engineering standup with Alice and Bob",
                    "entities": ["alice", "bob"],
                    "keywords": ["engineering", "standup", "weekly"],
                },
                {
                    "day": "20260412",
                    "activity_id": "act-003",
                    "description": "Weekly engineering standup and blocker review",
                    "entities": ["alice", "bob"],
                    "keywords": [
                        "blocker",
                        "engineering",
                        "review",
                        "standup",
                        "weekly",
                    ],
                },
            ],
            "created_at": "2026-04-10T00:00:00+00:00",
            "updated_at": "2026-04-12T00:00:00+00:00",
            "skill_generated": skill_generated,
        }
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            _write_patterns("test-facet", [pattern])

    def test_comparison_mode_is_noop(self, tmp_path):
        self._seed_pattern(tmp_path)
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            result = post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "comparison",
                    }
                },
            )

        assert result is None
        skill_path = (
            Path(tmp_path)
            / "facets"
            / "test-facet"
            / "skills"
            / "meeting-alice-bob-standup-engineering.md"
        )
        assert not skill_path.exists()

    def test_generates_skill_document(self, tmp_path):
        self._seed_pattern(tmp_path)
        with (
            patch("talent.skills.get_journal", return_value=str(tmp_path)),
            patch("talent.skills.update_identity_section"),
        ):
            post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )

        skill_path = (
            Path(tmp_path)
            / "facets"
            / "test-facet"
            / "skills"
            / "meeting-alice-bob-standup-engineering.md"
        )
        content = skill_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert 'slug: "meeting-alice-bob-standup-engineering"' in content
        assert "## Description" in content
        assert "## How" in content
        assert "## Why" in content

    def test_writes_patterns_atomically(self, tmp_path):
        self._seed_pattern(tmp_path)
        with (
            patch("talent.skills.get_journal", return_value=str(tmp_path)),
            patch("talent.skills.update_identity_section"),
        ):
            post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )
            patterns = _load_patterns("test-facet")

        assert patterns[0]["skill_generated"] is True
        assert patterns[0]["updated_at"] != "2026-04-12T00:00:00+00:00"

    def test_agency_notification(self, tmp_path):
        self._seed_pattern(tmp_path)
        with (
            patch("talent.skills.get_journal", return_value=str(tmp_path)),
            patch("talent.skills.update_identity_section") as mock_update,
            patch("talent.skills._read_agency_observations", return_value=""),
        ):
            post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )

        mock_update.assert_called_once()
        args = mock_update.call_args.args
        assert args[0] == "agency.md"
        assert args[1] == "observations"
        assert "Engineering Standup Facilitation" in args[2]

    def test_agency_notification_appends_to_existing(self, tmp_path):
        self._seed_pattern(tmp_path)
        with (
            patch("talent.skills.get_journal", return_value=str(tmp_path)),
            patch("talent.skills.update_identity_section") as mock_update,
            patch(
                "talent.skills._read_agency_observations",
                return_value="- Existing observation about something",
            ),
        ):
            post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )

        args = mock_update.call_args.args
        content = args[2]
        assert "Existing observation" in content
        assert "Engineering Standup Facilitation" in content

    def test_refresh_does_not_notify_agency(self, tmp_path):
        self._seed_pattern(tmp_path, skill_generated=True)
        with (
            patch("talent.skills.get_journal", return_value=str(tmp_path)),
            patch("talent.skills.update_identity_section") as mock_update,
        ):
            post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "refresh",
                    }
                },
            )

        mock_update.assert_not_called()
        skill_path = (
            Path(tmp_path)
            / "facets"
            / "test-facet"
            / "skills"
            / "meeting-alice-bob-standup-engineering.md"
        )
        assert skill_path.exists()

    def test_malformed_json_result(self, tmp_path):
        self._seed_pattern(tmp_path)
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            result = post_process(
                "not valid json",
                {
                    "meta": {
                        "pattern_id": "meeting-alice-bob-standup-engineering",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )
        assert result is None

    def test_missing_pattern_id(self, tmp_path):
        self._seed_pattern(tmp_path)
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            result = post_process(
                SAMPLE_LLM_RESULT,
                {
                    "meta": {
                        "pattern_id": "missing-pattern",
                        "facet": "test-facet",
                        "mode": "generate",
                    }
                },
            )
        assert result is None


class TestLoadPatterns:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "facets" / "test-facet" / "skills"
        path.mkdir(parents=True)
        (path / "patterns.jsonl").write_text("", encoding="utf-8")
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            assert _load_patterns("test-facet") == []

    def test_missing_file(self, tmp_path):
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            assert _load_patterns("test-facet") == []

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "facets" / "test-facet" / "skills"
        path.mkdir(parents=True)
        (path / "patterns.jsonl").write_text(
            '{"id": "one", "activity_type": "meeting"}\nnot-json\n{"id": "two", "activity_type": "meeting"}\n',
            encoding="utf-8",
        )
        with patch("talent.skills.get_journal", return_value=str(tmp_path)):
            patterns = _load_patterns("test-facet")
        assert [pattern["id"] for pattern in patterns] == ["one", "two"]
