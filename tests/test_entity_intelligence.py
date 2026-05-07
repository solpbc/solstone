# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest

from solstone.think.indexer.journal import (
    get_entity_intelligence,
    get_entity_strength,
    scan_journal,
    search_entities,
)


@pytest.fixture(autouse=True)
def indexed_journal(journal_copy):
    scan_journal(str(journal_copy), full=True)


class TestEntityStrength:
    def test_returns_ranked_list(self):
        results = get_entity_strength()
        assert isinstance(results, list)
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_has_component_signals(self):
        results = get_entity_strength()
        r = results[0]
        for key in (
            "kg_edge_count",
            "co_occurrence",
            "appearance",
            "recency",
            "facet_breadth",
            "observation_depth",
            "score",
        ):
            assert key in r

    def test_alice_johnson_has_signals(self):
        results = get_entity_strength()
        names = [r.get("entity_id") or r.get("entity_name") for r in results]
        assert "alice_johnson" in names

    def test_with_facet_filter(self):
        results = get_entity_strength(facet="work")
        assert isinstance(results, list)

    def test_with_since_filter(self):
        results = get_entity_strength(since="20240105")
        assert isinstance(results, list)

    def test_limit(self):
        results = get_entity_strength(limit=2)
        assert len(results) <= 2


class TestSearchEntities:
    def test_by_type_person(self):
        results = search_entities(entity_type="Person")
        assert isinstance(results, list)
        for r in results:
            assert r["type"] == "Person"

    def test_by_type_company(self):
        results = search_entities(entity_type="Company")
        for r in results:
            assert r["type"] == "Company"

    def test_by_facet(self):
        results = search_entities(facet="work")
        assert isinstance(results, list)

    def test_by_query(self):
        results = search_entities(query="Alice")
        assert isinstance(results, list)
        assert any(r["name"] in {"Alice Johnson", "Alice"} for r in results)

    def test_all_entities(self):
        results = search_entities()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_structure(self):
        results = search_entities()
        if results:
            r = results[0]
            assert "entity_id" in r
            assert "name" in r
            assert "type" in r

    def test_limit(self):
        results = search_entities(limit=3)
        assert len(results) <= 3


class TestEntityIntelligence:
    def test_full_briefing(self):
        result = get_entity_intelligence("Alice Johnson")
        assert result is not None
        for section in (
            "identity",
            "relationships",
            "observations",
            "activity",
            "strength",
            "network",
            "facets",
        ):
            assert section in result

    def test_identity_section(self):
        result = get_entity_intelligence("Alice Johnson")
        assert result["identity"]["entity_id"] == "alice_johnson"
        assert result["identity"]["name"] == "Alice Johnson"

    def test_activity_section(self):
        result = get_entity_intelligence("Alice Johnson")
        assert isinstance(result["activity"], list)
        assert len(result["activity"]) > 0

    def test_strength_section(self):
        result = get_entity_intelligence("Alice Johnson")
        s = result["strength"]
        assert "score" in s
        assert s["score"] > 0

    def test_network_section(self):
        result = get_entity_intelligence("Alice Johnson")
        assert isinstance(result["network"], dict)

    def test_by_slug(self):
        result = get_entity_intelligence("alice_johnson")
        assert result is not None
        assert result["identity"]["entity_id"] == "alice_johnson"

    def test_unknown_entity(self):
        result = get_entity_intelligence("Nonexistent Person")
        assert result is None

    def test_with_facet(self):
        result = get_entity_intelligence("Alice Johnson", facet="personal")
        assert result is not None

    def test_bob_smith(self):
        result = get_entity_intelligence("Bob Smith")
        assert result is not None
        assert result["identity"]["entity_id"] == "bob_smith"


class TestEntityIntelligenceBrief:
    def test_brief_has_meta(self):
        result = get_entity_intelligence("Alice Johnson", brief=True)
        assert result is not None
        assert "_meta" in result
        assert result["_meta"]["brief"] is True

    def test_brief_has_all_sections(self):
        result = get_entity_intelligence("Alice Johnson", brief=True)
        for section in (
            "identity",
            "relationships",
            "observations",
            "activity",
            "strength",
            "network",
            "facets",
        ):
            assert section in result

    def test_brief_truncates_activity(self):
        result = get_entity_intelligence("Alice Johnson", brief=True)
        assert len(result["activity"]) <= 20
        assert result["_meta"]["activity_included"] == len(result["activity"])

    def test_brief_truncates_network(self):
        result = get_entity_intelligence("Alice Johnson", brief=True)
        assert len(result["network"]) <= 20
        assert result["_meta"]["network_included"] == len(result["network"])

    def test_brief_meta_counts_consistent(self):
        result = get_entity_intelligence("Alice Johnson", brief=True)
        meta = result["_meta"]
        assert meta["activity_included"] <= meta["activity_total"]
        assert meta["network_included"] <= meta["network_total"]

    def test_brief_preserves_other_sections(self):
        full = get_entity_intelligence("Alice Johnson")
        brief = get_entity_intelligence("Alice Johnson", brief=True)
        for key in ("identity", "relationships", "observations", "strength", "facets"):
            assert full[key] == brief[key]

    def test_default_no_meta(self):
        result = get_entity_intelligence("Alice Johnson")
        assert "_meta" not in result
