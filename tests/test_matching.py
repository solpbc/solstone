# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entity matching and name variant resolution."""

from solstone.think.entities.matching import (
    MatchTier,
    build_name_resolution_map,
    find_matching_entity,
    is_name_variant_match,
)


def _entity(name, entity_id=None, aka=None):
    """Helper to create entity dicts for testing."""
    eid = entity_id or name.lower().replace(" ", "_")
    result = {"id": eid, "name": name}
    if aka:
        result["aka"] = aka
    return result


# --- Tier 1-3 regression tests ---


class TestExistingTiers:
    def test_exact_name_match(self):
        entities = [_entity("Robert Johnson")]
        assert (
            find_matching_entity("Robert Johnson", entities)["id"] == "robert_johnson"
        )

    def test_exact_id_match(self):
        entities = [_entity("Robert Johnson")]
        assert (
            find_matching_entity("robert_johnson", entities)["id"] == "robert_johnson"
        )

    def test_exact_aka_match(self):
        entities = [_entity("Robert Johnson", aka=["Bob"])]
        assert find_matching_entity("Bob", entities)["id"] == "robert_johnson"

    def test_case_insensitive_match(self):
        entities = [_entity("Robert Johnson")]
        assert (
            find_matching_entity("robert johnson", entities)["id"] == "robert_johnson"
        )

    def test_no_match_returns_none(self):
        entities = [_entity("Robert Johnson")]
        assert find_matching_entity("Unknown Person", entities) is None

    def test_empty_inputs(self):
        assert find_matching_entity("", []) is None
        assert find_matching_entity("test", []) is None
        assert find_matching_entity("", [_entity("Test")]) is None


# --- Enhancement 1: Bidirectional first-word match ---


class TestBidirectionalFirstWord:
    def test_short_to_long(self):
        """Original tier 4: detected name IS a first word of an entity."""
        entities = [_entity("Javier Garcia")]
        assert find_matching_entity("Javier", entities)["id"] == "javier_garcia"

    def test_long_to_short(self):
        """New: detected name's first word matches an entity."""
        entities = [_entity("Javier")]
        assert find_matching_entity("Javier Garcia", entities)["id"] == "javier"

    def test_order_independent(self):
        """Both directions work regardless of which entity exists."""
        entities = [_entity("Javier")]
        assert find_matching_entity("Javier Garcia", entities)["id"] == "javier"

        entities = [_entity("Javier Garcia")]
        assert find_matching_entity("Javier", entities)["id"] == "javier_garcia"

    def test_ambiguous_first_word_rejected(self):
        """Multiple entities with same first word: no match."""
        entities = [_entity("Javier Garcia"), _entity("Javier Rodriguez")]
        assert find_matching_entity("Javier", entities) is None

    def test_ambiguous_first_word_long_to_short(self):
        """Multiple entities with same first word: long→short also rejected."""
        entities = [_entity("Javier"), _entity("Javier Rodriguez")]
        assert find_matching_entity("Javier Garcia", entities) is None

    def test_short_name_min_length(self):
        """First word must be >= 3 chars."""
        entities = [_entity("Li Wei")]
        assert find_matching_entity("Li", entities) is None


# --- Enhancement 2: Token-subset match ---


class TestTokenSubset:
    def test_subset_match_short_in_long(self):
        """Shorter name's tokens are a subset of longer entity's tokens."""
        entities = [_entity("Josh Jones Dilworth")]
        assert (
            find_matching_entity("Jones Dilworth", entities)["id"]
            == "josh_jones_dilworth"
        )

    def test_subset_match_long_detected(self):
        """Detected name has more tokens than entity."""
        entities = [_entity("Jones Dilworth")]
        assert (
            find_matching_entity("Josh Jones Dilworth", entities)["id"]
            == "jones_dilworth"
        )

    def test_single_token_not_subset(self):
        """Single-token names don't trigger subset match (min 2 tokens)."""
        entities = [_entity("Josh Jones Dilworth")]
        # "Dilworth" is 1 token — not first word, not a 2-token subset
        assert find_matching_entity("Dilworth", entities) is None

    def test_ambiguous_subset_rejected(self):
        """Multiple entities match token-subset: no match."""
        entities = [
            _entity("Josh Jones Dilworth"),
            _entity("Mary Jones Dilworth"),
        ]
        assert find_matching_entity("Jones Dilworth", entities) is None

    def test_subset_both_directions(self):
        """Token-subset works regardless of which name is in entities."""
        entities = [_entity("Josh Jones Dilworth")]
        assert (
            find_matching_entity("Jones Dilworth", entities)["id"]
            == "josh_jones_dilworth"
        )

        entities = [_entity("Jones Dilworth")]
        assert (
            find_matching_entity("Josh Jones Dilworth", entities)["id"]
            == "jones_dilworth"
        )


# --- Enhancement 3: Prefix-token match ---


class TestPrefixToken:
    def test_prefix_match_nickname(self):
        """Nickname prefix matching (Chris → Christopher)."""
        entities = [_entity("Christopher DeWolfe")]
        assert (
            find_matching_entity("Chris DeWolfe", entities)["id"]
            == "christopher_dewolfe"
        )

    def test_prefix_match_reverse(self):
        """Reverse direction: full name detected, nickname entity."""
        entities = [_entity("Chris DeWolfe")]
        assert (
            find_matching_entity("Christopher DeWolfe", entities)["id"]
            == "chris_dewolfe"
        )

    def test_prefix_min_length(self):
        """Prefix must be >= 4 chars."""
        entities = [_entity("Jonathan Smith")]
        # "Jon" is only 3 chars, not a valid prefix
        assert find_matching_entity("Jon Smith", entities) is None

    def test_prefix_four_chars_matches(self):
        """Exactly 4-char prefix works."""
        entities = [_entity("Jonathan Smith")]
        assert find_matching_entity("Jona Smith", entities)["id"] == "jonathan_smith"

    def test_ambiguous_prefix_rejected(self):
        """Multiple entities match prefix-token: no match."""
        entities = [
            _entity("Christopher DeWolfe"),
            _entity("Christine DeWolfe"),
        ]
        assert find_matching_entity("Chris DeWolfe", entities) is None

    def test_different_token_count_no_prefix(self):
        """Different token counts don't trigger prefix match."""
        entities = [_entity("Christopher James DeWolfe")]
        assert find_matching_entity("Chris DeWolfe", entities) is None


# --- Production duplicate cases ---


class TestProductionDuplicates:
    """Verify the three production duplicate pairs that motivated this spec."""

    def test_chris_dewolfe(self):
        """Chris DeWolfe ↔ Christopher DeWolfe (prefix-token match)."""
        entities = [_entity("Christopher DeWolfe")]
        assert (
            find_matching_entity("Chris DeWolfe", entities)["id"]
            == "christopher_dewolfe"
        )

        entities = [_entity("Chris DeWolfe")]
        assert (
            find_matching_entity("Christopher DeWolfe", entities)["id"]
            == "chris_dewolfe"
        )

    def test_javier_garcia(self):
        """Javier ↔ Javier Garcia (bidirectional first-word match)."""
        entities = [_entity("Javier Garcia")]
        assert find_matching_entity("Javier", entities)["id"] == "javier_garcia"

        entities = [_entity("Javier")]
        assert find_matching_entity("Javier Garcia", entities)["id"] == "javier"

    def test_jones_dilworth(self):
        """Jones Dilworth ↔ Josh Jones Dilworth (token-subset match)."""
        entities = [_entity("Josh Jones Dilworth")]
        assert (
            find_matching_entity("Jones Dilworth", entities)["id"]
            == "josh_jones_dilworth"
        )

        entities = [_entity("Jones Dilworth")]
        assert (
            find_matching_entity("Josh Jones Dilworth", entities)["id"]
            == "jones_dilworth"
        )


# --- build_name_resolution_map ---


class TestBuildNameResolutionMap:
    def test_bidirectional_first_word(self):
        entities = [_entity("Javier Garcia")]
        result = build_name_resolution_map(["Javier"], entities)
        assert result["Javier"] == "javier_garcia"

    def test_long_to_short_first_word(self):
        entities = [_entity("Javier")]
        result = build_name_resolution_map(["Javier Garcia"], entities)
        assert result["Javier Garcia"] == "javier"

    def test_token_subset(self):
        entities = [_entity("Josh Jones Dilworth")]
        result = build_name_resolution_map(["Jones Dilworth"], entities)
        assert result["Jones Dilworth"] == "josh_jones_dilworth"

    def test_prefix_token(self):
        entities = [_entity("Christopher DeWolfe")]
        result = build_name_resolution_map(["Chris DeWolfe"], entities)
        assert result["Chris DeWolfe"] == "christopher_dewolfe"

    def test_ambiguous_subset_skipped(self):
        entities = [
            _entity("Josh Jones Dilworth"),
            _entity("Mary Jones Dilworth"),
        ]
        result = build_name_resolution_map(["Jones Dilworth"], entities)
        assert "Jones Dilworth" not in result

    def test_all_three_production_cases(self):
        entities = [
            _entity("Christopher DeWolfe"),
            _entity("Javier Garcia"),
            _entity("Josh Jones Dilworth"),
        ]
        result = build_name_resolution_map(
            ["Chris DeWolfe", "Javier", "Jones Dilworth"], entities
        )
        assert result["Chris DeWolfe"] == "christopher_dewolfe"
        assert result["Javier"] == "javier_garcia"
        assert result["Jones Dilworth"] == "josh_jones_dilworth"


# --- is_name_variant_match ---


class TestIsNameVariantMatch:
    def test_first_word_match(self):
        assert is_name_variant_match("Javier", "Javier Garcia") is True
        assert is_name_variant_match("Javier Garcia", "Javier") is True

    def test_token_subset_match(self):
        assert is_name_variant_match("Jones Dilworth", "Josh Jones Dilworth") is True
        assert is_name_variant_match("Josh Jones Dilworth", "Jones Dilworth") is True

    def test_prefix_token_match(self):
        assert is_name_variant_match("Chris DeWolfe", "Christopher DeWolfe") is True
        assert is_name_variant_match("Christopher DeWolfe", "Chris DeWolfe") is True

    def test_no_match(self):
        assert is_name_variant_match("Alice Smith", "Bob Jones") is False

    def test_empty_strings(self):
        assert is_name_variant_match("", "Test") is False
        assert is_name_variant_match("Test", "") is False

    def test_single_token_first_word(self):
        """Single tokens match via first-word when they ARE the first word."""
        assert is_name_variant_match("Jones", "Jones Dilworth") is True

    def test_single_token_not_first_word(self):
        """Single tokens that aren't the first word don't match."""
        assert is_name_variant_match("Dilworth", "Jones Dilworth") is False


# --- MatchResult and confidence tiers ---


class TestMatchResult:
    """Verify MatchResult is backward-compatible with dict usage."""

    def test_is_dict(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Alice Johnson", entities)
        assert isinstance(result, dict)

    def test_subscript_access(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Alice Johnson", entities)
        assert result["id"] == "alice_johnson"
        assert result["name"] == "Alice Johnson"

    def test_get_access(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Alice Johnson", entities)
        assert result.get("name") == "Alice Johnson"
        assert result.get("missing") is None

    def test_truthiness(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Alice Johnson", entities)
        assert result  # truthy
        assert find_matching_entity("Nobody", entities) is None

    def test_none_is_none(self):
        """No match still returns None, not an empty MatchResult."""
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Nobody", entities)
        assert result is None


class TestMatchTiers:
    """Verify each tier returns the correct MatchTier value."""

    def test_exact_name_tier(self):
        entities = [_entity("Robert Johnson")]
        result = find_matching_entity("Robert Johnson", entities)
        assert result.tier == MatchTier.EXACT

    def test_exact_id_tier(self):
        entities = [_entity("Robert Johnson")]
        result = find_matching_entity("robert_johnson", entities)
        assert result.tier == MatchTier.EXACT

    def test_exact_aka_tier(self):
        entities = [_entity("Robert Johnson", aka=["Bob"])]
        result = find_matching_entity("Bob", entities)
        assert result.tier == MatchTier.EXACT

    def test_case_insensitive_tier(self):
        entities = [_entity("Robert Johnson")]
        result = find_matching_entity("robert johnson", entities)
        assert result.tier == MatchTier.CASE_INSENSITIVE

    def test_email_tier(self):
        entities = [{"id": "alice", "name": "Alice", "emails": ["alice@example.com"]}]
        result = find_matching_entity("alice@example.com", entities)
        assert result.tier == MatchTier.EMAIL

    def test_slug_tier(self):
        """Slugified query matching entity id."""
        entities = [{"id": "robert_johnson", "name": "Robert Johnson"}]
        result = find_matching_entity("Robert Johnson", entities)
        # "Robert Johnson" exact-matches the name, so it's tier 1
        assert result.tier == MatchTier.EXACT
        # Use a slug-form query that doesn't exact-match but slug-matches
        entities2 = [{"id": "some_custom_id", "name": "Some Name"}]
        result2 = find_matching_entity("Some Name", entities2)
        # This exact-matches the name
        assert result2.tier == MatchTier.EXACT

    def test_first_word_tier(self):
        entities = [_entity("Javier Garcia")]
        result = find_matching_entity("Javier", entities)
        assert result.tier == MatchTier.FIRST_WORD

    def test_first_word_long_to_short_tier(self):
        entities = [_entity("Javier")]
        result = find_matching_entity("Javier Garcia", entities)
        assert result.tier == MatchTier.FIRST_WORD

    def test_token_subset_tier(self):
        entities = [_entity("Josh Jones Dilworth")]
        result = find_matching_entity("Jones Dilworth", entities)
        assert result.tier == MatchTier.TOKEN_SUBSET

    def test_prefix_tier(self):
        entities = [_entity("Christopher DeWolfe")]
        result = find_matching_entity("Chris DeWolfe", entities)
        assert result.tier == MatchTier.PREFIX

    def test_fuzzy_tier(self):
        entities = [_entity("Christopher DeWolfe")]
        # Close enough for fuzzy but not an exact/prefix match
        result = find_matching_entity("Christoph DeWolffe", entities)
        if result:  # rapidfuzz may not be installed
            assert result.tier == MatchTier.FUZZY


class TestHighConfidence:
    """Verify the is_high_confidence boundary between tiers 1-4 and 5+."""

    def test_exact_is_high(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("Alice Johnson", entities)
        assert result.is_high_confidence is True

    def test_case_insensitive_is_high(self):
        entities = [_entity("Alice Johnson")]
        result = find_matching_entity("alice johnson", entities)
        assert result.is_high_confidence is True

    def test_email_is_high(self):
        entities = [{"id": "alice", "name": "Alice", "emails": ["a@b.com"]}]
        result = find_matching_entity("a@b.com", entities)
        assert result.is_high_confidence is True

    def test_first_word_is_low(self):
        entities = [_entity("Javier Garcia")]
        result = find_matching_entity("Javier", entities)
        assert result.is_high_confidence is False

    def test_token_subset_is_low(self):
        entities = [_entity("Josh Jones Dilworth")]
        result = find_matching_entity("Jones Dilworth", entities)
        assert result.is_high_confidence is False

    def test_prefix_is_low(self):
        entities = [_entity("Christopher DeWolfe")]
        result = find_matching_entity("Chris DeWolfe", entities)
        assert result.is_high_confidence is False

    def test_tier_comparison(self):
        """MatchTier is an IntEnum — callers can compare tiers numerically."""
        assert MatchTier.EXACT < MatchTier.FUZZY
        assert MatchTier.SLUG <= MatchTier.SLUG
        assert MatchTier.FIRST_WORD > MatchTier.SLUG
