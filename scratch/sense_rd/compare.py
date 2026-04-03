"""
Comparison utilities for Sense A/B testing.

Compares unified Sense agent output against baseline multi-agent pipeline outputs
from the field journal.
"""


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace."""
    return name.strip().lower()


def _names_match(a: str, b: str) -> bool:
    """
    Fuzzy name matching: exact match after normalization, or one is a
    substring of the other (handles "Laura" matching "Laura Smith").
    """
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    # Substring: shorter name appears in longer name
    if na in nb or nb in na:
        return True
    return False


def _find_match(name: str, candidates: list[str]) -> str | None:
    """Find first matching candidate for a name."""
    for c in candidates:
        if _names_match(name, c):
            return c
    return None


# ---------------------------------------------------------------------------
# Entity comparison
# ---------------------------------------------------------------------------

def compare_entities(sense_entities: list[dict], baseline_entities: list[dict]) -> dict:
    """
    Compare Sense entity list against baseline entities.jsonl.

    Both are lists of dicts with at least {"type", "name"}.
    Returns precision, recall, f1, and unmatched lists.
    """
    sense_names = [e.get("name", "") for e in sense_entities]
    baseline_names = [e.get("name", "") for e in baseline_entities]

    if not sense_names and not baseline_names:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "sense_only": [], "baseline_only": [],
        }

    # Track which baseline entities were matched
    matched_baseline = set()
    matched_sense = set()

    for i, sn in enumerate(sense_names):
        for j, bn in enumerate(baseline_names):
            if j not in matched_baseline and _names_match(sn, bn):
                matched_sense.add(i)
                matched_baseline.add(j)
                break

    precision = len(matched_sense) / len(sense_names) if sense_names else 0.0
    recall = len(matched_baseline) / len(baseline_names) if baseline_names else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    sense_only = [sense_names[i] for i in range(len(sense_names)) if i not in matched_sense]
    baseline_only = [baseline_names[j] for j in range(len(baseline_names)) if j not in matched_baseline]

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "sense_only": sense_only,
        "baseline_only": baseline_only,
    }


# ---------------------------------------------------------------------------
# Speaker comparison
# ---------------------------------------------------------------------------

def compare_speakers(sense_speakers: list[str], baseline_speakers: list[str]) -> dict:
    """
    Compare Sense speaker list against baseline speakers.json.
    Uses fuzzy name matching with Jaccard-like similarity.
    """
    if not sense_speakers and not baseline_speakers:
        return {
            "exact_match": True, "overlap": 1.0,
            "sense_only": [], "baseline_only": [],
        }

    sense_norm = {_normalize(s) for s in sense_speakers}
    baseline_norm = {_normalize(b) for b in baseline_speakers}

    # Fuzzy matching: build matched sets
    matched_baseline = set()
    matched_sense = set()

    for sn in sense_norm:
        for bn in baseline_norm:
            if bn not in matched_baseline and _names_match(sn, bn):
                matched_sense.add(sn)
                matched_baseline.add(bn)
                break

    union_size = len(sense_norm) + len(baseline_norm) - len(matched_sense)
    overlap = len(matched_sense) / union_size if union_size > 0 else 0.0

    exact_match = (sense_norm == baseline_norm)

    sense_only = [s for s in sense_speakers if _normalize(s) not in matched_sense]
    baseline_only = [b for b in baseline_speakers if _normalize(b) not in matched_baseline]

    return {
        "exact_match": exact_match,
        "overlap": round(overlap, 4),
        "sense_only": sense_only,
        "baseline_only": baseline_only,
    }


# ---------------------------------------------------------------------------
# Facet comparison
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def compare_facets(sense_facets: list[dict], baseline_facets: list[dict]) -> dict:
    """
    Compare Sense facet classifications against baseline facets.json.
    Each facet is {"facet": str, "activity": str, "level": str}.
    """
    sense_by_id = {f["facet"]: f for f in sense_facets}
    baseline_by_id = {f["facet"]: f for f in baseline_facets}

    sense_ids = set(sense_by_id.keys())
    baseline_ids = set(baseline_by_id.keys())

    facet_match = (sense_ids == baseline_ids)

    # Level comparison for overlapping facets
    common = sense_ids & baseline_ids
    level_matches = 0
    level_close = 0

    for fid in common:
        sl = _LEVEL_ORDER.get(sense_by_id[fid].get("level", ""), -1)
        bl = _LEVEL_ORDER.get(baseline_by_id[fid].get("level", ""), -1)
        if sl == bl:
            level_matches += 1
            level_close += 1
        elif abs(sl - bl) <= 1:
            level_close += 1

    n_common = len(common)

    return {
        "facet_match": facet_match,
        "level_match": level_matches == n_common if n_common > 0 else True,
        "level_close": level_close == n_common if n_common > 0 else True,
        "level_match_count": level_matches,
        "level_close_count": level_close,
        "common_facets": n_common,
        "sense_only_facets": list(sense_ids - baseline_ids),
        "baseline_only_facets": list(baseline_ids - sense_ids),
    }


# ---------------------------------------------------------------------------
# Density comparison
# ---------------------------------------------------------------------------

def compare_density(sense_density: str, baseline_density_data: str) -> dict:
    """
    Compare Sense density classification against baseline.
    baseline_density_data: "active" if segment has full agent outputs, else "idle"/"low_change".
    """
    match = _normalize(sense_density) == _normalize(baseline_density_data)
    return {
        "match": match,
        "sense": sense_density,
        "baseline": baseline_density_data,
    }


# ---------------------------------------------------------------------------
# Activity summary comparison
# ---------------------------------------------------------------------------

def _significant_words(text: str) -> set[str]:
    """Extract significant words (>3 chars, lowercased) from text."""
    words = set()
    for word in text.split():
        # Strip punctuation
        cleaned = "".join(c for c in word if c.isalnum())
        if len(cleaned) > 3:
            words.add(cleaned.lower())
    return words


def compare_activity_summary(sense_summary: str, baseline_activity_md: str) -> dict:
    """
    Compare Sense activity summary against baseline activity.md.
    Returns keyword overlap (Jaccard) and length ratio.
    """
    if not sense_summary and not baseline_activity_md:
        return {"keyword_overlap": 1.0, "length_ratio": 1.0}

    sense_words = _significant_words(sense_summary)
    baseline_words = _significant_words(baseline_activity_md)

    if not sense_words and not baseline_words:
        overlap = 1.0
    elif not sense_words or not baseline_words:
        overlap = 0.0
    else:
        intersection = sense_words & baseline_words
        union = sense_words | baseline_words
        overlap = len(intersection) / len(union)

    bl_len = len(baseline_activity_md) if baseline_activity_md else 1
    length_ratio = len(sense_summary) / bl_len

    return {
        "keyword_overlap": round(overlap, 4),
        "length_ratio": round(length_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Meeting detection comparison
# ---------------------------------------------------------------------------

def compare_meeting_detection(sense_meeting: bool, baseline_speakers: list[str] | None) -> dict:
    """
    Compare Sense meeting_detected against baseline.
    Baseline meeting = speakers.json exists with non-empty array.
    """
    baseline_meeting = bool(baseline_speakers)
    match = sense_meeting == baseline_meeting
    return {
        "match": match,
        "sense": sense_meeting,
        "baseline": baseline_meeting,
        "baseline_speakers": baseline_speakers or [],
    }
