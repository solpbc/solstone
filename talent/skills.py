# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from think.activities import get_activity_output_path
from think.entities.core import atomic_write
from think.identity import update_identity_section
from think.utils import get_journal

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "before",
        "between",
        "into",
        "through",
        "during",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "also",
        "that",
        "this",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "its",
        "his",
        "her",
        "their",
        "our",
        "your",
        "any",
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "me",
        "him",
        "them",
        "us",
        "my",
        "up",
    }
)
AGENT_OUTPUT_KEYS = ["decisions", "followups", "meetings", "messaging"]
MATCH_THRESHOLD = 0.3


def _skills_dir(facet: str) -> Path:
    path = Path(get_journal()) / "facets" / facet / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _patterns_path(facet: str) -> Path:
    return _skills_dir(facet) / "patterns.jsonl"


def _load_patterns(facet: str) -> list[dict]:
    path = _patterns_path(facet)
    if not path.exists():
        return []

    patterns = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                patterns.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skills: malformed patterns.jsonl line %s", lineno)
    return patterns


def _write_patterns(facet: str, patterns: list[dict]) -> None:
    content = ""
    if patterns:
        content = "\n".join(
            json.dumps(pattern, ensure_ascii=False) for pattern in patterns
        )
        content += "\n"
    atomic_write(_patterns_path(facet), content)


def _extract_keywords(text: str) -> list[str]:
    keywords = set()
    for token in re.split(r"[^a-z0-9]+", text.lower()):
        if len(token) < 3 or token in STOPWORDS or token.isdigit():
            continue
        keywords.add(token)
    return sorted(keywords)


def _normalize_entities(entities: list) -> list[str]:
    normalized = set()
    for entity in entities:
        value = str(entity).strip().lower()
        if value:
            normalized.add(value)
    return sorted(normalized)


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _make_slug(activity_type: str, entities: list[str], keywords: list[str]) -> str:
    import hashlib

    tokens = [_slugify(activity_type)]
    tokens.extend(_slugify(entity) for entity in entities[:2])
    tokens.extend(_slugify(keyword) for keyword in keywords[:3])
    slug = "-".join(token for token in tokens if token)
    slug = _slugify(slug)
    if not entities and not keywords:
        base = _slugify(activity_type) or "activity"
        slug = f"{base}-pattern"
    elif not slug:
        base = _slugify(activity_type) or "activity"
        slug = f"{base}-pattern"
    if len(slug) > 80:
        digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:70].rstrip('-')}-{digest}"
    return slug


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _pattern_score(
    incoming_entities: set, incoming_keywords: set, pattern: dict
) -> float | None:
    scores = []
    pattern_entities = set(pattern.get("entities", []))
    pattern_keywords = set(pattern.get("keywords", []))

    if incoming_entities and pattern_entities:
        scores.append(_jaccard(incoming_entities, pattern_entities))
    if incoming_keywords and pattern_keywords:
        scores.append(_jaccard(incoming_keywords, pattern_keywords))
    if not scores:
        return None

    combined = sum(scores) / len(scores)
    if combined >= MATCH_THRESHOLD:
        return combined
    return None


def _find_best_match(
    activity_type: str, entities: set, keywords: set, patterns: list[dict]
) -> dict | None:
    best_match = None
    best_score = -1.0

    for pattern in patterns:
        if pattern.get("activity_type") != activity_type:
            continue
        score = _pattern_score(entities, keywords, pattern)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_match = pattern

    return best_match


def _make_observation(
    day: str,
    activity_id: str,
    description: str,
    entities: list[str],
    keywords: list[str],
) -> dict:
    return {
        "day": day,
        "activity_id": activity_id,
        "description": description,
        "entities": entities,
        "keywords": keywords,
    }


def _format_pattern_context(pattern: dict) -> str:
    observations = pattern.get("observations", [])
    first_seen = observations[0]["day"] if observations else ""
    last_seen = observations[-1]["day"] if observations else ""
    lines = [
        "## Pattern Context",
        f"Canonical slug: {pattern.get('id', '')}",
        f"Activity type: {pattern.get('activity_type', '')}",
        f"Observation count: {len(observations)}",
        f"First seen: {first_seen}",
        f"Last seen: {last_seen}",
        f"Merged entities: {', '.join(pattern.get('entities', [])) or '[none]'}",
        f"Merged keywords: {', '.join(pattern.get('keywords', [])) or '[none]'}",
        "",
        "Observations:",
    ]

    for obs in observations:
        description = obs.get("description", "") or "[no description]"
        entities = ", ".join(obs.get("entities", [])) or "[none]"
        lines.append(f"- {obs.get('day', '')}: {description}")
        lines.append(f"  entities: {entities}")

    return "\n".join(lines)


def _load_previous_outputs(facet: str, observations: list[dict]) -> str:
    if not observations:
        return "No prior agent outputs available."

    lines = ["## Prior Agent Outputs"]
    found_any = False

    for obs in observations[-3:]:
        obs_lines = []
        for key in AGENT_OUTPUT_KEYS:
            path = get_activity_output_path(facet, obs["day"], obs["activity_id"], key)
            try:
                content = path.read_text(encoding="utf-8").strip()
            except (FileNotFoundError, OSError):
                continue
            if not content:
                continue
            found_any = True
            obs_lines.append(f"### {key} ({path.name})")
            obs_lines.append(content[:2000])
            obs_lines.append("")

        if obs_lines:
            lines.append(f"Observation {obs['day']} / {obs['activity_id']}")
            lines.append(obs.get("description", "") or "[no description]")
            lines.append("")
            lines.extend(obs_lines)

    if not found_any:
        return "No prior agent outputs available."

    return "\n".join(lines).rstrip()


def _read_agency_observations() -> str:
    """Read the current ## observations section from agency.md."""
    try:
        path = Path(get_journal()) / "identity" / "agency.md"
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line == "## observations":
            start = i + 1
        elif start is not None and line.startswith("## "):
            return "\n".join(lines[start:i]).strip()
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return ""


def _skill_instruction(mode: str) -> str:
    if mode == "comparison":
        return (
            "Compare these two activity observations. Identify whether they reflect "
            "the same recurring skill or capability. Draft a provisional skill "
            "profile based on the evidence so far. Be conservative — only claim "
            "what both observations support."
        )
    if mode == "refresh":
        return (
            "Update this skill profile with new evidence from the latest "
            "observation. Preserve the core skill identity. Incorporate new "
            "details about tools, collaborators, or techniques observed. The "
            "slug must remain unchanged."
        )
    return (
        "Synthesize a complete skill profile from these recurring activity "
        "observations. You have 3+ observations of this pattern. Produce a "
        "thorough, grounded skill document that captures what the owner does, "
        "how they do it, and why."
    )


def pre_process(context: dict) -> dict | None:
    facet = context.get("facet")
    day = context.get("day")
    activity = context.get("activity")
    if not facet or not day or not activity:
        return None

    activity_type = activity.get("activity")
    activity_id = activity.get("id")
    if not activity_type or not activity_id:
        return None

    description = activity.get("description", "")
    entities = _normalize_entities(activity.get("active_entities", []))
    keywords = _extract_keywords(description)
    patterns = _load_patterns(facet)
    match = _find_best_match(activity_type, set(entities), set(keywords), patterns)
    now_iso = datetime.now(timezone.utc).isoformat()
    obs = _make_observation(day, activity_id, description, entities, keywords)

    if match is None:
        if not entities and not keywords:
            return {"skip_reason": "no signal to seed pattern"}
        pattern = {
            "id": _make_slug(activity_type, entities, keywords),
            "activity_type": activity_type,
            "keywords": keywords,
            "entities": entities,
            "observations": [obs],
            "created_at": now_iso,
            "updated_at": now_iso,
            "skill_generated": False,
        }
        patterns.append(pattern)
        _write_patterns(facet, patterns)
        return {"skip_reason": "first observation, seeded pattern"}

    match.setdefault("observations", []).append(obs)
    match["entities"] = sorted(set(match.get("entities", [])) | set(entities))
    match["keywords"] = sorted(set(match.get("keywords", [])) | set(keywords))
    match["updated_at"] = now_iso
    _write_patterns(facet, patterns)

    observation_count = len(match["observations"])
    if observation_count == 2:
        mode = "comparison"
    elif not match.get("skill_generated", False):
        mode = "generate"
    else:
        mode = "refresh"

    return {
        "template_vars": {
            "skill_instruction": _skill_instruction(mode),
            "pattern_context": _format_pattern_context(match),
            "previous_outputs": _load_previous_outputs(
                facet, match["observations"][:-1]
            ),
        },
        "meta": {
            "pattern_id": match["id"],
            "facet": facet,
            "mode": mode,
        },
    }


def post_process(result: str, context: dict) -> str | None:
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        logger.warning("skills: could not parse result as JSON")
        return None

    if not isinstance(data, dict):
        logger.warning("skills: expected JSON object result")
        return None

    meta = context.get("meta") or {}
    pattern_id = meta.get("pattern_id")
    facet = meta.get("facet")
    mode = meta.get("mode")
    if not pattern_id or not facet or not mode:
        logger.warning("skills: missing required post-process metadata")
        return None

    if mode == "comparison":
        return None

    patterns = _load_patterns(facet)
    pattern = next((item for item in patterns if item.get("id") == pattern_id), None)
    if pattern is None:
        logger.warning("skills: pattern %s not found for facet %s", pattern_id, facet)
        return None

    was_new = not pattern.get("skill_generated", False)
    pattern["skill_generated"] = True
    pattern["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_patterns(facet, patterns)

    observations = pattern.get("observations", [])
    first_seen = observations[0]["day"] if observations else ""
    last_seen = observations[-1]["day"] if observations else ""
    collaborators = sorted(
        {str(item) for item in data.get("collaborators", []) if item}
    )
    tools = sorted({str(item) for item in data.get("tools", []) if item})

    frontmatter = [
        "---",
        f'name: "{str(data.get("skill_name", pattern_id)).replace(chr(34), chr(39))}"',
        f'slug: "{pattern_id}"',
        f'category: "{str(data.get("category", "")).replace(chr(34), chr(39))}"',
        f"confidence: {data.get('confidence', 0.0)}",
        f"observations: {len(observations)}",
        f'first_seen: "{first_seen}"',
        f'last_seen: "{last_seen}"',
        "collaborators:",
    ]
    if collaborators:
        frontmatter.extend(
            f'  - "{item.replace(chr(34), chr(39))}"' for item in collaborators
        )
    else:
        frontmatter.append("  []")
    frontmatter.append("tools:")
    if tools:
        frontmatter.extend(f'  - "{item.replace(chr(34), chr(39))}"' for item in tools)
    else:
        frontmatter.append("  []")
    frontmatter.append("---")
    frontmatter.append("")
    frontmatter.append("## Description")
    frontmatter.append("")
    frontmatter.append(str(data.get("description", "")).strip())
    frontmatter.append("")
    frontmatter.append("## How")
    frontmatter.append("")
    frontmatter.append(str(data.get("how", "")).strip())
    frontmatter.append("")
    frontmatter.append("## Why")
    frontmatter.append("")
    frontmatter.append(str(data.get("why", "")).strip())
    frontmatter.append("")

    skill_path = _skills_dir(facet) / f"{pattern_id}.md"
    atomic_write(skill_path, "\n".join(frontmatter))

    if was_new:
        skill_name = data.get("skill_name", pattern_id)
        new_line = f"- Noticed recurring skill: {skill_name} — observed {len(observations)} times in {facet}"
        existing = _read_agency_observations()
        if existing and existing.strip() != "[watching and learning]":
            content = existing.rstrip("\n") + "\n" + new_line
        else:
            content = new_line
        update_identity_section(
            "agency.md",
            "observations",
            content,
            actor="agency-observations-tender",
            reason="agency observations refresh",
        )

    return None
