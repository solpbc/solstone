# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import frontmatter

from solstone.think import skills as think_skills
from solstone.think.activities import get_activity_output_path, get_activity_record
from solstone.think.identity import update_identity_section
from solstone.think.utils import get_journal

logger = logging.getLogger(__name__)

WATCHING_AND_LEARNING = "[watching and learning]"
NO_PENDING_SKILL_WORK = "no pending skill work"
SPAN_DIRNAME = "spans"


def _sort_key(value: Any) -> str:
    return str(value or "")


def _compact_json(value: Any, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _pattern_facets(pattern: dict[str, Any]) -> list[str]:
    facets = pattern.get("facets_touched")
    if isinstance(facets, list) and facets:
        return [str(item) for item in facets if item]
    derived = {
        str(obs.get("facet"))
        for obs in pattern.get("observations", [])
        if obs.get("facet")
    }
    return sorted(derived)


def _load_profile_metadata(markdown: str | None) -> dict[str, Any]:
    if not markdown:
        return {}
    try:
        post = frontmatter.loads(markdown)
    except Exception:
        logger.warning("skill_editor: failed to parse existing profile metadata")
        return {}
    meta = post.metadata
    return meta if isinstance(meta, dict) else {}


def _read_identity_section(file_name: str, heading: str) -> str:
    path = Path(get_journal()) / "identity" / Path(file_name).name
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""

    lines = text.split("\n")
    start = None
    target_heading = f"## {heading}"
    for index, line in enumerate(lines):
        if line == target_heading:
            start = index + 1
        elif start is not None and line.startswith("## "):
            return "\n".join(lines[start:index]).strip()
    if start is not None:
        return "\n".join(lines[start:]).strip()
    return ""


def _mark_edit_request_processed(request_id: str, *, error: str | None = None) -> None:
    now = think_skills.utc_now_iso()

    def mutate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if row.get("id") != request_id or row.get("processed_at") is not None:
                continue
            row["processed_at"] = now
            if error:
                row["processing_error"] = error
            break
        return rows

    think_skills.locked_modify_edit_requests(mutate)


def _build_metadata_section(
    pattern: dict[str, Any],
    profile_meta: dict[str, Any],
    *,
    request: dict[str, Any] | None,
) -> str:
    observations = pattern.get("observations", [])
    lines = [
        "## Metadata",
        f"Name: {pattern.get('name', '')}",
        f"Slug: {pattern.get('slug', '')}",
        f"Display name: {profile_meta.get('display_name', '')}",
        f"Category: {profile_meta.get('category', '')}",
        f"Confidence: {profile_meta.get('confidence', '')}",
        f"Status: {pattern.get('status', '')}",
        f"First seen: {pattern.get('first_seen', '')}",
        f"Last seen: {pattern.get('last_seen', '')}",
        f"Observation count: {len(observations)}",
        f"Facet count: {len(_pattern_facets(pattern))}",
        f"Facets touched: {', '.join(_pattern_facets(pattern))}",
        f"Created at: {pattern.get('created_at', '')}",
        f"Updated at: {pattern.get('updated_at', '')}",
        f"Profile generated at: {pattern.get('profile_generated_at', '')}",
    ]

    if request is not None:
        lines.extend(
            [
                "",
                "Last edit request:",
                f"- text: {request.get('instructions', '')}",
                f"- requested_at: {request.get('requested_at', '')}",
                f"- requested_by: {request.get('requested_by', '')}",
            ]
        )

    return "\n".join(lines)


def _build_observation_ledger(pattern: dict[str, Any]) -> str:
    lines = ["## Observation ledger"]
    observations = pattern.get("observations", [])
    if not observations:
        lines.append("[no observations]")
        return "\n".join(lines)
    lines.extend(
        _compact_json(observation)
        for observation in observations
        if isinstance(observation, dict)
    )
    return "\n".join(lines)


def _iter_recent_activity_refs(
    pattern: dict[str, Any], *, observation_limit: int
) -> list[tuple[dict[str, Any], str]]:
    refs: list[tuple[dict[str, Any], str]] = []
    for observation in pattern.get("observations", [])[-observation_limit:]:
        if not isinstance(observation, dict):
            continue
        for activity_id in observation.get("activity_ids", []) or []:
            refs.append((observation, str(activity_id)))
    return refs


def _segment_ledger(record: dict[str, Any]) -> str:
    summary = {
        "id": record.get("id"),
        "activity": record.get("activity"),
        "title": record.get("title"),
        "description": record.get("description"),
        "segments": (record.get("segments") or [])[:5],
        "active_entities": (record.get("active_entities") or [])[:5],
        "created_at": record.get("created_at"),
    }
    return _compact_json(summary, limit=400)


def _build_recent_activity_records(pattern: dict[str, Any]) -> str:
    lines = ["## Recent activity records"]
    refs = _iter_recent_activity_refs(pattern, observation_limit=5)
    if not refs:
        lines.append("[activity record not available]")
        return "\n".join(lines)

    for observation, activity_id in refs:
        facet = str(observation.get("facet") or "")
        day = str(observation.get("day") or "")
        lines.extend(
            [
                "",
                f"### {day} / {facet} / {activity_id}",
            ]
        )
        record = get_activity_record(facet, day, activity_id)
        if record is None:
            lines.append("[activity record not available]")
            continue
        lines.append(_compact_json(record, limit=600))
        lines.append(_segment_ledger(record))
    return "\n".join(lines)


def _build_recent_narratives(pattern: dict[str, Any]) -> str:
    lines = ["## Recent narratives"]
    found_any = False
    for observation, activity_id in _iter_recent_activity_refs(
        pattern, observation_limit=3
    ):
        facet = str(observation.get("facet") or "")
        day = str(observation.get("day") or "")
        path = get_activity_output_path(facet, day, activity_id, "narrative")
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if not content:
            continue
        found_any = True
        lines.extend(
            [
                "",
                f"### {day} / {facet} / {activity_id} / {path.name}",
                content[:800],
            ]
        )
    if not found_any:
        lines.append("[narrative not available]")
    return "\n".join(lines)


def _build_recent_spans(pattern: dict[str, Any]) -> str:
    lines = ["## Recent span bodies"]
    refs = pattern.get("observations", [])[-3:]
    if not refs:
        lines.append("[spans unavailable]")
        return "\n".join(lines)

    journal = Path(get_journal())
    for observation in refs:
        if not isinstance(observation, dict):
            continue
        facet = str(observation.get("facet") or "")
        day = str(observation.get("day") or "")
        activity_ids = {str(item) for item in observation.get("activity_ids", []) or []}
        lines.extend(
            ["", f"### {day} / {facet} / ids={','.join(sorted(activity_ids))}"]
        )
        span_file = journal / "facets" / facet / Path(SPAN_DIRNAME) / f"{day}.jsonl"
        if not span_file.exists():
            lines.append("[spans unavailable]")
            continue

        matched = False
        try:
            with open(span_file, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "skill_editor: malformed spans row in %s", span_file
                        )
                        continue
                    if str(row.get("span_id") or "") not in activity_ids:
                        continue
                    matched = True
                    body = str(row.get("body", "") or "")[:400]
                    talent = str(row.get("talent", "unknown") or "unknown")
                    lines.append(f"{talent}: {body}")
        except OSError:
            lines.append("[spans unavailable]")
            continue

        if not matched:
            lines.append("[no matching spans]")

    return "\n".join(lines)


def _build_skill_context(
    pattern: dict[str, Any],
    profile_meta: dict[str, Any],
    *,
    request: dict[str, Any] | None,
) -> str:
    sections = [
        _build_metadata_section(pattern, profile_meta, request=request),
        _build_observation_ledger(pattern),
        _build_recent_activity_records(pattern),
        _build_recent_narratives(pattern),
        _build_recent_spans(pattern),
    ]
    return "\n\n".join(section for section in sections if section).strip()


def _validate_updated_at(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def pre_process(context: dict) -> dict | None:
    day = context.get("day")
    if not day:
        return None

    request: dict[str, Any] | None = None
    pattern: dict[str, Any] | None = None
    mode = ""
    request_id: str | None = None
    owner_instructions = ""

    pending_requests = sorted(
        [
            row
            for row in think_skills.load_edit_requests()
            if isinstance(row, dict) and row.get("processed_at") is None
        ],
        key=lambda row: _sort_key(row.get("requested_at")),
    )
    if pending_requests:
        request = pending_requests[0]
        request_id = str(request.get("id") or "") or None
        target_slug = str(request.get("slug") or "")
        pattern = think_skills.find_pattern(target_slug)
        if pattern is None:
            if request_id:
                _mark_edit_request_processed(request_id, error="slug missing")
            logger.warning(
                "skill_editor: edit request targets missing slug %s", target_slug
            )
            return {"skip_reason": "edit-request target slug missing"}
        mode = "edit_request"
        owner_instructions = str(request.get("instructions") or "")
    else:
        patterns = think_skills.load_patterns()
        for candidate in sorted(
            patterns,
            key=lambda row: _sort_key(row.get("updated_at")),
        ):
            if not isinstance(candidate, dict) or not candidate.get("needs_profile"):
                continue
            observations = candidate.get("observations", [])
            if not observations:
                logger.warning(
                    "skill_editor: skipping zero-observation needs_profile %s",
                    candidate.get("slug"),
                )
                continue
            pattern = think_skills.find_pattern(str(candidate.get("slug") or ""))
            if pattern is None:
                logger.warning(
                    "skill_editor: missing needs_profile slug %s",
                    candidate.get("slug"),
                )
                continue
            mode = "create"
            break

        if pattern is None:
            for candidate in sorted(
                patterns,
                key=lambda row: _sort_key(row.get("updated_at")),
            ):
                if not isinstance(candidate, dict) or not candidate.get(
                    "needs_refresh"
                ):
                    continue
                observations = candidate.get("observations", [])
                if not observations:
                    logger.warning(
                        "skill_editor: skipping zero-observation needs_refresh %s",
                        candidate.get("slug"),
                    )
                    continue
                pattern = think_skills.find_pattern(str(candidate.get("slug") or ""))
                if pattern is None:
                    logger.warning(
                        "skill_editor: missing needs_refresh slug %s",
                        candidate.get("slug"),
                    )
                    continue
                mode = "refresh"
                break

    if pattern is None:
        return {"skip_reason": NO_PENDING_SKILL_WORK}

    slug = str(pattern.get("slug") or "")
    existing_profile = ""
    if mode in {"refresh", "edit_request"}:
        existing_profile = think_skills.load_profile(slug) or ""
        if mode == "refresh" and not existing_profile:
            logger.info(
                "skill_editor: refresh target %s has no profile, normalizing to create",
                slug,
            )
            mode = "create"

    profile_meta = _load_profile_metadata(existing_profile)
    skill_context = _build_skill_context(pattern, profile_meta, request=request)

    if mode == "create":
        mode_instruction = (
            "Produce a complete skill profile for this pattern. "
            "The observation evidence supports writing a grounded first version."
        )
        existing_profile = ""
    elif mode == "refresh":
        mode_instruction = (
            "Update this existing skill profile with new evidence from the most recent "
            "observations. Preserve the core skill identity and slug. Incorporate new "
            "tools, collaborators, or techniques that the evidence supports."
        )
    else:
        mode_instruction = (
            "The owner has provided specific instructions for refining this skill "
            "profile. Prioritize the instructions while staying grounded in the "
            "observation evidence."
        )

    return {
        "template_vars": {
            "slug": slug,
            "skill_mode_instruction": mode_instruction,
            "skill_context": skill_context,
            "existing_profile": existing_profile,
            "owner_instructions": owner_instructions,
        },
        "meta": {
            "slug": slug,
            "mode": mode,
            "request_id": request_id,
        },
    }


def post_process(result: str, context: dict) -> str | None:
    if not result or not result.strip().startswith("---"):
        logger.warning("skill_editor: result missing frontmatter")
        return None

    meta = context.get("meta") or {}
    slug = str(meta.get("slug") or "")
    mode = str(meta.get("mode") or "")
    request_id = meta.get("request_id")
    if not slug or mode not in {"create", "refresh", "edit_request"}:
        logger.warning("skill_editor: missing hook metadata")
        return None

    try:
        post = frontmatter.loads(result)
    except Exception:
        logger.warning("skill_editor: failed to parse profile markdown")
        return None

    data = post.metadata if isinstance(post.metadata, dict) else {}
    name = str(data.get("name") or "").strip()
    if name != slug:
        logger.warning("skill_editor: frontmatter name mismatch %s != %s", name, slug)
        return None

    display_name = str(data.get("display_name") or "").strip()
    description = str(data.get("description") or "").strip()
    category = str(data.get("category") or "").strip()
    confidence = data.get("confidence")

    if not display_name or len(display_name) > 80:
        logger.warning("skill_editor: invalid display_name")
        return None
    if not description or len(description) > 1024:
        logger.warning("skill_editor: invalid description length")
        return None
    if not category:
        logger.warning("skill_editor: missing category")
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        logger.warning("skill_editor: invalid confidence type")
        return None
    confidence_value = float(confidence)
    if confidence_value < 0.0 or confidence_value > 1.0:
        logger.warning("skill_editor: confidence out of range")
        return None

    aliases = data.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, list) or any(
            not isinstance(item, str) or not item.strip() for item in aliases
        ):
            logger.warning("skill_editor: invalid aliases")
            return None
        aliases = [item.strip() for item in aliases]

    updated_at = data.get("updated_at")
    if updated_at is not None:
        if not isinstance(updated_at, str) or not _validate_updated_at(updated_at):
            logger.warning("skill_editor: invalid updated_at")
            return None

    was_new_profile = think_skills.load_profile(slug) is None

    ordered_meta: dict[str, Any] = {
        "name": slug,
        "display_name": display_name,
        "description": description,
        "category": category,
        "confidence": confidence_value,
    }
    if aliases:
        ordered_meta["aliases"] = aliases
    if updated_at is not None:
        ordered_meta["updated_at"] = updated_at

    body = post.content.rstrip() + "\n" if post.content.strip() else ""
    markdown = frontmatter.dumps(frontmatter.Post(body, **ordered_meta))
    think_skills.save_profile(slug, markdown)

    pattern = think_skills.find_pattern(slug)
    observation_count = 0
    if pattern is not None:
        observation_count = len(pattern.get("observations", []))

    now = think_skills.utc_now_iso()

    def mutate_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target = think_skills.find_pattern(slug, rows)
        if target is None:
            logger.warning(
                "skill_editor: pattern missing during post_process: %s", slug
            )
            return rows

        changed = False
        if not target.get("profile_generated_at"):
            target["profile_generated_at"] = now
            changed = True
        if mode == "create":
            if bool(target.get("needs_profile")):
                target["needs_profile"] = False
                changed = True
            if target.get("status") == "emerging":
                target["status"] = "mature"
                changed = True
        elif mode == "refresh":
            if bool(target.get("needs_refresh")):
                target["needs_refresh"] = False
                changed = True
        else:
            if bool(target.get("needs_profile")):
                target["needs_profile"] = False
                changed = True
            if bool(target.get("needs_refresh")):
                target["needs_refresh"] = False
                changed = True
            if target.get("status") == "emerging":
                target["status"] = "mature"
                changed = True

        if changed:
            think_skills.touch_updated(target)
        return rows

    think_skills.locked_modify_patterns(mutate_patterns)

    if mode == "edit_request" and request_id:

        def mutate_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for row in rows:
                if row.get("id") != request_id or row.get("processed_at") is not None:
                    continue
                row["processed_at"] = now
                break
            return rows

        think_skills.locked_modify_edit_requests(mutate_requests)

    if mode == "create" and was_new_profile:
        nudge_line = f"- Noticed recurring skill: {display_name} — observed {observation_count} times"
        existing = _read_identity_section("agency.md", "observations")
        if nudge_line not in existing.splitlines():
            if existing and existing.strip() != WATCHING_AND_LEARNING:
                content = existing.rstrip("\n") + "\n" + nudge_line
            else:
                content = nudge_line
            update_identity_section(
                "agency.md",
                "observations",
                content,
                actor="skill_editor",
                reason="new recurring skill observed",
            )

    return None
