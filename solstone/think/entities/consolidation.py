# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Consolidate segment-detected entities into journal identities."""

import json
import logging
from datetime import datetime
from pathlib import Path

from solstone.think.entities.core import entity_slug, is_noise_entity
from solstone.think.entities.journal import (
    load_all_journal_entities,
    save_journal_entity,
)
from solstone.think.entities.matching import MatchTier, find_matching_entity
from solstone.think.utils import CHRONICLE_DIR, DATE_RE, now_ms

logger = logging.getLogger(__name__)


def consolidate_detected_entities(
    journal: str,
    full: bool = False,
    fuzzy_threshold: int = 85,
) -> int:
    """Consolidate segment-detected entities into journal identities."""
    journal_path = Path(journal)
    day_root = (
        journal_path / CHRONICLE_DIR
        if (journal_path / CHRONICLE_DIR).is_dir()
        else journal_path
    )
    today = datetime.now().strftime("%Y%m%d")

    segment_files = []
    for path in day_root.glob("**/talents/entities.jsonl"):
        if not path.is_file():
            continue
        try:
            day = path.relative_to(day_root).parts[0]
        except (ValueError, IndexError):
            continue
        if not DATE_RE.fullmatch(day):
            continue
        if full or day == today:
            segment_files.append(path)

    seen: dict[tuple[str, str], dict[str, str]] = {}
    for seg_file in segment_files:
        try:
            with open(seg_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping malformed JSONL in %s: %s", seg_file, e
                        )
                        continue

                    name = (data.get("name") or "").strip()
                    etype = (data.get("type") or "").strip()
                    description = (data.get("description") or "").strip()

                    if not name or not etype or is_noise_entity(name):
                        continue

                    key = (name.lower(), etype.lower())
                    if key not in seen:
                        seen[key] = {
                            "name": name,
                            "type": etype,
                            "description": description,
                        }
                    elif len(description) > len(seen[key]["description"]):
                        seen[key]["description"] = description
        except OSError as e:
            logger.warning("Skipping %s: %s", seg_file, e)

    entities_list = list(load_all_journal_entities().values())
    total_detections = len(seen)
    fuzzy_count = 0
    exact_count = 0
    new_count = 0
    ts = now_ms()

    for (name_lower, _type_lower), data in seen.items():
        name = data["name"]
        etype = data["type"]
        description = data["description"]

        match = find_matching_entity(
            name, entities_list, fuzzy_threshold=fuzzy_threshold
        )
        if match:
            if match.tier == MatchTier.FUZZY:
                fuzzy_count += 1
            else:
                exact_count += 1
            continue

        base_slug = entity_slug(name)
        if not base_slug:
            continue

        final_slug = None
        for attempt in range(1, 102):
            candidate = base_slug if attempt == 1 else f"{base_slug}_{attempt}"
            candidate_path = journal_path / "entities" / candidate / "entity.json"

            if not candidate_path.exists():
                final_slug = candidate
                break

            try:
                with open(candidate_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if (existing.get("name") or "").lower().strip() == name_lower:
                    break
            except (json.JSONDecodeError, OSError):
                continue
        else:
            logger.warning("Too many slug collisions for '%s', skipping", name)
            continue

        if final_slug is None:
            continue

        entity = {
            "id": final_slug,
            "name": name,
            "type": etype,
            "source": "detected",
            "created_at": ts,
            "updated_at": ts,
        }
        if description:
            entity["description"] = description

        try:
            save_journal_entity(entity)
            new_count += 1
        except OSError as e:
            logger.warning("Failed to write entity %s: %s", final_slug, e)

    logger.info(
        "consolidate: %d detections, %d matched-skipped (%d fuzzy, %d exact), %d new entities",
        total_detections,
        fuzzy_count + exact_count,
        fuzzy_count,
        exact_count,
        new_count,
    )
    return new_count
