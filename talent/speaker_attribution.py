# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speaker attribution talent hook — orchestrates the 4-layer pipeline.

pre_process:  Runs Layers 1-3 (computational).  If all sentences are
              resolved, writes speaker_labels.json and skips the LLM.
              Otherwise, injects unmatched-sentence context for Layer 4.

post_process: Merges Layer 4 LLM results with Layers 1-3, writes the
              final speaker_labels.json, and runs voiceprint accumulation.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def pre_process(context: dict) -> dict | None:
    """Run Layers 1-3 and decide whether Layer 4 (LLM) is needed."""
    from apps.speakers.attribution import (
        accumulate_voiceprints,
        attribute_segment,
        save_speaker_labels,
    )
    from think.utils import segment_path

    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream") or ""

    if not day or not segment:
        return {"skip_reason": "no_segment_context"}

    result = attribute_segment(day, stream, segment)
    seg_dir = segment_path(day, segment, stream)

    if result.get("error"):
        logger.info("Attribution skipped: %s", result["error"])
        reason = result["error"]
        if any(seg_dir.glob("*.npz")):
            agents_dir = seg_dir / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            out_path = agents_dir / "speaker_labels.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"labels": [], "skipped": True, "reason": reason},
                    fh,
                    indent=2,
                )
            logger.info("Wrote attribution stub: %s (%s)", out_path, reason)
        return {"skip_reason": reason}

    labels = result.get("labels", [])
    if not labels:
        reason = "no_embeddings"
        if any(seg_dir.glob("*.npz")):
            agents_dir = seg_dir / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            out_path = agents_dir / "speaker_labels.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"labels": [], "skipped": True, "reason": reason},
                    fh,
                    indent=2,
                )
            logger.info("Wrote attribution stub: %s (%s)", out_path, reason)
        return {"skip_reason": reason}

    unmatched = result.get("unmatched", [])
    metadata = result.get("metadata", {})
    source = result.get("source")

    # Stash Layer 1-3 results for the post-hook
    meta = dict(context.get("meta") or {})
    meta["attribution_result"] = result

    if not unmatched:
        # All sentences resolved — write output and skip the LLM
        save_speaker_labels(seg_dir, labels, metadata)

        # Voiceprint accumulation
        if source:
            try:
                saved = accumulate_voiceprints(day, stream, segment, labels, source)
                if saved:
                    logger.info("Accumulated voiceprints: %s", saved)
            except Exception as exc:
                logger.warning("Voiceprint accumulation failed: %s", exc)

        return {"skip_reason": "all_resolved", "meta": meta}

    # Layer 4 needed — inject context for the LLM
    unmatched_texts = result.get("unmatched_texts", {})
    candidates = result.get("candidates", [])

    lines = [
        "## Speaker Attribution — Layer 4 Analysis Required",
        "",
        "Layers 1-3 (owner detection, structural heuristics, acoustic matching)",
        "resolved most sentences.  The following need contextual identification.",
        "",
    ]

    if candidates:
        lines.append(f"**Known speakers in this segment:** {', '.join(candidates)}")
        lines.append("")

    lines.append("**Unmatched sentences:**")
    lines.append("")
    for sid in unmatched:
        text = unmatched_texts.get(sid, "[text unavailable]")
        lines.append(f'- Sentence {sid}: "{text}"')
    lines.append("")

    unmatched_block = "\n".join(lines)

    return {"meta": meta, "template_vars": {"unmatched_context": unmatched_block}}


def post_process(result: str, context: dict) -> str | None:
    """Merge Layer 4 LLM results with Layers 1-3 and write speaker_labels.json."""
    from apps.speakers.attribution import (
        accumulate_voiceprints,
        save_speaker_labels,
    )
    from think.entities import find_matching_entity
    from think.entities.journal import load_all_journal_entities
    from think.utils import segment_path

    day = context.get("day")
    segment = context.get("segment")
    stream = context.get("stream") or ""
    meta = context.get("meta") or {}
    attribution_result = meta.get("attribution_result")

    if not attribution_result or not day or not segment:
        return None

    labels = attribution_result.get("labels", [])
    metadata = attribution_result.get("metadata", {})
    source = attribution_result.get("source")

    # Parse LLM Layer 4 attributions
    layer4: dict[int, dict] = {}
    if result:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = parsed.get("attributions", parsed.get("labels", []))
            else:
                items = []

            journal_entities = load_all_journal_entities()
            entities_list = [
                e for e in journal_entities.values() if not e.get("blocked")
            ]

            for item in items:
                if not isinstance(item, dict):
                    continue
                sid = item.get("sentence_id")
                speaker_name = item.get("speaker")
                if sid is None or not speaker_name:
                    continue

                entity = find_matching_entity(speaker_name, entities_list)
                if entity:
                    layer4[int(sid)] = {
                        "speaker": entity["id"],
                        "confidence": "medium",
                        "method": "contextual",
                    }
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse Layer 4 result: %s", exc)

    # Merge: Layer 4 fills in unmatched sentences
    for label in labels:
        if label["speaker"] is None:
            l4 = layer4.get(label["sentence_id"])
            if l4:
                label["speaker"] = l4["speaker"]
                label["confidence"] = l4["confidence"]
                label["method"] = l4["method"]

    # Write final speaker_labels.json
    seg_dir = segment_path(day, segment, stream)
    save_speaker_labels(seg_dir, labels, metadata)

    # Voiceprint accumulation
    if source:
        try:
            saved = accumulate_voiceprints(day, stream, segment, labels, source)
            if saved:
                logger.info("Accumulated voiceprints: %s", saved)
        except Exception as exc:
            logger.warning("Voiceprint accumulation failed: %s", exc)

    return None  # Don't modify the generator's own output
