# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Cross-segment speaker identity tracker.

Maintains a persistent pool of CandidateProfile objects across segments.
Each time a segment is processed, its diarized speaker cluster centroids are
compared against the existing pool:

  - Match >= MERGE_THRESHOLD  → same person, merge (weighted centroid update)
  - Match <  SPLIT_THRESHOLD  → new unknown person, create candidate
  - Between the two           → ambiguous, hold without merging

Clusters with high centroid spread (std >= STABILITY_THRESHOLD) are discarded
as unreliable before any merge/create decision.

Once a candidate accumulates enough evidence (>= 2 segments, >= 5 intervals,
>= 25s of speech) it is added to the confirmation queue.

Store lives at: {journal}/awareness/speaker_candidates.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MERGE_THRESHOLD = 0.72
SPLIT_THRESHOLD = 0.55
STABILITY_THRESHOLD = 0.25  # max mean(1 - cosine_sim_to_centroid) for a valid cluster
CONFIRM_MIN_SEGMENTS = 2
CONFIRM_MIN_INTERVALS = 5
CONFIRM_MIN_DURATION_S = 25.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CandidateProfile:
    cand_id: str
    centroid: list[float]  # 256 floats, normalized
    n_segments: int = 0
    n_intervals: int = 0
    total_duration_s: float = 0.0
    source_segments: list[dict] = field(default_factory=list)
    confirmed_entity: str | None = None
    status: str = "unconfirmed"  # unconfirmed | confirmed | rejected

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid, dtype=np.float32)

    def ready_for_confirmation(self) -> bool:
        return (
            self.status == "unconfirmed"
            and self.n_segments >= CONFIRM_MIN_SEGMENTS
            and self.n_intervals >= CONFIRM_MIN_INTERVALS
            and self.total_duration_s >= CONFIRM_MIN_DURATION_S
        )


# ---------------------------------------------------------------------------
# CandidateTracker
# ---------------------------------------------------------------------------


class CandidateTracker:
    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            from think.utils import get_journal

            store_path = Path(get_journal()) / "awareness" / "speaker_candidates.json"
        self.store_path = store_path
        self._candidates: dict[str, CandidateProfile] = {}
        self._next_id: int = 0

    # --- persistence --------------------------------------------------------

    def load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = data.get("next_id", 0)
        for raw in data.get("candidates", []):
            c = CandidateProfile(**raw)
            self._candidates[c.cand_id] = c

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "candidates": [asdict(c) for c in self._candidates.values()],
        }
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    def reset(self) -> None:
        self._candidates = {}
        self._next_id = 0
        self.store_path.unlink(missing_ok=True)

    # --- internal helpers ---------------------------------------------------

    def _new_id(self) -> str:
        cid = f"cand_{self._next_id}"
        self._next_id += 1
        return cid

    def _best_match(self, centroid: np.ndarray) -> tuple[str | None, float]:
        best_id: str | None = None
        best_score = -1.0
        for cid, cand in self._candidates.items():
            if cand.status == "rejected":
                continue
            score = float(np.dot(centroid, cand.centroid_array()))
            if score > best_score:
                best_score = score
                best_id = cid
        return best_id, best_score

    def _merge(
        self,
        cand_id: str,
        new_centroid: np.ndarray,
        n_new_intervals: int,
        duration_s: float,
        provenance: dict,
    ) -> None:
        cand = self._candidates[cand_id]
        old_arr = cand.centroid_array()
        old_w = float(cand.n_intervals)
        new_w = float(n_new_intervals)
        combined = old_arr * old_w + new_centroid * new_w
        norm = np.linalg.norm(combined)
        if norm > 1e-9:
            combined = combined / norm
        cand.centroid = combined.tolist()
        cand.n_intervals += n_new_intervals
        cand.total_duration_s += duration_s
        existing_segs = {(s["day"], s["segment"]) for s in cand.source_segments}
        if (provenance["day"], provenance["segment"]) not in existing_segs:
            cand.n_segments += 1
        cand.source_segments.append(provenance)

    def _create(
        self,
        centroid: np.ndarray,
        n_intervals: int,
        duration_s: float,
        provenance: dict,
    ) -> str:
        cid = self._new_id()
        cand = CandidateProfile(
            cand_id=cid,
            centroid=centroid.tolist(),
            n_segments=1,
            n_intervals=n_intervals,
            total_duration_s=duration_s,
            source_segments=[provenance],
        )
        self._candidates[cid] = cand
        return cid

    # --- public API ---------------------------------------------------------

    def process_segment(
        self,
        seg_dir: Path,
        source: str,
        day: str,
        segment_key: str,
    ) -> dict[str, Any]:
        """Update candidate pool from a segment's existing diarization labels.

        Reads cluster labels from {seg_dir}/{source}.jsonl and embeddings from
        {seg_dir}/{source}.npz. Skips re-diarization — uses labels already
        written by the transcription pipeline.
        """
        from think.entities import normalize_embedding

        jsonl_path = seg_dir / f"{source}.jsonl"
        npz_path = seg_dir / f"{source}.npz"

        if not jsonl_path.exists() or not npz_path.exists():
            return {"status": "missing_files"}

        try:
            npz_data = np.load(npz_path)
            embeddings = npz_data.get("embeddings")
            statement_ids = npz_data.get("statement_ids")
            durations_s = npz_data.get("durations_s")
            if embeddings is None or statement_ids is None:
                return {"status": "bad_embeddings"}
        except Exception as exc:
            logger.warning("Failed to load embeddings %s: %s", npz_path, exc)
            return {"status": "bad_embeddings"}

        sid_to_idx = {int(sid): i for i, sid in enumerate(statement_ids)}

        # Read cluster labels from JSONL (speaker field = integer cluster label)
        cluster_sids: dict[int, list[int]] = defaultdict(list)
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            for line_idx, line in enumerate(lines[1:], start=1):  # skip header
                try:
                    entry = json.loads(line)
                    lbl = entry.get("speaker")
                    if isinstance(lbl, int):
                        cluster_sids[lbl].append(line_idx)
                except json.JSONDecodeError:
                    pass
        except OSError as exc:
            logger.warning("Failed to read JSONL %s: %s", jsonl_path, exc)
            return {"status": "jsonl_read_error"}

        if not cluster_sids:
            return {"status": "no_clusters"}

        n_merges = n_new = n_ambiguous = n_unstable = 0

        for cluster_label, cluster_sid_list in cluster_sids.items():
            cluster_embs: list[np.ndarray] = []
            cluster_dur = 0.0

            for sid in cluster_sid_list:
                idx = sid_to_idx.get(sid)
                if idx is None:
                    continue
                v = normalize_embedding(embeddings[idx])
                if v is not None:
                    cluster_embs.append(v)
                    if durations_s is not None and idx < len(durations_s):
                        cluster_dur += float(durations_s[idx])

            if not cluster_embs:
                continue

            # Stability check: discard mixed-speaker clusters by measuring mean
            # cosine distance from the cluster centroid (0=tight, 1=maximally spread).
            if len(cluster_embs) >= 2:
                stacked = np.stack(cluster_embs)
                raw_centroid = np.mean(stacked, axis=0)
                c_norm = np.linalg.norm(raw_centroid)
                if c_norm > 1e-9:
                    c_unit = raw_centroid / c_norm
                    stability = float(np.mean(1.0 - stacked @ c_unit))
                    if stability >= STABILITY_THRESHOLD:
                        n_unstable += 1
                        continue

            centroid = normalize_embedding(np.mean(np.stack(cluster_embs), axis=0))
            if centroid is None:
                continue

            n_intervals = len(cluster_embs)
            provenance = {
                "day": day,
                "segment": segment_key,
                "cluster_label": cluster_label,
                "n_intervals": n_intervals,
                "duration_s": round(cluster_dur, 2),
            }

            best_id, best_score = self._best_match(centroid)

            if best_id is not None and best_score >= MERGE_THRESHOLD:
                self._merge(best_id, centroid, n_intervals, cluster_dur, provenance)
                n_merges += 1
            elif best_id is None or best_score < SPLIT_THRESHOLD:
                self._create(centroid, n_intervals, cluster_dur, provenance)
                n_new += 1
            else:
                n_ambiguous += 1

        self.save()
        return {
            "status": "ok",
            "n_clusters_found": len(cluster_sids),
            "n_merges": n_merges,
            "n_new": n_new,
            "n_ambiguous": n_ambiguous,
            "n_unstable": n_unstable,
            "total_candidates": len(self._candidates),
        }

    def get_confirmation_queue(self) -> list[CandidateProfile]:
        """Return candidates that have enough evidence to surface for confirmation."""
        return [c for c in self._candidates.values() if c.ready_for_confirmation()]

    def confirm(self, cand_id: str, entity_name: str) -> dict[str, Any]:
        """Mark a candidate as confirmed with a given entity name."""
        if cand_id not in self._candidates:
            return {"error": f"Unknown candidate {cand_id}"}
        cand = self._candidates[cand_id]
        cand.confirmed_entity = entity_name
        cand.status = "confirmed"
        self.save()
        return {"status": "confirmed", "cand_id": cand_id, "entity": entity_name}

    def retroactive_confirm(
        self,
        cand_id: str,
        entity_id: str,
        journal_root: Path | None = None,
    ) -> dict[str, Any]:
        """Backfill an entity's voiceprints from a confirmed candidate's source segments.

        For each segment the candidate appeared in, loads the JSONL + NPZ, finds
        sentences belonging to the candidate's cluster_label, and saves qualifying
        embeddings to the entity's voiceprints. Applies the same outlier rejection
        guard used by accumulate_voiceprints (_VP_OUTLIER_MIN_SIMILARITY).

        Returns a dict with n_added, n_skipped_outlier, n_skipped_missing.
        """
        from think.entities import normalize_embedding
        from think.entities.voiceprints import (
            load_entity_voiceprints_file,
            save_voiceprints_batch,
        )
        from think.utils import now_ms

        if cand_id not in self._candidates:
            return {"error": f"Unknown candidate {cand_id}"}

        cand = self._candidates[cand_id]

        if journal_root is None:
            from think.utils import get_journal

            journal_root = Path(get_journal())

        # Build existing centroid for outlier check (if enough samples exist)
        _OUTLIER_MIN_SIM = 0.18
        _OUTLIER_MIN_SAMPLES = 5
        existing_centroid: np.ndarray | None = None
        vp_result = load_entity_voiceprints_file(entity_id)
        if vp_result is not None and len(vp_result[0]) >= _OUTLIER_MIN_SAMPLES:
            existing_centroid = normalize_embedding(np.mean(vp_result[0], axis=0))

        n_added = n_skipped_outlier = n_skipped_missing = 0
        new_items: list[tuple[np.ndarray, dict]] = []

        for prov in cand.source_segments:
            day = prov["day"]
            segment_key = prov["segment"]
            cluster_label = prov["cluster_label"]

            # Locate the segment directory (try chronicle layout first, then flat)
            seg_dir: Path | None = None
            for stream in ("field.audio", "audio", "screen"):
                candidate_dir = journal_root / "chronicle" / day / stream / segment_key
                if candidate_dir.exists():
                    seg_dir = candidate_dir
                    break
                flat_dir = journal_root / day / stream / segment_key
                if flat_dir.exists():
                    seg_dir = flat_dir
                    break

            if seg_dir is None:
                n_skipped_missing += 1
                continue

            # Find matching source file
            npz_files = list(seg_dir.glob("*.npz"))
            if not npz_files:
                n_skipped_missing += 1
                continue
            npz_path = npz_files[0]
            jsonl_path = npz_path.with_suffix(".jsonl")
            if not jsonl_path.exists():
                n_skipped_missing += 1
                continue

            try:
                npz_data = np.load(npz_path)
                embeddings = npz_data.get("embeddings")
                statement_ids = npz_data.get("statement_ids")
                if embeddings is None or statement_ids is None:
                    n_skipped_missing += 1
                    continue
            except Exception:
                n_skipped_missing += 1
                continue

            sid_to_idx = {int(sid): i for i, sid in enumerate(statement_ids)}

            # Find sentences belonging to this cluster label
            try:
                lines = jsonl_path.read_text(encoding="utf-8").splitlines()
                for line_idx, line in enumerate(lines[1:], start=1):
                    try:
                        entry = json.loads(line)
                        if entry.get("speaker") != cluster_label:
                            continue
                        idx = sid_to_idx.get(line_idx)
                        if idx is None:
                            continue
                        v = normalize_embedding(embeddings[idx])
                        if v is None:
                            continue
                        if existing_centroid is not None:
                            if float(np.dot(v, existing_centroid)) < _OUTLIER_MIN_SIM:
                                n_skipped_outlier += 1
                                continue
                        source = npz_path.stem
                        new_items.append(
                            (
                                v,
                                {
                                    "day": day,
                                    "segment_key": segment_key,
                                    "source": source,
                                    "sentence_id": line_idx,
                                    "added_at": now_ms(),
                                    "method": "retroactive_confirm",
                                },
                            )
                        )
                    except json.JSONDecodeError:
                        pass
            except OSError:
                n_skipped_missing += 1
                continue

        if new_items:
            try:
                n_added = save_voiceprints_batch(entity_id, new_items)
            except Exception as exc:
                logger.warning(
                    "retroactive_confirm save failed for %s: %s", entity_id, exc
                )

        logger.info(
            "retroactive_confirm %s → %s: added=%d outlier_skip=%d missing_skip=%d",
            cand_id,
            entity_id,
            n_added,
            n_skipped_outlier,
            n_skipped_missing,
        )
        return {
            "status": "ok",
            "entity_id": entity_id,
            "n_added": n_added,
            "n_skipped_outlier": n_skipped_outlier,
            "n_skipped_missing": n_skipped_missing,
        }

    def reject(self, cand_id: str) -> dict[str, Any]:
        """Mark a candidate as rejected (bad cluster, won't resurface)."""
        if cand_id not in self._candidates:
            return {"error": f"Unknown candidate {cand_id}"}
        self._candidates[cand_id].status = "rejected"
        self.save()
        return {"status": "rejected", "cand_id": cand_id}

    def summary(self) -> dict[str, Any]:
        total = len(self._candidates)
        confirmed = sum(1 for c in self._candidates.values() if c.status == "confirmed")
        rejected = sum(1 for c in self._candidates.values() if c.status == "rejected")
        queue = len(self.get_confirmation_queue())
        return {
            "total_candidates": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "pending_confirmation": queue,
            "unconfirmed": total - confirmed - rejected,
        }
