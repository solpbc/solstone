# 260425 owner bootstrap UX collapse and manual-tag accelerant

## 1. Summary

This lode keeps `solstone/apps/speakers/owner.py:233-476` as the HDBSCAN-only owner-detection path and adds a sibling manual rebuild path driven by explicit owner attestations. The manual path does not trust `voiceprints.npz` alone: it walks the principal's voiceprint rows, cross-checks each row against the segment's `talents/speaker_labels.json`, accepts only `method in {user_assigned, user_corrected, user_confirmed}` with `speaker == principal_id`, then reloads the embedding and duration from the segment NPZ. The manual path reuses the same three quality gates and writes `owner_centroid.npz` with the exact current schema: `centroid`, `cluster_size`, `threshold`, `version`. Owner-contamination guarding gains an in-memory provisional centroid cache only; there is no new journal file and no wipe-target change.

## 2. Files touched

| Path | Planned change |
|---|---|
| `solstone/apps/speakers/encoder_config.py` | Add locked constant `OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS = 5` at the end of the locked block, with a one-line rationale comment. |
| `solstone/apps/speakers/owner.py` | Add manual-tag collection, shared quality-gate helper, provisional-guard cache helpers, and the sibling manual build function. Factor centroid persistence so the manual and confirmed paths write the same NPZ schema. |
| `solstone/apps/speakers/routes.py` | Add `POST /api/owner/build-from-tags`, switch contamination reads to confirmed-or-provisional owner guard data, and enrich owner-status payloads for the collapsed banner. |
| `solstone/apps/speakers/workspace.html` | Collapse owner states into one banner shell, add expandable diagnostics plus manual-build CTA, and refresh owner status after principal manual attestations. |
| `solstone/apps/speakers/status.py` | Mirror the richer owner-state fields in the status surface so CLI/admin diagnostics stay aligned with the UI. |
| `solstone/apps/speakers/tests/conftest.py` | Clear the provisional owner-cache between tests so route- and owner-level cache assertions stay isolated. |
| `solstone/apps/speakers/tests/test_encoder_config.py` | Assert the new locked constant and owner import contract. |
| `solstone/apps/speakers/tests/test_owner.py` | Cover manual-tag collection, shared gate behavior, manual-build success/low-quality cases, status payload branches, and schema parity with `confirm_owner_candidate()`. |
| `solstone/apps/speakers/tests/test_routes.py` | Cover provisional contamination on manual attribution routes and the new build-from-tags endpoint behavior. |
| `solstone/apps/speakers/tests/test_status.py` | Cover the expanded owner diagnostic fields. |

## 3. Decisions

### 3.1 API shape

Decision: add a new endpoint, `POST /app/speakers/api/owner/build-from-tags`.

Why: `detect_owner_candidate()` is currently the HDBSCAN entry point, and keeping that contract intact preserves both the implementation boundary and the current tests. A query/body switch on `/api/owner/detect` would mix two different acquisition strategies into one route, add invalid state combinations, and make the status flow harder to reason about. A separate endpoint also gives the owner-banner CTA a clear target and keeps action logging explicit (`owner_voiceprint_build_from_tags` versus auto-detect). This route can return either the canonical `low_quality` payload with `source: "manual_tags"` or the canonical `confirmed` payload. The new route is easy to gate, test, and remove independently if the manual accelerant changes later.

Files touched: `solstone/apps/speakers/routes.py`, `solstone/apps/speakers/owner.py`, `solstone/apps/speakers/tests/test_owner.py`, `solstone/apps/speakers/tests/test_routes.py`.

### 3.2 Function placement in `owner.py`

Decision: add a sibling function, `bootstrap_owner_from_manual_tags()`, rather than branching inside `detect_owner_candidate()`.

Why: the manual path diverges immediately from the HDBSCAN path. It is principal-specific, begins from persisted voiceprint rows, cross-checks labels, and never clusters. Forcing both flows through `detect_owner_candidate(source=...)` would create a misleading API and make the function harder to test and maintain. The two paths only need to share small private helpers: `_collect_manual_tag_embeddings(...)`, `_apply_owner_quality_gates(cluster_embeddings: np.ndarray, durations: list[float], segment_count: int, embeddings_count: int, source: str) -> dict | None`, and a centroid writer/state updater. The gate helper returns the canonical low-quality dict, including `source`, or `None` when all three gates pass. This keeps the public mental model simple: `detect_owner_candidate()` finds a candidate; `bootstrap_owner_from_manual_tags()` promotes sufficiently attested manual tags into a confirmed centroid.

Files touched: `solstone/apps/speakers/owner.py`, `solstone/apps/speakers/routes.py`, `solstone/apps/speakers/tests/test_owner.py`.

### 3.3 Manual-tag set materialization

Decision: use `_collect_manual_tag_embeddings(principal_id) -> tuple[np.ndarray, list[dict[str, Any]]]` in `solstone/apps/speakers/owner.py`, backed by label cross-reference rather than voiceprint metadata alone.

Why: the prep findings rule out trusting `voiceprints.npz` as “owner-attested.” The helper should load the principal's `voiceprints.npz`, dedupe rows by `(day, segment_key, source, sentence_id)`, resolve the segment directory from `stream` when present, and fall back to a `chronicle/<day>/*/<segment_key>` scan for historical rows that lack `stream`. For each row it should load `talents/speaker_labels.json`, confirm `speaker == principal_id` and `method in {user_assigned, user_corrected, user_confirmed}`, then reload the segment NPZ through `_load_embeddings_file()` to recover the exact embedding and `durations_s`. If `durations_s` is missing, it should reuse `_fallback_statement_durations()` so the manual path behaves like the HDBSCAN path on legacy segments. On success, `bootstrap_owner_from_manual_tags()` writes the same `owner_centroid.npz` schema as `confirm_owner_candidate()` and reuses `cluster_size` for the validated manual-tag count so downstream readers do not need a parallel schema. The returned provenance rows should carry `day`, `stream`, `segment_key`, `source`, `sentence_id`, and `duration_s`, which is enough for diagnostics and for the shared quality-gate helper.

Files touched: `solstone/apps/speakers/owner.py`, `solstone/apps/speakers/tests/test_owner.py`.

### 3.4 Provisional contamination strategy

Decision: add a module-level in-memory provisional guard cache in `solstone/apps/speakers/owner.py`, keyed by the principal voiceprints file `mtime_ns` plus the validated manual-tag count.

Why: `_check_owner_contamination` is cold enough that an in-memory cache is sufficient and much simpler than a persisted artifact. Add `OWNER_BOOTSTRAP_PROVISIONAL_GUARD_MIN_TAGS = 5` in `encoder_config.py`; when no confirmed `owner_centroid.npz` exists and at least five validated manual owner tags are available, compute a normalized mean centroid and cache `(mtime_ns, manual_tag_count, centroid, threshold)`. When a confirmed centroid exists, clear the provisional cache and always use the on-disk confirmed vector. Also clear the cache when the principal disappears, `voiceprints.npz` disappears, or the validated manual-tag count drops below the minimum. No file is written, nothing is added to `wipe.py`, and process-restart loss is acceptable because rebuild cost is low and the call surface is narrow.

Files touched: `solstone/apps/speakers/encoder_config.py`, `solstone/apps/speakers/owner.py`, `solstone/apps/speakers/routes.py`, `solstone/apps/speakers/tests/test_encoder_config.py`, `solstone/apps/speakers/tests/test_routes.py`.

### 3.5 Diagnostics affordance in `workspace.html`

Decision: use an expand toggle inside the owner banner, not a title tooltip.

Why: the low-signal state now needs more than a one-line explanation. The owner banner should render one shared panel shell with summary copy, an optional “Build from tagged samples” button, and a toggle such as “Why not yet?” that reveals a compact diagnostics block. The expanded block should show the gate name, observed value, threshold value, source (`hdbscan` or `manual_tags`), and three counters: validated manual tags, segments with audio, and embeddings available. `api_owner_status` should expose those fields on the low-signal branches together with `can_build_from_tags`, so the client does not need to recompute eligibility. This is richer than a `title=` attribute, keeps the information accessible on mobile, and matches the existing inline status styling pattern already used in the workspace. The collapsed state stays lightweight while still giving jer the detailed affordance needed for debugging or support.

Files touched: `solstone/apps/speakers/workspace.html`, `solstone/apps/speakers/routes.py`, `solstone/apps/speakers/tests/test_owner.py`.

### 3.6 Voiceprint metadata discriminator

Decision: do pure labels cross-reference; do not add a `method` field to voiceprint metadata.

Why: `speaker_labels.json` is already the source of truth for manual versus automatic attribution, while `voiceprints.npz` is intentionally a generic embedding store shared by manual review, bootstrap, discovery, attribution accumulation, and merge flows. Adding `method` only to `_save_voiceprint()` would create a partial schema that says nothing about rows written by `bootstrap.py`, `discovery.py`, or `attribution.py`, which is worse than having no provenance flag at all. Skipping the schema change also avoids migrating historical journals and keeps merge/idempotency behavior unchanged. If the product later needs first-class voiceprint provenance, that should be a separate full-schema lode that updates every writer and reader together.

Files touched: `solstone/apps/speakers/owner.py` only.

### 3.7 Status re-fetch trigger on `confirmAttribution`

Decision: use existing client data, not a new server response field, and centralize the refresh logic in `workspace.html`.

Why: `api_review()` already tells the client whether a sentence belongs to the principal (`sentence.is_owner`) and whether an entity is principal (`all_entities[].is_principal`). `confirmAttribution()` does not change speaker identity, so it can decide locally whether the success path just produced an owner attestation and call `checkOwnerStatus()` when true. The same small helper should be reused by `correctAttribution()` and `assignAttribution()` when the chosen entity is principal, because `loadReview()` does not itself hit `/api/owner/status`. This avoids unnecessary route churn, keeps the principal lookup on the client where the data already exists, and ensures the owner banner updates immediately after any principal manual tag is created.

Files touched: `solstone/apps/speakers/workspace.html`.

## 4. Implementation sequence

1. Add the locked constant and the new `owner.py` helpers: `_collect_manual_tag_embeddings(...)`, `_apply_owner_quality_gates(...)`, provisional guard loader/cache, and shared centroid persistence/state update.
2. Add the new route and status-payload fields in `solstone/apps/speakers/routes.py`, then mirror the owner diagnostics in `solstone/apps/speakers/status.py`.
3. Refactor `workspace.html` owner rendering into one banner shell, add the diagnostics toggle and manual-build CTA, and add principal-attestation status refresh hooks.
4. Add/adjust tests in `test_owner.py`, `test_routes.py`, `test_discovery.py`, `test_status.py`, and `test_encoder_config.py`.

## 5. Risks and open questions

- Historical principal voiceprint rows without `stream` metadata will require a segment-directory fallback scan. The design treats missing or ambiguous rows as non-qualifying and skips them rather than guessing.
- The provisional guard minimum (`5`) is intentionally lower than the full confirmation gate (`30`). The UI copy must make it clear that provisional contamination protection and confirmed centroid promotion are different thresholds.
- `solstone/apps/speakers/owner.py:541-608` currently writes awareness with direct `update_state(...)` calls, not a dedicated state helper. Implementation can either keep that style or add a tiny private wrapper, but both the manual and confirmed paths must record `status: "confirmed"` consistently and persist the same NPZ schema.
- HDBSCAN and manual paths can race on `owner_centroid.npz`; both produce gate-passing centroids, so correctness is unaffected. First write wins; the second is a no-op via the short-circuit in `bootstrap_owner_from_manual_tags()`.
- Reverse contamination — a user mistakenly Assigning their own identity to a non-owner sentence — is unguarded by design. Out of scope for this lode.

## 6. Audit-time correction

During audit, the provisional contamination cache was found to invalidate only on `voiceprints.npz` mtime. Because correcting a principal manual label away does not rewrite `voiceprints.npz`, a stale cached centroid could keep blocking non-owner saves. Fixed by keying the cache on `(mtime_ns, validated_manual_tag_count)` and adding a regression at `solstone/apps/speakers/tests/test_routes.py` for the label-only removal path.

## 7. Handoff items

- **Spec roll-forward.** `cpo/specs/shipped/speaker-attribution-wespeaker.md` lives outside this worktree. The shipped-lodes row referencing this lode (id `4kverujw`, branch `hopper-4kverujw-owner-bootstrap-ux`) needs to be added on the CPO side from this decision record.
- **Visual states.** Sandbox capture was skipped — a venv-level ONNX runtime issue made `make sandbox` painful during implement. Screenshots of the four owner-banner states (no embeddings, manual progress, HDBSCAN candidate, confirmed/hidden) should be captured opportunistically post-merge.
- **Test reason-string consolidation.** A handful of older HDBSCAN/status assertions in `test_owner.py` still hardcode `low_quality_reason` strings rather than importing the centralized constants. Cleanup-only — not blocking.
