# Yesterday's Processing Card

## 1. Helper layout in `apps/home/routes.py`

Use the proposed top-level decomposition. It is the smallest layout that keeps template logic dumb and testable.

- `_yesterday() -> str`
  Returns yesterday in local time as `YYYYMMDD`.

- `_count_journal_age_days(today: str) -> int`
  Scans `Path(get_journal()) / "chronicle"` for `YYYYMMDD` children and returns `(today - earliest).days`. Returns `0` when `chronicle/` is missing or has no day dirs. This matches `think/utils.py:105-124`, `think/utils.py:154-208`.

- `_summarize_yesterday_processing(yesterday: str, journal_age_days: int) -> dict | None`
  Returns the full rendered card contract or `None` to hide the card.

Internal helpers called only by `_summarize_yesterday_processing`:

- `_load_yesterday_stats(yesterday: str) -> dict | None`
  Reads `chronicle/{yesterday}/stats.json`. Returns `None` on missing/invalid file.

- `_load_yesterday_pipeline_summary(yesterday: str) -> dict`
  Thin wrapper over `summarize_pipeline_day(yesterday)` from `think/pipeline_health.py:23-110`.

- `_collect_entities_yesterday(yesterday: str) -> list[dict[str, Any]]`
  Same DB access pattern as `_collect_entities_today` at `apps/home/routes.py:296+`: `try`/`finally`, `conn.close()`, graceful empty on error.
  Query `entity_signals` grouped by `entity_name`, join/fallback through `entities` identity rows the same way the existing helper does.

- `_collect_top_activities_yesterday(yesterday: str) -> list[dict[str, Any]]`
  Iterate enabled facets, call `load_activity_records(facet, yesterday)`, derive `duration_minutes` via `estimate_duration_minutes(record["segments"])`, annotate `facet`, normalize display title, and sort descending by duration.

- `_top_heatmap_hours(stats_data: dict) -> list[int]`
  Reads `stats_data["heatmap_data"]["hours"]`, keeps the top 3 non-zero hours, sorts by minutes desc then hour asc.

- `_knowledge_graph_freshness(yesterday: str) -> dict`
  Reads `chronicle/{yesterday}/talents/knowledge_graph.md`, checks existence and `st_mtime` freshness using the relaxed rule in section 4.

- `_briefing_freshness(today: str) -> dict`
  Reads `journal/sol/briefing.md` with local `frontmatter.load`. Valid only when frontmatter has `type: morning_briefing` and a parseable `generated` timestamp whose local date is `today`.

- `_newsletter_attempts_from_think_logs(yesterday: str) -> tuple[int, int]`
  Option A helper from section 3. Counts successful facet newsletters from files plus failed facet newsletter attempts from think logs.

- Formatting helpers
  `_format_duration`, `_format_hour_label`, `_format_entity_summary`, `_format_activity_label`, `_format_newsletter_summary`, `_format_processing_summary`.
  All user-facing copy, pluralization, and warning phrasing stays here, not in Jinja.

Hide conditions:

- `stats.json` missing.
- `journal_age_days == 0`.
- Stats are effectively empty: no transcript duration, no transcript segments, no facet activity, no activity records, no entity signals.

Mode resolution:

- `sparse`
  Yesterday has some transcript duration or segments, but no facet-level activity (`facet_data` empty) and no derived top activities.

- `degraded`
  Any overnight processing gap is present:
  newsletter partial count, pipeline summary not healthy, missing/invalid briefing, or stale/missing knowledge graph.

- `healthy`
  Non-sparse day with no degradation signals.

Collapse defaults:

- `sparse`: expanded.
- `degraded`: expanded.
- `healthy`: expanded for journal age days 1-7, collapsed for day 8+.

## 2. Return contract

Use one dict under the pulse context key `yesterday_processing`.

Fields:

- `title`
  `"Yesterday's processing"` or `"⚠ Yesterday's processing"`.

- `mode`
  One of `sparse`, `healthy`, `degraded`.

- `default_collapsed`
  Server-computed boolean; template only mirrors it into `data-collapsed` and `aria-expanded`.

- `first_week_framing`
  Optional single paragraph shown only when `journal_age_days <= 7` and `mode != "sparse"`.

- `summary_line`
  Always-visible collapsed summary.

- `details`
  Preformatted bullet strings for healthy/degraded modes.

- `sparse_lines`
  Exactly two preformatted strings when `mode == "sparse"`, else `None`.

- `status_reasons`
  Internal-only list of machine-readable reason tags such as `newsletter_partial`, `pipeline_warning`, `knowledge_graph_stale`, `briefing_missing`. Useful for tests and future copy changes; template does not inspect it.

Why this shape:

- It keeps the template mechanical.
- It keeps copy decisions fully in Python.
- It supports stable tests without string-parsing Jinja output.

Recommended rendered content by mode:

- `sparse`
  Summary line: light-day summary using total duration.
  Body: two lines only, no bullet list, no first-week framing.

- `healthy`
  Summary line: “I wrote N newsletters, refreshed your knowledge graph, and prepared your morning briefing.”
  Details:
  newsletter result, knowledge graph refresh, briefing generation time, top heatmap hours, top activities, top entities.

- `degraded`
  Summary line: “I wrote N of M newsletters, but some overnight processing didn’t finish.” if Option A is selected.
  If no denominator is available, summary shifts to “I wrote N newsletters, but some overnight processing didn’t finish.”
  Details:
  failure/gap bullets first, then normal detail bullets.

## 3. Q2 — degraded-day denominator semantics

### Ground truth

- `talent.fail` records include `name`, `use_id`, `state`, and optional `facet`, but `summarize_pipeline_day()` counts every failure and drops `facet` from `failed_list`. See `think/pipeline_health.py:81-99`.
- `stats.json.facet_data` is not a newsletter ledger. It is built from `events.jsonl` durations in `think/journal_stats.py:296-319` and surfaced in `apps/home/routes.py:616-621`.
- The facet newsletter writer is `sol call journal news`, implemented by `think/tools/facets.py:61-106`.
- The newsletter prompt key is stable: `facet_newsletter`.
  Reason:
  system talent config keys come from `talent/*.md` filename stems in `think/talent.py:228-235`, and the file is `talent/facet_newsletter.md:1-15`.
  Think logs emit `name=prompt_name` unchanged for dispatch and fail/complete events in `think/thinking.py:1277-1292` and `think/thinking.py:365-389`.

### Option A — re-parse think JSONL for newsletter-specific facet fails

Read `chronicle/{yesterday}/health/*_daily.jsonl` and count `talent.fail` records where:

- `event == "talent.fail"`
- `facet` is present
- `name == "facet_newsletter"`

Count successes from actual files:

- `len(list(Path(get_journal()).glob(f"facets/*/news/{yesterday}.md")))`

Formula:

- `N = successful_newsletter_files`
- `M = successful_newsletter_files + failed_facet_newsletter_attempts`

Pros:

- Faithful to the product language “wrote N of M newsletters”.
- Uses stable prompt key.
- Reads only local files already in the journal.

Cons:

- If the runtime is not currently dispatching `facet_newsletter` into daily think logs, `failed_facet_newsletter_attempts` will often be `0`.
- Non-newsletter pipeline failures still need separate degraded copy.

### Option B — re-parse any facet-scoped fail

Count every `talent.fail` with a `facet` field, regardless of `name`.

Pros:

- Captures “something facet-scoped didn’t finish”.

Cons:

- Overstates the denominator for “newsletters”.
- Would require copy shift to “facet summaries” or similar.

### Option C — drop denominator

Use only `N` from newsletter files and separate degraded copy for generic pipeline issues.

Pros:

- Never lies about `M`.

Cons:

- Gives up the specific `N of M` story.

### Option D — extend `summarize_pipeline_day`

Preserve `facet` and maybe richer classification in `failed_list`.

Pros:

- Best long-term shape.

Cons:

- Wider-than-needed change. Out of scope for this card.

### Recommendation

Pick **Option A**.

Implementation details:

- Success path reads `facets/*/news/{yesterday}.md`.
- Failure path reads `chronicle/{yesterday}/health/*_daily.jsonl`.
- Exact agent-name match: `facet_newsletter`.

Fallback behavior inside Option A:

- If `M == 0`, do not render `0 of 0`.
  Use the non-denominator sentence: “I didn’t produce any facet newsletters.”
- Generic non-newsletter pipeline failures still flip the card to `degraded`, but they appear as separate warning bullets rather than inflating `M`.

## 4. Q3 — `knowledge_graph.md` mtime false-negative edge case

Use the relaxed rule:

- Fresh when the file exists and `mtime >= start_of_yesterday_local`.

Do not require `mtime` to fall strictly within yesterday’s wall-clock day.

Rationale:

- Prep already found a real case where `chronicle/20260415/talents/knowledge_graph.md` had `mtime` on `2026-04-16 07:23:43`.
- The intent of the card is “did the overnight processing refresh yesterday’s graph?”, not “did the write finish before midnight”.
- This rule admits same-day and overnight-after-midnight completions without introducing an arbitrary 36-hour window.

Notes:

- Compare using local time boundaries.
- Use `st_mtime`, not birth time or `ctime`.

## 5. Template placement + CSS

Placement:

- Insert immediately after the closing `pulse-vitals` block in `apps/home/workspace.html`, before the welcome/briefing blocks. In current file layout that is directly after the `pulse-vitals` div shown around `apps/home/workspace.html:924-953`.

Markup:

- Outer element: `<section class="pulse-yesterday" id="pulse-yesterday" data-collapsed="...">`
- Header:
  `.pulse-yesterday-header`
  `role="button"`
  `tabindex="0"`
  `aria-expanded="..."`
- Summary row:
  `.pulse-yesterday-summary`
- Body:
  `.pulse-yesterday-body`
- Optional framing paragraph:
  `.pulse-yesterday-framing`
- Bullet list:
  `.pulse-yesterday-details`

Interaction:

- Mirror the briefing card pattern in `apps/home/workspace.html:966-1005` and `apps/home/workspace.html:1379-1392`.
- Add `toggleYesterdayCard` as a sibling to `toggleBriefingCard`, or generalize both onto a shared helper if that is a pure extraction with no behavior change.
- Do not add `pulse-yesterday` to `SECTION_IDS`; collapse state is server-driven each render.

CSS:

- Inline in `apps/home/workspace.html` with the other `.pulse-*` blocks.
- Reuse briefing-card visual structure:
  rounded card, bordered shell, hover/focus treatment on header, summary hidden when expanded, body hidden when collapsed.
- New classes only:
  `.pulse-yesterday`
  `.pulse-yesterday-header`
  `.pulse-yesterday-body`
  `.pulse-yesterday-summary`
  `.pulse-yesterday-details`
  `.pulse-yesterday-framing`
- No new global CSS files.

## 6. First-week framing

Behavior:

- Render one framing paragraph at the top of the expanded body when `journal_age_days <= 7` and `mode != "sparse"`.

Open issue:

- I could not recover the exact first-week copy from the checked-in repo or local journal/task artifacts.
- The implementation should use the scope’s verbatim wording once Jer confirms it.

Interim design assumption:

- The framing remains a single sentence or short paragraph, not a bullet.
- It is omitted entirely in sparse mode.

## 7. Refresh behavior (v1)

- No new endpoint.
- No new telemetry.
- No new app-event listener dedicated to this card.
- The card data is included in the existing `_build_pulse_context()` return and therefore in `/app/home/api/pulse`.
- Initial page render is the primary path.
- Full-page reload fallback is acceptable in v1.

This keeps the frontend change small. If later we want live card refresh, it can read from the already-extended `/app/home/api/pulse` payload without another backend change.

## 8. Tests plan

Create a new test module: `tests/test_home_yesterdays_processing.py`.

Unit tests:

- `test_yesterdays_card_hidden_when_stats_missing`
- `test_yesterdays_card_hidden_when_all_zero`
- `test_yesterdays_card_sparse_mode_copy`
- `test_yesterdays_card_healthy_collapsed_on_day_8_plus`
- `test_yesterdays_card_healthy_expanded_with_framing_on_days_1_to_7`
- `test_yesterdays_card_degraded_shows_warning_and_partial_count`
- `test_format_duration_boundaries`
- `test_entity_grouping_people_first_zero_dropped_plurals`
- `test_heatmap_peaks_top_3`
- `test_activity_bullet_title_duration_facet`
- `test_knowledge_graph_refresh_detection_yesterday_and_overnight`
- `test_briefing_frontmatter_missing_counts_as_gap`
- `test_newsletter_attempts_option_a_matches_facet_newsletter_failures_only`

Fixture plan:

- `tests/fixtures/journal/chronicle/20260415/`
  Dense day fixture with:
  `stats.json`,
  one or two `health/*_daily.jsonl` files,
  one activity file under `facets/*/activities/20260415.jsonl`,
  `agents/knowledge_graph.md`.

- `tests/fixtures/journal/chronicle/20260414/`
  Sparse day fixture with:
  `stats.json` showing transcript duration/segments but empty `facet_data`,
  no activities,
  optional empty `health/`.

- `tests/fixtures/journal/chronicle/20260416/`
  Existing empty day fixture reused for all-zero/missing cases.

Supporting non-chronicle fixture:

- `tests/fixtures/journal/sol/briefing.md`
  Valid morning-briefing frontmatter fixture for healthy cases.
  Tests that need missing/invalid frontmatter can overwrite or delete it in `tmp_path`.

Fixture minimization rule:

- Seed only the fields each test asserts on.
- Keep think logs to the minimum lines needed: `run.start`, `talent.dispatch`, `talent.complete` or `talent.fail`, `run.complete`.

## 9. Non-goals

- No “today’s processing” surface.
- No historical browser or date picker.
- No weekly rollup.
- No email digest.
- No sharing or screenshots.
- No new endpoints.
- No new telemetry or new event types.
- No changes to briefing preamble or other pulse sections.
- No addition to `SECTION_IDS`.
- No new HTTP dependencies.

## Implementation sequence

1. Add route helpers and card contract assembly in `apps/home/routes.py`.
2. Wire the new dict into `_build_pulse_context()` and `/app/home/api/pulse`.
3. Add template markup and scoped CSS in `apps/home/workspace.html`.
4. Add the card toggle helper, but no new refresh wiring.
5. Add focused tests and minimal fixtures.

## Review gate — decisions for jer

- **Q2 denominator choice:** Recommend **Option A**. Match failed newsletter attempts by exact think-log agent name `facet_newsletter`; count successes from `facets/*/news/{yesterday}.md`.
- **Q3 knowledge-graph freshness rule:** Recommend **fresh when `mtime >= start_of_yesterday_local`**. This intentionally counts overnight-after-midnight completions as fresh.
- **First-week framing copy:** Exact scope text was not recoverable from checked-in artifacts I could search. Need Jer to confirm the verbatim copy before implementation.

## Gate answers (VPE, 2026-04-17)

All three gate items resolved. Proceed to `implement` stage.

- **Q2 denominator:** Go with **Option A** as recommended. Successes from `facets/*/news/{yesterday}.md`. Failures from think-log `talent.fail` where `name == "facet_newsletter"` and `facet` is present. When current pipeline emits no `facet_newsletter` fails (which is the common case today), `M == N` and the `N of M` sentence degenerates into a simple `N` — that's fine, honest, and forward-compatible for when we start logging newsletter failures under that exact key. Use the sparse fallback "I didn't produce any facet newsletters." when both are zero.
- **Q3 knowledge-graph freshness:** Use the **relaxed rule**: fresh when `knowledge_graph.md` exists and `st_mtime >= start_of_yesterday_local`. Overnight-after-midnight completions count. Use local time boundaries. Don't use birth/ctime.
- **First-week framing copy (verbatim):** The exact copy IS in the scope (top-level note) and in the approved CPO spec. Use this text, unchanged, when `journal_age_days <= 7` and `mode != "sparse"`:

  > Most of what I learn becomes useful in the third or fourth week, when I've seen enough patterns to surface them. For now, here's what's already happening:

  Render as one `<p class="pulse-yesterday-framing">` at the top of the expanded body. Omit entirely in sparse mode.

Also confirming by reference (no change to spec): the mockup copy for the other states lives in the scope and in `cpo/specs/in-flight/yesterdays-processing-card.md`. Match those strings where they appear verbatim; don't paraphrase owner-facing language.

Proceed with the implementation sequence already in the design doc. Run `make test` before `hop processed`.
