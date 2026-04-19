# Story-Talent Refactor
This refactor replaces the old storyteller span-row write path with activity-record
story merges. It is a clean break:
- storytellers stop writing `facets/*/spans/*.jsonl`
- story content lives on the activity record itself
- `think/activities.py` remains the only writer for activity records
- priority ordering, not extra locking, serializes participation before story
## 1. New writer: `merge_story_fields`
Location:
- add `merge_story_fields(...)` in `think/activities.py`
- place it next to `update_record_fields()` and `update_activity_record()`
- this is the only activity-record write added by the refactor, so L2 stays
  satisfied inside the domain owner
Signature and docstring:
```python
def merge_story_fields(
    facet: str,
    day: str,
    record_id: str,
    *,
    story: dict,
    commitments: list[dict],
    closures: list[dict],
    decisions: list[dict],
    actor: str,
    note: str | None = None,
) -> bool:
    """Replace story-derived fields on an activity record and append one edit."""
```
Behavior and return semantics:
- use one `locked_modify(...)` call only, following the same pattern as
  `update_activity_record()` and `_set_activity_hidden_state()`
  (`think/activities.py:762-790`, `1046-1089`, `1092-1133`)
- inside the callback, find the record with the same `record.get("id") == record_id`
  match used by `update_activity_record()`
- if found:
  - normalize the current record
  - replace `story`, `commitments`, `closures`, and `decisions` wholesale
  - call `append_edit(...)` exactly once with
    `fields=["story", "commitments", "closures", "decisions"]`
  - pass through `actor`
  - pass through `note`
  - return `True`
- if the day file is missing or the record is absent:
  - log a warning
  - return `False`
  - do not raise
Why this shape:
- `update_activity_record()` is intentionally narrow and only allows
  `title`, `description`, and `details` (`think/activities.py:1046-1062`)
- the CLI mirrors that exact scope
  (`apps/activities/call.py:503-532`,
  `apps/activities/talent/activities/SKILL.md:116-130`)
- `_set_activity_hidden_state()` already establishes the specialized-writer
  pattern in this module (`think/activities.py:1092-1133`)
- `update_record_fields()` stays the generic no-edit helper used by participation
  (`think/activities.py:979-1011`, `talent/participation.py:99-105`)
## 2. `story.py` `post_process` flow
The new hook lives at `talent/story.py` and always returns `""` so the JSON
generator artifact is suppressed.
Dispatcher context shape, confirmed:
- `run_activity_prompts()` sends `facet`, `day`, `span`, `activity`, and
  `output_path` in the activity request (`think/thinking.py:2064-2083`)
- `prepare_config()` merges those request keys into the full talent config and
  always carries `name` (`think/talents.py:438-520`)
- `_run_post_hooks()` passes the full prepared config dict directly to the hook
  (`think/talents.py:712-734`)
The hook can rely on:
- `context["name"]`
- `context["facet"]`
- `context["day"]`
- `context["activity"]`
- `context["span"]`
- `context["output_path"]`
Execution order:
1. Parse `result` with `json.loads(result.strip())`.
   On failure: log and return `""`.
2. Require a top-level `dict`.
   Otherwise: log and return `""`.
3. Validate required top-level fields.
   - `body`: `str`, non-empty after strip
   - `topics`: `list[str]`, may be empty
   - `confidence`: numeric in `0.0..1.0`
   - `commitments`: `list`
   - `closures`: `list`
   - `decisions`: `list`
   Any missing field, wrong type, or out-of-range `confidence` logs and returns
   `""`.
4. Validate required context.
   - `context["activity"]` must be a `dict`
   - `context["activity"]["id"]` must exist
   - `context["facet"]` and `context["day"]` must exist
   Missing context logs and returns `""`.
5. Load entities once with:
   `load_entities(facet=context["facet"], day=context["day"])`.
6. Validate `commitments` entry by entry.
   - each entry must be a `dict`
   - required keys: `owner`, `action`, `counterparty`, `when`, `context`
   - each required value must be a `str`
   - invalid entries are skipped with a per-entry log
7. Validate `closures` entry by entry.
   - each entry must be a `dict`
   - required keys: `owner`, `action`, `counterparty`, `resolution`, `context`
   - each required value must be a `str`
   - `resolution` must be one of:
     `sent`, `done`, `signed`, `dropped`, `deferred`
   - invalid entries are skipped with a per-entry log
8. Validate `decisions` entry by entry.
   - each entry must be a `dict`
   - required keys: `owner`, `action`, `context`
   - each required value must be a `str`
   - invalid entries are skipped with a per-entry log
9. Resolve entity ids for every valid entry with
   `find_matching_entity(name, entities, fuzzy_threshold=90)`.
   - commitments: add `owner_entity_id` and `counterparty_entity_id`
   - closures: add `owner_entity_id` and `counterparty_entity_id`
   - decisions: add `owner_entity_id`
   - unmatched values become `None`
   - preserve the original `owner` and `counterparty` strings
10. Build:
    `story = {"body": body, "topics": topics, "confidence": confidence}`.
11. Extract:
    - `record_id = context["activity"]["id"]`
    - `facet = context["facet"]`
    - `day = context["day"]`
12. Call:
    `merge_story_fields(facet, day, record_id, story=..., commitments=..., closures=..., decisions=..., actor="story", note=None)`.
    If it returns `False`: log a warning and return `""`.
13. Return `""`.
    This is required because `_execute_with_tools()` only writes the output file
    when `result` is truthy (`think/talents.py:837-846`); returning `None` would
    fall back to the original JSON result (`think/talents.py:726-734`).
Intentional differences from `talent/spans.py`:
- no spans-file write
- no topic dedupe/normalization
- no confidence clamping
- no fence-stripping carryover unless explicitly added during implementation
## 3. Activity-record formatter extension
Target:
- extend `think/activities.py::format_activities()`
- current order is:
  title, activity, facet, day, time, level, description, details, participation,
  hidden (`think/activities.py:1271-1307`)
Chosen insertion point:
- add the story block after participation and before hidden
Behavior:
- if `record.get("story")` is not a `dict`, do nothing
- if `story["body"]` is a non-empty string, render it as prose rather than
  `- Story: ...`
- if `story["topics"]` is a non-empty list of strings, render one line as
  `Topics: a, b, c`
- if `body` is missing/empty, skip the prose block
- if `topics` is missing, non-list, or empty, skip the topics line
- keep all other formatter output unchanged
- keep the existing activity formatter registration; no new registry entry is
  needed because activities are already mapped to `format_activities()`
  (`think/formatters.py:143-144`)
Why this insertion point is best:
- description/details remain raw activity metadata
- participation remains the structured who-was-involved summary
- story reads naturally after those structured fields
- hidden stays last because it is record state, not content
## 4. Storyteller prompt changes
Common frontmatter changes for all three storyteller talents:
- `priority: 10` -> `priority: 20`
- `hook: {"post": "spans"}` -> `hook: {"post": "story"}`
- keep `schedule: "activity"`
- keep `output: "json"`
- keep existing activity filters per talent
Common schema changes for all three:
- require exactly:
  `body`, `topics`, `confidence`, `commitments`, `closures`, `decisions`
- all six fields are required on every response
- `topics` may be `[]`
- `commitments`, `closures`, and `decisions` may be `[]`
- add the explicit instruction:
  `Return [] if you do not observe a clear commitment / closure / decision. Better to omit than invent.`
- state the controlled closure `resolution` vocabulary exactly:
  `sent`, `done`, `signed`, `dropped`, `deferred`
`talent/conversation.md`
- keep the meeting/call/messaging/email narrative focus
- expand the schema block to the six-field JSON shape
- inline examples:
  - commitment: send a follow-up, draft, or deck by a date
  - closure: an open item was `sent` or `done`
  - decision: the group chose a direction, owner, or timing
- keep the current guidance that brief quotes are allowed when they sharpen a
  decision, commitment, or disagreement
`talent/work.md`
- keep the coding/browsing/reading progress focus
- expand the schema block to the six-field JSON shape
- inline examples:
  - commitment: ship a patch, benchmark, or send results
  - closure: a task was `done` or a review was `sent`
  - decision: a code-path, retry strategy, or API choice was made
- keep the instruction to emphasize actual work performed over UI description
`talent/event.md`
- keep the appointment/event/travel/errand outcome focus
- expand the schema block to the six-field JSON shape
- inline examples:
  - commitment: a travel or logistics follow-up
  - closure: a form was `signed`, a reservation was `done`, or a task was
    `deferred`
  - decision: a route, plan, or next-step choice was made
- keep the guidance to prefer what actually happened over generic event labels
## 5. Test matrix
| test name | file | pins |
| --- | --- | --- |
| `test_story_hook_parses_and_writes` | `tests/test_story_hook.py` | Valid JSON writes `story`, `commitments`, `closures`, `decisions` onto the activity record and appends one edit with actor `story`. |
| `test_story_hook_empty_arrays` | `tests/test_story_hook.py` | Empty `commitments`/`closures`/`decisions` still persist alongside the story payload. |
| `test_story_hook_bad_resolution_skipped` | `tests/test_story_hook.py` | Invalid closure `resolution` is dropped while valid sibling closures survive. |
| `test_story_hook_missing_required_field_skipped` | `tests/test_story_hook.py` | Missing required per-entry fields skip only the bad item. |
| `test_story_hook_resolves_entities` | `tests/test_story_hook.py` | `owner`/`counterparty` resolve to `*_entity_id` with `fuzzy_threshold=90`; misses become `None`. |
| `test_story_hook_idempotent_rerun` | `tests/test_story_hook.py` | Second run replaces story/list fields wholesale and appends one more edit entry. |
| `test_story_hook_missing_record_logs_and_returns` | `tests/test_story_hook.py` | `merge_story_fields()` returns `False`, hook logs warning, nothing raises. |
| `test_story_hook_no_json_file_written` | `tests/test_story_hook.py` | Returning `""` suppresses the storyteller JSON artifact. |
| `test_format_activities_renders_story` | `tests/test_activities.py` | Story prose and `Topics:` line appear when present and disappear cleanly when absent. |
| `test_no_spans_formatter_registered` | `tests/test_formatters.py` | `"facets/*/spans/*.jsonl"` is removed from `FORMATTERS`; spans paths no longer resolve to a formatter. |
| `test_no_spans_writes` | `tests/test_formatters.py` | Search-style assertion that no `format_spans` or `spans/` write targets remain in `think/`, `talent/`, or `apps/`. |
Existing test templates to reuse:
- `tests/test_activity_record_merge.py` for temp-journal setup, activity seeding,
  hook execution, and record reload assertions
- `tests/test_schedule_hook.py` for per-entry skip behavior and entity-resolution
  patterns
## 6. Files touched / deleted
Create:
- `docs/design/story-talent-refactor.md`
- `talent/story.py`
- `tests/test_story_hook.py`
Modify:
- `think/activities.py`
- `think/formatters.py`
- `talent/conversation.md`
- `talent/work.md`
- `talent/event.md`
- `tests/test_activities.py`
- `tests/test_formatters.py`
- `tests/baselines/api/stats/stats.json`
- `talent/journal/references/captures.md`
Delete:
- `talent/spans.py`
- `think/spans.py`
- `tests/test_spans_hook.py`
- `tests/test_spans_formatter.py`
Intentionally untouched:
- `think/thinking.py` because priority-group serialization already does the job
- `apps/activities/call.py` because no CLI surface change is needed
## 7. Risks / gotchas
- Preserve the hook-return behavior exactly: `""`, not `None`.
  `None` would fall back to the original JSON result and write a generator file
  (`think/talents.py:726-734`, `837-846`).
- Preserve the missing-record behavior of `update_record_fields()`:
  no raise, but the story path must log the failure like participation does
  (`think/activities.py:1007-1011`, `talent/participation.py:104-105`).
- `format_activities()` is already registered for
  `"facets/*/activities/*.jsonl"` (`think/formatters.py:144`).
  Do not add a new formatter entry for story data.
- Layer hygiene L2 stays strict:
  only `think/activities.py` writes the activity record.
  `talent/story.py` imports and calls the new writer; it does not perform raw
  file I/O.
- Story merges serialize after participation via priority ordering, not new locks.
  Keep participation at `10`, storytellers at `20`, and rely on the existing
  group ordering/drain in `run_activity_prompts()`
  (`think/thinking.py:1925-1928`, `2150-2183`).
