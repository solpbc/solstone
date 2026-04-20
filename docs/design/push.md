# Wave 3 push server

## 1. Summary

Wave 3 ships a root-level push API on the existing Convey server, backed by a dedicated push runtime and APNs dispatch layer. The server surface is a new `convey/push.py` blueprint mounted at `/api/push/*`, following the same root-blueprint pattern as the Wave 2 voice server rather than adding an `apps/push/` package (`convey/voice.py:26-184`, `convey/__init__.py:150-166`, `docs/design/voice-server.md:1-37`, `0a693381 voice: ship Wave 2 voice server (root /api/voice/*, 9-tool sideband)`). The runtime starts from `convey.create_app()`, owns both a `CallosumConnection` listener and an asyncio loop, dispatches Daily Briefing when `cortex.finish` reports `name=="morning_briefing"`, and runs a 60-second periodic check for Pre-Meeting Prep across enabled facets only (`think/callosum.py:245-346`, `think/cortex.py:433-441`, `think/activities.py:877-890`, `think/facets.py:255-261`). The shipped notification categories are `SOLSTONE_DAILY_BRIEFING`, `SOLSTONE_PRE_MEETING_PREP`, and `SOLSTONE_AGENT_ALERT`; the server also defines `SOLSTONE_COMMITMENT_NUDGE` for client forward-compatibility, but no Wave 3 trigger emits it.

Wave 3 explicitly defers commitment nudges to Wave 3.1 because the current ledger surface does not provide a machine-readable due date. `LedgerItem` has `when: str | None`, no `due` field, and `age_days` is derived from `opened_at`, so any age-threshold proxy would blur “old” and “overdue” in misleading ways (`think/surfaces/types.py:16-31`, `think/surfaces/ledger.py:395-410`). Wave 3 also defers live APNs validation pending Apple Developer enrollment, and it does not cover the native iOS client implementation, which ships in the companion lode.

Wave 3 is accepted when the mocked round-trip tests pass, the new push paths remain inside their declared write ownership, and the live-validation handoff is explicit enough that enrollment unblocks the final production exercise without design work. The gate stays the same as other repo work: keep layer hygiene clean, keep behavior testable from the Flask surface downward, and make runtime startup and shutdown deterministic (`scripts/check_layer_hygiene.py:38-72`, `scripts/check_layer_hygiene.py:183-240`, `tests/test_voice_runtime.py:19-103`, `tests/test_voice_integration.py:102-149`).

## 2. Module layout

Wave 3 adds a small `think/push/` package plus one new root blueprint. The layout deliberately mirrors the voice-server split where that split is sound, and deliberately does not reuse the voice runtime name because voice shutdown still hardcodes `brain.clear_brain_state()` and voice-specific app attachment (`think/voice/runtime.py:44-50`, `think/voice/runtime.py:88-105`).

| Path | Role |
|---|---|
| `convey/push.py` | Root Flask blueprint at `/api/push/*`. Defines local request validators mirroring `convey/voice.py` (`_error`, `_required_json_object`, `_optional_json_object`) and exposes `POST /register`, `DELETE /register`, `GET /status`, and `POST /test` (`convey/voice.py:26-55`, `convey/voice.py:58-184`). |
| `convey/__init__.py` | Registers `push_bp` beside the other root blueprints and calls `start_push_runtime(app)` during `create_app()`, mirroring the existing voice wire-in. The implementation must keep using `get_journal()` at call time because `state.journal_root` is assigned only after runtime startup (`convey/__init__.py:138-166`). |
| `think/push/__init__.py` | Package marker plus narrow re-export surface for runtime helpers and the public trigger entry point. This keeps callers out of module-private helpers and matches the small public surfaces used elsewhere in `think/voice/` (`think/voice/runtime.py:121-127`). |
| `think/push/config.py` | Journal-scoped config readers for `push.apns_key_path`, `push.apns_key_id`, `push.apns_team_id`, `push.bundle_id`, and `push.environment`. It mirrors the small-reader style of `think/voice/config.py`, but intentionally does not add env-var fallback because push credentials are journal-scoped operational config, not process-scoped ambient state (`think/voice/config.py:17-42`, `think/journal_default.json:35-39`). |
| `think/push/devices.py` | Device store owner. Exposes `load_devices()`, `register_device(token, bundle_id, environment, platform)`, and `remove_device(token)`. It is the sole writer for `journal/config/push_devices.json`, normalizes token input, recovers from malformed stores by treating them as empty with a warning, and rewrites the store atomically on each mutation. |
| `think/push/dispatch.py` | APNs transport owner. Defines the four category constants, the APNs HTTP/2 client using `httpx`, the ES256 JWT signer using `PyJWT`, the 55-minute bearer-token cache, payload builders, and `send(device, payload)` / `send_many(devices, payload)` helpers. `httpx` is already in the repo dependency set, so only `PyJWT` is added (`pyproject.toml:53`). |
| `think/push/triggers.py` | Trigger owner. Defines `handle_briefing_finish(message)`, `check_pre_meeting_prep(now)`, and `send_agent_alert(title, body, context_id)`. This module is the sole writer for `journal/push/nudge_log.jsonl`; there is no separate `log.py`, so dedupe stays adjacent to trigger decisions. Trigger logic reads only enabled facets, reuses `_load_briefing_md(today)` for Daily Briefing, and reuses `load_activity_records(facet, day)` plus `record["source"] == "anticipated"` filtering for Pre-Meeting Prep (`apps/home/routes.py:149-198`, `apps/home/routes.py:305-337`, `think/activities.py:877-890`, `think/activities.py:937-939`, `think/facets.py:255-261`). |
| `think/push/runtime.py` | Dedicated runtime singleton. Exposes `start_push_runtime(app)`, `stop_push_runtime(app)`, and `stop_all_push_runtime()`. It mirrors the voice runtime’s daemon-thread + asyncio-loop + `atexit` pattern, but keeps push lifecycle independent and owns both the callosum listener and the 60-second periodic task (`think/voice/runtime.py:21-109`, `think/callosum.py:254-346`). |

Deliberate non-changes:

- No `sol push` top-level CLI. Wave 3 is a root API plus in-process runtime, matching the Wave 2 voice-server precedent rather than adding a separate command surface (`docs/design/voice-server.md:7-37`, `0a693381 voice: ship Wave 2 voice server (root /api/voice/*, 9-tool sideband)`).
- No `schedules.json` entry. The scheduler only understands `hourly`, `daily`, and `weekly`, which is too coarse for a 15-minute pre-meeting reminder (`think/scheduler.py:29-30`, `think/scheduler.py:375-438`).
- No supervisor hook. `think/supervisor.py::supervise()` is the one-second orchestration loop, but Wave 3 keeps push-domain logic out of supervisor and self-contains it in the push runtime (`think/supervisor.py:1311-1371`).

`push_devices.json` uses this storage model:

- Top-level JSON object: `{"devices": [...]}`.
- Each device row stores `token`, `bundle_id`, `environment`, `platform`, and `registered_at`.
- Token identity is unique per row. Re-registering the same token updates the row in place and refreshes `registered_at`.
- Dispatch reads all rows, then filters to rows whose `bundle_id`, `environment`, and `platform` match the current push configuration before sending.

`nudge_log.jsonl` uses this append-only model:

- One JSON object per successful trigger fire.
- Common fields: `ts`, `category`, `dedupe_key`, `sent`, `failed`.
- Category-specific context: `day` for Daily Briefing, `activity_id` and `facet` for Pre-Meeting Prep, `context_id` for Agent Alert.
- A line is appended only when at least one device send succeeds. Zero-success attempts stay retryable inside the same trigger window.

## 3. Flow diagrams

### 3.1 Device register

```text
iOS client
  -> POST /api/push/register
  -> convey/push.py validates JSON body
  -> think.push.devices.register_device(...)
  -> journal/config/push_devices.json rewrite
  -> 200 {"registered": true, "device_count": N}
```

This mirrors the voice blueprint’s “validate locally, then hand off to the feature module” pattern, but the handoff is synchronous because device registration is just a journal write and does not need the runtime loop (`convey/voice.py:29-55`, `convey/voice.py:58-123`).

### 3.2 Daily Briefing dispatch

```text
cortex subprocess
  -> think.cortex emits finish event to Callosum
  -> push runtime listener callback receives message
  -> triggers.handle_briefing_finish(message)
  -> schedule coroutine on push runtime loop
  -> poll _load_briefing_md(today) up to 10 x 1s
  -> dispatch.send_many(eligible_devices, briefing_payload)
  -> append journal/push/nudge_log.jsonl
```

The polling step is mandatory because `cortex.finish` is broadcast before `_write_output(...)` runs and before the `_active.jsonl` file is renamed to its completed name. `_load_briefing_md(today)` already enforces the `type=="morning_briefing"` and `metadata.date==today` gates, so the trigger reuses it instead of inventing a second briefing reader (`think/cortex.py:433-441`, `think/cortex.py:461-510`, `think/cortex.py:621-626`, `apps/home/routes.py:149-198`).

### 3.3 Pre-Meeting Prep dispatch

```text
push runtime periodic task (every 60s)
  -> triggers.check_pre_meeting_prep(now)
  -> get_enabled_facets().keys()
  -> load_activity_records(facet, YYYYMMDD)
  -> keep rows where source == "anticipated"
  -> keep rows where start-now is within [14m, 16m]
  -> skip rows already present in nudge_log
  -> dispatch.send_many(eligible_devices, meeting_payload)
  -> append journal/push/nudge_log.jsonl
```

The trigger must use `get_enabled_facets().keys()` so muted facets never produce push. The activity scan follows the existing repo convention: load all rows for `(facet, day)`, then filter `record["source"] == "anticipated"` in-process (`think/facets.py:255-261`, `think/activities.py:877-890`, `think/activities.py:937-939`, `apps/home/routes.py:305-337`).

### 3.4 Agent Alert dispatch

```text
in-process caller
  -> triggers.send_agent_alert(title, body, context_id)
  -> dispatch.send_many(eligible_devices, alert_payload)
  -> append journal/push/nudge_log.jsonl
```

Agent Alert is intentionally the simplest path: no callosum subscription, no scheduler, no extra persistence besides dedupe log. It is a public in-process API for future callers that want to fire a push without adding new transport plumbing.

### 3.5 Shutdown

```text
process exit or test cleanup
  -> atexit / stop_push_runtime(app)
  -> stop CallosumConnection listener
  -> cancel periodic asyncio task
  -> stop runtime loop
  -> join daemon thread
  -> clear module-level runtime state
```

The shutdown contract matches the voice runtime: cancel tracked work, stop the loop from the owning thread, join with a bounded timeout, and leave the singleton reusable for the next app instance (`think/voice/runtime.py:53-109`, `tests/test_voice_runtime.py:19-103`).

## 4. Endpoint specs

All four endpoints inherit the default Convey auth gate because `convey/root.py` wraps every request unless the endpoint is on the explicit bypass allowlist, and new `/api/push/*` routes are not on that list (`convey/root.py:81-139`). Request validation mirrors the voice helpers in `convey/voice.py`, so malformed JSON and non-object bodies fail before feature code runs (`convey/voice.py:29-55`).

### 4.1 `POST /api/push/register`

- URL: `/api/push/register`
- Method: `POST`
- Auth: default gate only. Accepts a logged-in session, Basic Auth, or the existing `trust_localhost` bypass when setup is complete and proxy headers are absent (`convey/root.py:111-139`).
- Request schema:
  - Required JSON object.
  - `device_token: string` — trimmed, non-empty. The server strips embedded spaces and lowercases before storage.
  - `bundle_id: string` — trimmed, non-empty.
  - `environment: string` — must be `"development"` or `"production"`.
  - `platform: string` — must be `"ios"` in Wave 3.
- Success response:
  - HTTP 200
  - Body: `{"registered": true, "device_count": <int>}`
- Error cases:
  - HTTP 400 `{"error": "request body must be valid JSON"}`
  - HTTP 400 `{"error": "request body must be a JSON object"}`
  - HTTP 400 `{"error": "device_token is required"}`
  - HTTP 400 `{"error": "bundle_id is required"}`
  - HTTP 400 `{"error": "environment must be development or production"}`
  - HTTP 400 `{"error": "platform must be ios"}`
  - HTTP 500 `{"error": "device registration failed"}`
- Notes:
  - Duplicate tokens upsert in place instead of adding a second row.
  - `device_count` reports stored rows after the upsert.
  - Registration does not require APNs credentials to be configured. Clients can register before the operator finishes `journal.json`.

### 4.2 `DELETE /api/push/register`

- URL: `/api/push/register`
- Method: `DELETE`
- Auth: default gate only (`convey/root.py:111-139`).
- Request schema:
  - Required JSON object.
  - `device_token: string` — trimmed, non-empty; normalized with the same rules as register.
- Success response:
  - HTTP 200
  - Body: `{"removed": true, "device_count": <int>}` when the token existed.
  - HTTP 200
  - Body: `{"removed": false, "device_count": <int>}` when the token was not present.
- Error cases:
  - HTTP 400 `{"error": "request body must be valid JSON"}`
  - HTTP 400 `{"error": "request body must be a JSON object"}`
  - HTTP 400 `{"error": "device_token is required"}`
  - HTTP 500 `{"error": "device removal failed"}`
- Notes:
  - Removing an unknown token is not an error because uninstall and token churn are expected.
  - The mutation rewrites `push_devices.json` only when the stored set actually changes.

### 4.3 `GET /api/push/status`

- URL: `/api/push/status`
- Method: `GET`
- Auth: default gate only (`convey/root.py:111-139`).
- Request schema:
  - No request body.
- Success response:
  - HTTP 200
  - Body:
    - `configured: bool`
    - `device_count: int`
    - `devices: [{token_suffix, bundle_id, platform, environment, registered_at}]`
- `configured` semantics:
  - `true` only when `push.apns_key_path`, `push.apns_key_id`, `push.apns_team_id`, and `push.bundle_id` are non-empty, `push.environment` resolves to `"development"` or `"production"`, and the configured `.p8` path is absolute, exists, and is readable.
  - `false` for any missing or invalid APNs config, including a relative or missing key file.
- `devices` semantics:
  - `token_suffix` is the last four characters of the stored token.
  - `registered_at` is an ISO-8601 timestamp string.
  - The list is sorted newest-first by `registered_at`.
- Error cases:
  - No endpoint-specific 4xx errors.
  - Malformed `push_devices.json` is treated as an empty store with a warning from `think/push/devices.py`; the route still returns HTTP 200.
- Notes:
  - This route never returns full device tokens.
  - This route reports all stored devices, not just devices matching the active bundle/environment filter.

### 4.4 `POST /api/push/test`

- URL: `/api/push/test`
- Method: `POST`
- Auth: default gate only. There is no extra debug header or test-only bypass (`convey/root.py:111-139`).
- Request schema:
  - Empty body allowed.
  - If a body is present, it must decode to a JSON object.
  - Optional `title: string`
  - Optional `body: string`
  - Optional `category: string`
- Category rules:
  - Default: `SOLSTONE_AGENT_ALERT`
  - Allowed explicit values: `SOLSTONE_DAILY_BRIEFING`, `SOLSTONE_PRE_MEETING_PREP`, `SOLSTONE_AGENT_ALERT`, `SOLSTONE_COMMITMENT_NUDGE`
  - The route validates the category value but still dispatches through `send_agent_alert(...)`, so Wave 3 test sends always use the Agent Alert payload shape.
- Success response:
  - HTTP 200
  - Body: `{"sent": <int>, "failed": <int>}`
- Error cases:
  - HTTP 400 `{"error": "request body must be valid JSON"}`
  - HTTP 400 `{"error": "request body must be a JSON object"}`
  - HTTP 400 `{"error": "category must be a known push category"}`
  - HTTP 503 `{"error": "push not configured"}`
- Notes:
  - The route dispatches through `send_agent_alert(...)`, so successful test sends append `SOLSTONE_AGENT_ALERT` lines to `nudge_log.jsonl` with unique `push-test-<uuid>` context ids.
  - The route sends only to stored devices whose `bundle_id`, `environment`, and `platform` match the current push configuration.
  - Empty device store is a normal case and returns `{"sent": 0, "failed": 0}`.

## 5. Config keys

Push configuration lives only in `journal/config/journal.json`. Unlike voice, it does not fall back to environment variables, because these values describe the current journal’s push environment rather than a process-global secret cache (`think/voice/config.py:30-42`). Every read uses `get_config()` and every path lookup uses `get_journal()` at call time; the runtime does not cache the journal root during startup because `create_app()` does not assign `state.journal_root` until after `start_voice_runtime(app)` today, and Wave 3 preserves that ordering when it adds `start_push_runtime(app)` (`convey/__init__.py:163-166`, `think/voice/runtime.py:44-46`).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `push.apns_key_path` | `string \| null` | `null` | Absolute path to the APNs `.p8` signing key file. |
| `push.apns_key_id` | `string \| null` | `null` | Apple-issued APNs Key ID. |
| `push.apns_team_id` | `string \| null` | `null` | Apple Developer Team ID. |
| `push.bundle_id` | `string \| null` | `null` | App bundle identifier, for example `org.solpbc.solstone-swift`. |
| `push.environment` | `"development" \| "production" \| null` | `"development"` | APNs environment. `null` resolves to `"development"`. |

Literal `think/journal_default.json` block:

```json
  "voice": {
    "openai_api_key": null,
    "model": "gpt-realtime",
    "brain_model": "haiku"
  },
  "push": {
    "apns_key_path": null,
    "apns_key_id": null,
    "apns_team_id": null,
    "bundle_id": null,
    "environment": "development"
  },
  "retention": {
    "raw_media": "days",
    "raw_media_days": 7,
    "per_stream": {},
    "storage_warning_disk_percent": 80,
    "storage_warning_raw_media_gb": null
  }
```

Validation rules:

- Blank strings normalize to `null`.
- `push.environment == null` resolves to `"development"`.
- Any non-null environment outside `{"development", "production"}` is invalid and makes dispatch unavailable until corrected.
- `push.apns_key_path` must be an absolute path. Relative paths are rejected because the server already has one journal-root concept and should not invent a second config root.
- `push.apns_key_path` must point to a readable file before the server reports push as configured.

## 6. Payload shapes per category

Common transport rules:

- `dispatch.send(...)` uses `httpx.AsyncClient(http2=True)` against `https://api.sandbox.push.apple.com` for `"development"` and `https://api.push.apple.com` for `"production"`.
- Authorization is APNs bearer JWT signed with ES256 over the configured `.p8` key, using header `{"alg":"ES256","kid":<apns_key_id>}` and claims `{"iss":<apns_team_id>,"iat":<now>}`.
- The JWT is cached in memory and regenerated when older than 55 minutes. Wave 3 does not mint a fresh token per request.
- `apns-topic` is always the configured `push.bundle_id`.
- `apns-priority` is `10` for every Wave 3 send because every shipped payload includes a visible alert. `5` remains reserved for future silent-only pushes and is not used in Wave 3.
- `BadDeviceToken` and `Unregistered` responses from APNs cause `dispatch.send(...)` to call `devices.remove_device(token)` before returning failure.

Category constants defined by the server:

- `SOLSTONE_DAILY_BRIEFING`
- `SOLSTONE_PRE_MEETING_PREP`
- `SOLSTONE_AGENT_ALERT`
- `SOLSTONE_COMMITMENT_NUDGE`

### 6.1 Daily Briefing

Headers:

- `apns-topic: <bundle_id>`
- `apns-collapse-id: briefing.<YYYYMMDD>`
- `apns-priority: 10`

Payload:

```json
{
  "aps": {
    "alert": {
      "title": "Daily Briefing",
      "body": "Your briefing is ready — tap to view"
    },
    "category": "SOLSTONE_DAILY_BRIEFING",
    "sound": "default",
    "mutable-content": 1,
    "content-available": 1
  },
  "data": {
    "action": "open_briefing",
    "day": "20260419",
    "generated": "2026-04-19T06:45:00",
    "needs_attention_count": 3
  }
}
```

Builder rules:

- `day` is the local journal day in `YYYYMMDD` format.
- `generated` comes from briefing frontmatter when present, because `_load_briefing_md(today)` already loads that metadata (`apps/home/routes.py:149-198`, `tests/fixtures/journal/identity/briefing.md:1-14`).
- `needs_attention_count` is the length of the bullets list returned by `_load_briefing_md(today)` (`apps/home/routes.py:191-198`).

### 6.2 Pre-Meeting Prep

Headers:

- `apns-topic: <bundle_id>`
- `apns-collapse-id: meeting.<activity_id>`
- `apns-priority: 10`

Payload:

```json
{
  "aps": {
    "alert": {
      "title": "Pre-Meeting Prep",
      "body": "Meeting in 15 minutes — tap to view"
    },
    "category": "SOLSTONE_PRE_MEETING_PREP",
    "sound": "default",
    "mutable-content": 1,
    "content-available": 1,
    "interruption-level": "time-sensitive"
  },
  "data": {
    "action": "open_pre_meeting",
    "activity_id": "anticipated_meeting_090000_0420",
    "facet": "work",
    "day": "20260420",
    "start": "09:00",
    "title": "Launch sync",
    "location": "Room A",
    "participants": [
      "Juliet Capulet"
    ],
    "prep_notes": "Bring launch notes"
  }
}
```

Builder rules:

- `activity_id` comes from the activity record’s `id`, which already exists on anticipated rows and is the right natural key for collapse and dedupe (`tests/test_voice_tools.py:197-211`, `think/activities.py:893-915`).
- `start` accepts stored `HH:MM` or `HH:MM:SS`; the trigger parser is tolerant, but the payload preserves the stored string as-is.
- `participants` pulls attendee names from `participation` entries where `role=="attendee"`, following the existing Home surface pattern (`apps/home/routes.py:312-337`, `tests/test_voice_tools.py:197-214`).
- `interruption-level` is present only on this category.

### 6.3 Agent Alert

Headers:

- `apns-topic: <bundle_id>`
- `apns-collapse-id: alert.<context_id|uuid>`
- `apns-priority: 10`

Payload:

```json
{
  "aps": {
    "alert": {
      "title": "Agent Alert",
      "body": "A workflow needs attention"
    },
    "category": "SOLSTONE_AGENT_ALERT",
    "sound": "default",
    "mutable-content": 1,
    "content-available": 1
  },
  "data": {
    "action": "open_alert",
    "context_id": "triage-20260419-001"
  }
}
```

Builder rules:

- `title` and `body` come from the caller.
- `context_id` is required for the public `send_agent_alert(...)` helper and is used directly in both `data.context_id` and the collapse id.
- `POST /api/push/test` generates a UUID context when the client does not supply one through the route body.

### 6.4 Deferred Commitment Nudge

Headers:

- `apns-topic: <bundle_id>`
- `apns-collapse-id: commitment.<ledger_id>`
- `apns-priority: 10`

Forward-compatible payload:

```json
{
  "aps": {
    "alert": {
      "title": "Commitment Nudge",
      "body": "A commitment needs attention — tap to view"
    },
    "category": "SOLSTONE_COMMITMENT_NUDGE",
    "sound": "default",
    "mutable-content": 1,
    "content-available": 1
  },
  "data": {
    "action": "open_commitment",
    "ledger_id": "lg_123"
  }
}
```

Wave 3 defines this payload shape and constant only for client forward-compatibility. No trigger emits it until Wave 3.1 lands a real due-date primitive.

### 6.5 PII fallback rule

Wave 3 hardcodes the lock-screen-safe fallback into the payload builders. It is not a TODO for the iOS client.

- Daily Briefing body is always generic: “Your briefing is ready — tap to view.”
- Pre-Meeting Prep body is always generic: “Meeting in 15 minutes — tap to view.”
- Deferred Commitment Nudge body is always generic: “A commitment needs attention — tap to view.”
- Sensitive detail lives in `data`, where the Notification Service Extension can read it off-device and decide what to reveal.
- Agent Alert is the exception: `title` and `body` are caller-provided, so the caller is responsible for keeping them lock-screen safe.

## 7. Domain write-ownership (L1–L9 declarations)

Push owns its own journal domain. It is not an indexer, importer, scheduler, or search subsystem, so its writes belong in `think/push/*` and do not need to be routed through another domain owner (`scripts/check_layer_hygiene.py:38-57`, `scripts/check_layer_hygiene.py:199-209`).

| Path | Owner module | Write API | Read API |
|---|---|---|---|
| `journal/config/push_devices.json` | `think/push/devices.py` | `register_device`, `remove_device` | `load_devices` |
| `journal/push/nudge_log.jsonl` | `think/push/triggers.py` | `_append_nudge_log` | `_has_nudged` |

L1 declaration:

- `think/push/*` is a feature-owned runtime and transport layer, not infrastructure. Its journal writes are feature state, not accidental cross-domain mutation.

L2 declaration:

- No module outside `think/push/` writes `journal/config/push_devices.json`.
- No module outside `think/push/` writes `journal/push/nudge_log.jsonl`.
- `convey/push.py` calls feature APIs but never writes these paths directly.

L3 declaration:

- `load_devices()` never writes. Malformed-store recovery is explicit: the reader returns an empty list and leaves rewrite responsibility to the next write path.
- `_has_nudged(...)` never writes. Dedupe is read-first, then write on success.
- There is no create-on-miss hidden behind any `load_*` or `get_*` helper.

L6/L7 declaration:

- Indexers and importers do not touch push paths.
- The current hygiene script only scans infrastructure scopes `think/indexer`, `think/importers`, `think/search`, and `think/graph`, plus read-verb `apps/*/call.py` handlers; `think/push/*` is outside those scopes (`scripts/check_layer_hygiene.py:38-44`, `scripts/check_layer_hygiene.py:124-145`, `scripts/check_layer_hygiene.py:156-180`).
- The current hygiene script also only looks for writes near `journal/entities`, `journal/facets`, and `journal/observations`, not `journal/config/push_devices.json` or `journal/push/nudge_log.jsonl`, so no allowlist entry is required (`scripts/check_layer_hygiene.py:59-72`, `scripts/check_layer_hygiene.py:105-108`).

L8 declaration:

- Hooks do not apply. Push has no talent hook and does not write through `think/hooks.py` or `talent/*.py`.

L9 declaration:

- Daily Briefing dedupe key: `(SOLSTONE_DAILY_BRIEFING, YYYYMMDD)`
- Pre-Meeting Prep dedupe key: `(SOLSTONE_PRE_MEETING_PREP, activity_id, YYYYMMDD)`
- Agent Alert dedupe key: `(SOLSTONE_AGENT_ALERT, context_id)`
- Commitment Nudge reserved dedupe key: `(SOLSTONE_COMMITMENT_NUDGE, ledger_id)`
- Trigger order is always:
  - compute dedupe key
  - check `_has_nudged(...)`
  - send to eligible devices
  - append log only when `sent > 0`
- That write-after-success rule keeps Pre-Meeting Prep retryable if the first attempt produces zero successful sends, while still preventing duplicate delivery after one device has already received the notification.

`nudge_log.jsonl` line shape:

- Daily Briefing line: `ts`, `category`, `dedupe_key`, `day`, `sent`, `failed`
- Pre-Meeting Prep line: `ts`, `category`, `dedupe_key`, `day`, `facet`, `activity_id`, `sent`, `failed`
- Agent Alert line: `ts`, `category`, `dedupe_key`, `context_id`, `sent`, `failed`

## 8. Tests

All push tests use `tests/fixtures/journal/` plus `monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", ...)` where necessary, following the existing voice integration and route test setup pattern (`tests/test_voice_routes.py:14-23`, `tests/test_voice_integration.py:102-149`).

### 8.1 `tests/test_push_config.py`

Purpose: mirror the small-reader coverage shape of `tests/test_voice_config.py` while locking down the journal-only push config contract (`tests/test_voice_config.py:9-43`).

- Defaults: all four required APNs fields resolve to `None`, and `push.environment` resolves to `"development"` when omitted.
- Whitespace cleanup: blank strings normalize to `None`.
- Journal precedence: populated `journal.json` values are used directly.
- No env fallback: setting unrelated env vars does not change push config reads.
- Invalid environment: non-`development` / non-`production` values raise a validation error or mark config unavailable, depending on the caller.
- Missing key path: status/configured becomes false when the configured `.p8` path does not exist.

### 8.2 `tests/test_push_devices.py`

Purpose: cover the device store as its own write-owning module.

- Register/load/remove round trip against a temporary journal.
- Duplicate token registration updates the existing row instead of creating a second device.
- `registered_at` refreshes on duplicate registration.
- Remove returns `False` for unknown token and leaves the count unchanged.
- Empty store returns `[]` and does not create the file.
- Malformed `push_devices.json` is treated as empty with a warning, then repaired by the next successful write.
- Status-shaping helper masks tokens to last four only.

### 8.3 `tests/test_push_dispatch.py`

Purpose: unit-test APNs transport, JWT handling, payload builders, and token-redaction rules.

- JWT signing shape: header includes `alg=ES256` and `kid=<apns_key_id>`, claims include `iss=<apns_team_id>` and `iat=<now>`.
- JWT caching: second send inside 55 minutes reuses the cached token.
- JWT refresh: send at 60+ minutes mints a fresh token.
- Payload shape for Daily Briefing matches §6 exactly.
- Payload shape for Pre-Meeting Prep includes `interruption-level: time-sensitive`.
- Payload shape for Agent Alert matches §6 exactly.
- Commitment payload constant and builder exist but are not wired to a trigger.
- `apns-collapse-id` is `briefing.<day>`, `meeting.<activity_id>`, `alert.<context_id>`, and `commitment.<ledger_id>`.
- `BadDeviceToken` response calls `devices.remove_device(token)`.
- `Unregistered` response calls `devices.remove_device(token)`.
- Captured logs never include full tokens, JWTs, or raw `.p8` contents.

### 8.4 `tests/test_push_triggers.py`

Purpose: lock down idempotency, timing, and source filtering in the trigger layer.

- Daily Briefing finish polls until `briefing.md` exists, using a delayed write to simulate the `cortex.finish` before-write ordering.
- Daily Briefing gives up after 10 polls and logs a warning.
- Daily Briefing fires once per day even if the same finish message is delivered twice.
- Daily Briefing uses `_load_briefing_md(today)` and ignores stale or wrong-type briefing files.
- Pre-Meeting Prep scans only `get_enabled_facets().keys()` and skips muted facets.
- Pre-Meeting Prep filters `record["source"] == "anticipated"` and ignores all other activity rows.
- Pre-Meeting Prep detects the 15-minute window with both `HH:MM` and `HH:MM:SS` starts.
- Pre-Meeting Prep is idempotent across repeated periodic ticks within the same 2-minute window.
- `send_agent_alert(...)` builds the expected payload and appends the expected dedupe log line.
- Zero-device and zero-success sends do not append dedupe log lines.

### 8.5 `tests/test_push_routes.py`

Purpose: mirror the validation-heavy style of `tests/test_voice_routes.py` at the new `/api/push/*` surface (`tests/test_voice_routes.py:26-118`).

- `POST /api/push/register` happy path.
- `POST /api/push/register` rejects non-object JSON.
- `POST /api/push/register` rejects missing fields and invalid `environment` / `platform`.
- `DELETE /api/push/register` happy path for an existing token.
- `DELETE /api/push/register` returns `removed: false` for a missing token.
- `GET /api/push/status` returns `configured`, `device_count`, and masked device rows.
- `POST /api/push/test` returns `503` when APNs config is missing.
- `POST /api/push/test` aggregates sent/failed counts from the dispatch layer.
- `POST /api/push/test` rejects unknown category strings.

### 8.6 `tests/test_push_integration.py`

Purpose: prove the whole runtime path from Flask startup through callosum listener and periodic dispatch, following the same end-to-end philosophy as `tests/test_voice_integration.py` (`tests/test_voice_integration.py:115-149`).

- Boot `create_app()` against a fixture journal and verify `start_push_runtime(app)` runs.
- Patch `httpx.AsyncClient.post` so APNs sends are fully mocked.
- Patch JWT signing inputs so tests can assert stable Authorization headers.
- Fire a fake `cortex.finish` event through the push runtime’s `CallosumConnection` callback and assert Daily Briefing dispatch occurs.
- Confirm `nudge_log.jsonl` receives the expected Daily Briefing line.
- Seed anticipated activities and run one periodic `check_pre_meeting_prep(now)` pass through the runtime loop.
- Confirm muted facets do not dispatch.
- Stop the runtime cleanly and assert the loop and thread are cleared.

## 9. Security considerations

- Device-token redaction: full tokens are never returned by `GET /api/push/status` and never written to logs. Status exposes last four characters only, and dispatch logs use the same redaction rule.
- APNs JWT secrecy: bearer JWT values are never logged. Refresh decisions may log age or cache-hit state, but not the token string itself.
- `.p8` key secrecy: `push.apns_key_path` may appear in operator config, but the file contents themselves are never logged or echoed back through routes.
- PII fallback: Daily Briefing, Pre-Meeting Prep, and deferred Commitment Nudge use generic lock-screen bodies, with detail moved into `data` for device-side handling. Agent Alert is caller-provided and inherits caller responsibility for lock-screen safety.
- Auth: all `/api/push/*` endpoints use the default root auth gate. There is no debug header and no push-specific bypass. Default auth is session cookie, then Basic Auth, then opt-in `trust_localhost` after setup when proxy headers are absent (`convey/root.py:49-57`, `convey/root.py:81-139`).
- `trust_localhost` stays narrow by design: it only applies after setup completion and only when `request.remote_addr` is local and proxy headers are absent (`convey/root.py:119-139`).
- Facet eligibility: all push triggers operate on `get_enabled_facets().keys()`, so muted facets are excluded from both dispatch and device-visible summaries (`think/facets.py:255-261`, `think/surfaces/ledger.py:454-456`).
- Terminology covenant: operator-visible strings and payload labels use the repo’s “observer/listen” vocabulary and avoid “capture”, “record”, “keeper”, or “assistant”.
- Hosted-MVP privacy stance: payloads are cleartext to APNs and the device; Wave 3 is explicitly non-E2E.
- No analytics: Wave 3 adds no tracking, analytics beacons, crash reporting, or delivery pixel equivalents.

## 10. Live validation

**Live APNs validation is deferred pending Apple Developer enrollment.** Wave 3 ships infrastructure and mocked tests only. When enrollment completes:

1. Configure `push.apns_key_path`, `push.apns_key_id`, `push.apns_team_id`, `push.bundle_id`, and `push.environment` in `journal.json`.
2. Register a real device from the iOS client.
3. Exercise `POST /api/push/test` against the development APNs environment and confirm the device receives the notification.
4. Manually trigger a `morning_briefing` cortex run and confirm the Daily Briefing push lands.
5. Wait for a scheduled meeting and confirm Pre-Meeting Prep lands within approximately ±30 seconds of T-15:00.
6. Only then flip `push.environment` to `production` and deploy.

Sandbox smoke-test commands:

```sh
BASE_URL=${BASE_URL:-http://127.0.0.1:5015}
AUTH=${AUTH:-":$SOL_PASSWORD"}
TOKEN=${TOKEN:-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef}

curl -u "$AUTH" \
  -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/api/push/register" \
  -d '{
    "device_token": "'"$TOKEN"'",
    "bundle_id": "org.solpbc.solstone-swift",
    "environment": "development",
    "platform": "ios"
  }'

curl -u "$AUTH" \
  "$BASE_URL/api/push/status"

curl -u "$AUTH" \
  -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/api/push/test" \
  -d '{
    "title": "Push test",
    "body": "This is a sandbox test notification.",
    "category": "SOLSTONE_AGENT_ALERT"
  }'

curl -u "$AUTH" \
  -H 'Content-Type: application/json' \
  -X DELETE "$BASE_URL/api/push/register" \
  -d '{
    "device_token": "'"$TOKEN"'"
  }'
```

Basic Auth uses only the password component, so `-u ":$SOL_PASSWORD"` is the portable curl form for these routes (`convey/root.py:49-57`).

## 11. Open questions

- Agent Alert body limits: Wave 3 should probably enforce a soft cap before the native client ships, but the exact truncation policy can wait until the iOS notification UI settles.
- Multi-build device identity: Wave 3 keys stored devices by token alone. If one journal starts registering multiple app builds that share a token namespace, revisit whether identity should widen to `(token, bundle_id, environment)`.
- Retry telemetry: Wave 3 records dedupe state in `nudge_log.jsonl`, but it does not yet record APNs failure reasons in a separate operator-facing history file.

## 12. Sources

Voice-server analog:

- `docs/design/voice-server.md:1-465`, `convey/voice.py:26-184`, `convey/__init__.py:112-166`, `think/voice/runtime.py:21-109`, `think/voice/config.py:17-42`, `think/voice/sideband.py:20-61`, `tests/test_voice_config.py:9-43`, `tests/test_voice_runtime.py:19-103`, `tests/test_voice_routes.py:26-118`, `tests/test_voice_integration.py:102-149`

Callosum / cortex finish timing:

- `think/callosum.py:245-346`, `think/cortex.py:433-441`, `think/cortex.py:461-510`, `think/cortex.py:621-626`, `apps/home/routes.py:149-198`, `apps/home/workspace.html:1468-1470`, `apps/home/workspace.html:1732-1788`, `convey/bridge.py:45-86`, `apps/home/events.py:21-55`

Ledger / activities APIs:

- `think/surfaces/types.py:16-31`, `think/surfaces/ledger.py:395-487`, `think/activities.py:877-945`, `apps/home/routes.py:305-337`, `think/facets.py:255-261`, `tests/test_voice_tools.py:197-214`, `tests/fixtures/journal/identity/briefing.md:1-14`

Auth model:

- `convey/root.py:49-57`, `convey/root.py:81-139`

Scheduler / runtime choice:

- `think/scheduler.py:29-30`, `think/scheduler.py:375-438`, `think/supervisor.py:1311-1371`, `think/heartbeat.py:45-138`

Layer-hygiene script:

- `scripts/check_layer_hygiene.py:38-72`, `scripts/check_layer_hygiene.py:105-108`, `scripts/check_layer_hygiene.py:124-145`, `scripts/check_layer_hygiene.py:156-180`, `scripts/check_layer_hygiene.py:199-220`

Config and dependencies:

- `think/journal_default.json:35-39`, `pyproject.toml:53`

Wave 2 commit:

- `0a693381 voice: ship Wave 2 voice server (root /api/voice/*, 9-tool sideband)`
