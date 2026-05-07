# Wave 4 observer actions

## 1. Summary

Wave 4 adds a second per-session voice side-effect queue beside nav hints: observer actions. The new queue is purpose-built for structured action payloads, not string hints, and it extends the established Wave 2 tool-dispatch pattern rather than refactoring it. `observer.start_listening` stops being a pure stub and starts returning an internal `_observer_action` sentinel; `dispatch_tool_call(...)` strips that sentinel from the model-facing JSON, enqueues a structured action under the voice session `call_id`, and leaves the existing sideband/runtime flow unchanged. A new root voice route, `GET /api/voice/observer-actions`, lets the iOS client poll for queued actions and drain them with intentionally lenient semantics: missing, blank, or unknown `call_id` returns HTTP 200 with `{"actions": [], "consumed": true}` so polling can stay robust on a simple cadence even if the client temporarily loses the echoed call id (`solstone/think/voice/tools.py:686-739`, `solstone/convey/voice.py:160-166`, `solstone/think/voice/sideband.py:20-37`, `solstone/think/voice/nav_queue.py:17-84`).

## 2. Module layout

New files:

- `solstone/think/voice/observer_queue.py` — thread-safe per-`call_id` FIFO for observer action payloads. Mirrors `solstone/think/voice/nav_queue.py` mechanically, but stores dict payloads instead of strings, owns its own TTL/capacity constants, and exports a module-level singleton accessor.
- `tests/test_voice_observer_queue.py` — unit coverage for TTL expiry, FIFO capacity, drain-clears semantics, malformed enqueue rejection, and basic thread-safety, following the pattern in `tests/test_voice_nav_queue.py`.

Existing files changed:

- `solstone/think/voice/tools.py` — keep the `observer.start_listening` manifest entry and valid modes, but change `handle_observer_start_listening(...)` from stub-ack to sentinel-emitting return data; extend `dispatch_tool_call(...)` so it strips `_observer_action` and enqueues the structured payload keyed by the session `call_id` it already receives (`solstone/think/voice/tools.py:121-135`, `solstone/think/voice/tools.py:686-739`).
- `solstone/convey/voice.py` — add `GET /api/voice/observer-actions` beside `GET /api/voice/nav-hints`, with intentionally lenient `call_id` semantics and drain-on-read behavior (`solstone/convey/voice.py:26-30`, `solstone/convey/voice.py:160-166`).
- `tests/test_voice_routes.py` — add route coverage for the new endpoint.
- `tests/test_voice_tools.py` — update the observer handler happy-path assertion and add dispatch-enqueue coverage parallel to the existing nav-target stripping test.
- `tests/test_voice_integration.py` — add a fake-Realtime round trip that drives `observer.start_listening`, verifies the stripped public tool output, and drains the observer action queue through the new route.

Deliberate non-changes:

- No `solstone/think/voice/nav_queue.py` refactor into a shared generic base. The payload type, validation semantics, and current scale do not justify generification.
- No `solstone/think/voice/sideband.py` or `solstone/think/voice/runtime.py` changes. The existing sideband loop already routes every tool call through `dispatch_tool_call(...)`, which remains the single place where per-session queue side effects happen (`solstone/think/voice/sideband.py:20-37`).
- No `solstone/apps/observer/routes.py` changes. The existing ingest endpoint is already compatible with the planned iOS multipart upload shape; this design only documents that compatibility and the filename-stem caveat (`solstone/apps/observer/routes.py:503-643`).

Public surface for the new queue module:

```python
@dataclass(frozen=True)
class QueuedAction:
    payload: dict[str, Any]
    created_at: float


class ObserverActionQueue:
    def __init__(
        self,
        *,
        ttl_seconds: int = 60,
        capacity: int = 8,
    ) -> None: ...

    def push(
        self,
        call_id: str,
        action: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None: ...

    def drain(
        self,
        call_id: str,
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]]: ...

    def clear(self) -> None: ...


def get_observer_queue() -> ObserverActionQueue: ...


@voice_bp.get("/observer-actions")
def observer_actions():
    ...
```

## 3. Flow diagram

```text
OpenAI Realtime tool call
  -> think.voice.sideband._sideband_loop(...)
  -> think.voice.tools.dispatch_tool_call(name, arguments, call_id, app)
  -> handle_observer_start_listening(payload, app)
  -> handler returns:
       {
         "status": "requested",
         "mode": "meeting" | "voice_memo",
         "note": "sol will start listening shortly",
         "_observer_action": {
           "type": "start_observer",
           "mode": ...
         }
       }
  -> dispatch_tool_call(...) strips "_observer_action"
  -> dispatch_tool_call(...) enqueues action in ObserverActionQueue under session call_id
  -> stripped JSON is posted back to Realtime as function_call_output

iOS client
  -> observes tool-call completion on the data channel
  -> polls GET /api/voice/observer-actions?call_id=<session-call-id>
  -> route drains queued actions for that call_id
  -> route returns {"actions": [...], "consumed": true}
  -> client applies {"type": "start_observer", "mode": "..."}
  -> client starts observer capture and uploads multipart media to /app/observer/ingest
```

## 4. Endpoint spec

### `GET /api/voice/observer-actions`

Request:

- Method: `GET`
- Path: `/api/voice/observer-actions`
- Query params:
  - `call_id: string` — optional in practice. The route trims it if present, but blank, missing, or unknown values all resolve to an empty successful response.

Success response:

- HTTP 200
- Body: `{"actions": [...], "consumed": true}`

Response behavior:

- Missing `call_id` returns `{"actions": [], "consumed": true}` with HTTP 200.
- Blank `call_id` returns `{"actions": [], "consumed": true}` with HTTP 200.
- Unknown `call_id` returns `{"actions": [], "consumed": true}` with HTTP 200.
- Empty queue for a known `call_id` returns `{"actions": [], "consumed": true}` with HTTP 200.
- Non-empty queue returns FIFO-ordered actions, drains the queue entry, and still reports `consumed: true`.

Side effects:

- Drains and clears the queue for the provided `call_id`.
- Drops expired actions before returning.

Intentional divergence from nav hints:

- `GET /api/voice/nav-hints` currently rejects missing `call_id` with HTTP 400 (`solstone/convey/voice.py:160-166`).
- `GET /api/voice/observer-actions` is intentionally lenient and always returns HTTP 200 with an array payload, because the iOS polling loop is simple cadence-based infrastructure, not in-web-session UI state. Robust empty success responses are a better failure mode than hard 400s when the client temporarily misses or delays the echoed `call_id`.

## 5. Action shape and handler return shape

Internal handler return shape from `handle_observer_start_listening(...)`:

```json
{
  "status": "requested",
  "mode": "meeting",
  "note": "sol will start listening shortly",
  "_observer_action": {
    "type": "start_observer",
    "mode": "meeting"
  }
}
```

Public tool output after `dispatch_tool_call(...)` strips the sentinel:

```json
{
  "status": "requested",
  "mode": "meeting",
  "note": "sol will start listening shortly"
}
```

Queued observer action payload:

```json
{
  "type": "start_observer",
  "mode": "meeting"
}
```

Rules:

- Valid modes remain `meeting` and `voice_memo`, matching the existing manifest contract (`solstone/think/voice/tools.py:121-135`).
- `dispatch_tool_call(...)` remains the only place that turns an internal handler sentinel into a per-session side effect, matching the existing `_nav_target` pattern (`solstone/think/voice/tools.py:736-739`).
- Handler signature stays uniform: `(payload, app) -> dict[str, Any]`.

## 6. Queue semantics

`ObserverActionQueue` intentionally matches the proven shape of `NavHintQueue` while keeping its own module and payload typing (`solstone/think/voice/nav_queue.py:21-84`):

- Per-`call_id` queue stored in a lock-protected `defaultdict(deque)`.
- TTL: 60 seconds.
- Capacity: 8 actions per `call_id`.
- FIFO ordering.
- Oldest item dropped on overflow.
- Expired items dropped on both `push(...)` and `drain(...)`.
- `drain(...)` returns all currently valid actions in order and clears the queue entry.
- `clear()` wipes all queue state for tests and explicit cleanup.
- Blank `call_id` is rejected as a no-op.
- Blank action payload is rejected as a no-op.
- Rejected enqueue attempts should log a warning so malformed producers are diagnosable without surfacing an HTTP failure.

Validation scope:

- The queue owns only generic enqueue sanity: blank `call_id` and empty payload rejection.
- The producer (`handle_observer_start_listening(...)` plus `dispatch_tool_call(...)`) owns the concrete Wave 4 action contract: `{"type": "start_observer", "mode": "meeting"|"voice_memo"}`.

## 7. Polling cadence guidance

Observer-actions polling should mirror nav-hints polling:

- The client should begin or continue polling after each tool-call event it observes on the Realtime data channel.
- Default poll interval: 500ms.
- The client should stop polling when the voice session ends.
- Because the route is lenient, empty 200 responses are part of the normal cadence rather than a client error condition.
- The client should use the same session `call_id` it passed to `POST /api/voice/connect`, because that is the key `dispatch_tool_call(...)` uses for queueing. The route is not keyed by the per-tool `event.call_id` echoed back in the `function_call_output` envelope (`solstone/convey/voice.py:119-127`, `solstone/think/voice/sideband.py:24-35`).

## 8. Failure semantics

- Tool JSON decode errors, unknown tools, and handler exceptions remain inside the existing `dispatch_tool_call(...)` wrapper and return the existing inline error payloads; no observer action is enqueued in those cases (`solstone/think/voice/tools.py:720-739`).
- Invalid listen modes still return `{"error": "invalid mode"}` and do not enqueue an action (`solstone/think/voice/tools.py:690-698`).
- Queue overflow drops the oldest queued action for that `call_id`, logs a warning, and keeps processing the new action.
- TTL expiry drops stale queued actions without turning them into route or tool failures.
- Blank enqueue inputs are no-ops with warning logs, not raised exceptions.
- The route itself has no endpoint-specific 4xx failure path; all missing, blank, unknown, or empty queue states collapse to HTTP 200 with an empty `actions` array.
- The queue is in-memory only, matching nav hints. Process restart or runtime teardown drops queued observer actions; that is acceptable for this adjunct path because the client polls immediately and the action is cheap to reissue on a later tool turn.

## 9. Observer-ingest compatibility note

Compatibility verdict:

- No ingest code changes are required for Wave 4.
- The existing observer ingest surfaces already accept the planned iOS multipart upload shape:
  - Primary path: `POST /app/observer/ingest` with `Authorization: Bearer <key>`
  - Legacy fallback: `POST /app/observer/ingest/<key>`
- Required fields: `segment=HHMMSS_LEN`, `day=YYYYMMDD`, and one or more `files`.
- Optional fields: `host`, `platform`, and `meta` (JSON object encoded as a form field).
- The ingest layer is binary-format-agnostic: it reads uploaded file bytes, sanitizes filenames, skips empty uploads, and writes the files to disk, but does not validate MIME type, codec, sample rate, or channel layout (`solstone/apps/observer/routes.py:309-367`, `solstone/apps/observer/routes.py:503-585`).

Stream resolution:

- If `meta["stream"]` is present and matches the stream-name regex, the server trusts it.
- Otherwise the server derives the stream from the observer registration name via `stream_name(observer=observer_name)`.
- `meta.stream` is therefore optional. The client only needs to send it when it wants a qualified stream name that should not be normalized from the observer registration name (`solstone/apps/observer/routes.py:589-600`).
- The existing regression coverage for qualified stream preservation is in `solstone/apps/observer/tests/test_routes.py:1654-1694`.

Host / platform semantics:

- `host` and `platform` are metadata only.
- If both form fields and `meta` contain those values, the `meta` dict wins.
- Hostname mismatch against the registered observer name logs a warning but does not reject the upload (`solstone/apps/observer/routes.py:547-567`, `solstone/apps/observer/routes.py:627-643`).
- The client should send a deterministic `host` so observer records remain coherent across sessions.

Filename-stem constraint:

- The ingest layer does not enforce a filename stem, but downstream consumers do care.
- The client should upload `audio.m4a`, not an arbitrary stem like `meeting.m4a`, so transcription writes `audio.jsonl` and existing readers keep finding the transcript under their current glob conventions.
- The transcribe pipeline writes the transcript as a sibling JSONL using the raw file stem, so `audio.m4a` becomes `audio.jsonl` (`solstone/observe/transcribe/main.py:165`, `solstone/observe/transcribe/main.py:594-609`).
- Transcript/cluster consumers currently glob for `*audio.jsonl`, `audio.jsonl`, or `*_audio.jsonl`, not arbitrary stems (`solstone/think/cluster.py:132-136`, `solstone/think/cluster.py:397-405`, `solstone/apps/transcripts/routes.py:256-258`, `solstone/think/retention.py:73-100`).
- `.m4a` itself is supported end-to-end: the media registry includes it, the sensor registers all `AUDIO_EXTENSIONS`, and transcribe accepts `.m4a` as a supported raw input (`solstone/think/media.py:6-25`, `solstone/observe/sense.py:606-667`, `solstone/observe/sense.py:1096-1100`, `solstone/observe/transcribe/main.py:58`, `solstone/observe/transcribe/main.py:885-889`).

Operational caveats:

- Duplicate submissions short-circuit with `status="duplicate"` and do not emit `observe.observing` (`solstone/apps/observer/routes.py:369-395`, `solstone/apps/observer/routes.py:604-605`, `solstone/apps/observer/tests/test_routes.py:1276-1368`).
- Segment collisions can rewrite the final segment key; the server returns the adjusted `segment` in the response body and records `segment_original` in history (`solstone/apps/observer/routes.py:406-425`, `solstone/apps/observer/routes.py:470-500`, `solstone/apps/observer/tests/test_routes.py:1552-1586`).
- If the client ever needs server-truth correlation after upload, it should trust the response body’s `segment`, not assume the requested key survived unchanged.

## 10. Tests

New file: `tests/test_voice_observer_queue.py`

- Unknown `call_id` drains to `[]`.
- `drain(...)` clears the queue entry.
- Expired actions are dropped on drain.
- Capacity enforces FIFO oldest-drop behavior.
- Blank `call_id` and blank payload enqueue attempts are ignored.
- Basic thread-safety sanity for concurrent `push(...)` followed by `drain(...)`.

Route updates: `tests/test_voice_routes.py`

- Missing `call_id` returns HTTP 200 with `{"actions": [], "consumed": true}`.
- Unknown `call_id` returns the same empty successful shape.
- Pre-populated queue drains actions through `GET /api/voice/observer-actions`.
- A second drain on the same `call_id` returns an empty successful shape.

Tool updates: `tests/test_voice_tools.py`

- `test_observer_start_listening_happy` changes from stub ack to sentinel-bearing internal handler return shape.
- `test_observer_start_listening_failure` stays functionally the same unless valid modes change.
- Add a parallel dispatch test, for example `test_dispatch_tool_call_strips_observer_action`, that asserts:
  - the returned JSON omits `_observer_action`
  - the public payload is `{"status": "requested", "mode": ..., "note": ...}`
  - the observer queue drains one structured action for the session `call_id`
- Keep the existing nav-target stripping test so both queue side-effect channels stay covered.

Integration updates: `tests/test_voice_integration.py`

- Add a `_FakeConn` scripted-events case that emits `observer.start_listening`.
- Assert the tool output returned to the fake OpenAI conversation omits `_observer_action`.
- Poll `GET /api/voice/observer-actions?call_id=...` and assert the queued action is returned once and then cleared.
- Keep the current journal/nav-hint round trip as-is; the observer-action case is a sibling integration scenario, not a replacement.

Expected non-change:

- `tests/test_voice_sideband.py` should not need structural changes because `dispatch_tool_call(...)` keeps the same signature and remains the only queueing seam the sideband loop calls (`tests/test_voice_sideband.py:60-90`).

## 11. Decision list

- Separate queue module vs. generic refactor: separate module, `solstone/think/voice/observer_queue.py`.
- Lenient `call_id` semantics: `GET /api/voice/observer-actions` is always HTTP 200 with an `actions` array, even for missing or unknown `call_id`.
- Sentinel `_observer_action` vs. handler `call_id` parameter: sentinel. `dispatch_tool_call(...)` remains the uniform side-effect seam, matching `_nav_target`.
- Filename-stem constraint surfaced to iOS client: document `audio.m4a` as the required raw filename shape for downstream transcript discoverability.
- No ingest code changes: confirmed compatible. Wave 4 only documents the existing server contract and downstream caveats.

## 12. Risks / open questions

- The observer-action queue is in-memory and ephemeral. Restarting the process drops queued actions, just like nav hints. That is acceptable for this adjunct path but worth keeping explicit.
- Lenient HTTP 200 semantics are the right client contract here, but they can hide client-side `call_id` plumbing mistakes if logs are not watched. Warning logs on blank enqueue attempts help, but the route itself will not reveal misuse with a 4xx.
- The queue is keyed by the session `call_id` passed to `POST /api/voice/connect`, not the per-tool `event.call_id` echoed in `function_call_output`. The design makes that distinction explicit because the current fake integration/tests mostly use the same string for both (`tests/test_voice_integration.py:15-57`).
- Queue TTL and cap mean actions can disappear under long client stalls or bursty tool output. That matches nav-hint behavior and is acceptable, but the client should keep polling cadence tight.
- `GET /api/voice/nav-hints` currently returns HTTP 400 on missing `call_id`, but `tests/test_voice_routes.py` does not cover that branch today. That test gap is out of scope for this lode but worth a small follow-up.

## 13. Sources

- `docs/design/voice-server.md:71-87`, `docs/design/voice-server.md:172-191`, `docs/design/voice-server.md:215-233`, `docs/design/voice-server.md:412-446`
- `docs/design/push.md:11-45`, `docs/design/push.md:117-224`, `docs/design/push.md:500-583`
- `solstone/convey/__init__.py:152-169`
- `solstone/convey/voice.py:26-30`, `solstone/convey/voice.py:105-127`, `solstone/convey/voice.py:160-166`
- `solstone/think/voice/nav_queue.py:17-84`
- `solstone/think/voice/sideband.py:20-37`
- `solstone/think/voice/tools.py:121-135`, `solstone/think/voice/tools.py:686-739`
- `tests/test_voice_nav_queue.py:11-59`
- `tests/test_voice_tools.py:251-289`
- `tests/test_voice_routes.py:51-61`
- `tests/test_voice_sideband.py:60-90`
- `tests/test_voice_integration.py:15-57`, `tests/test_voice_integration.py:115-159`
- `solstone/apps/observer/routes.py:309-367`, `solstone/apps/observer/routes.py:503-643`
- `solstone/apps/observer/tests/test_routes.py:1276-1368`, `solstone/apps/observer/tests/test_routes.py:1552-1586`, `solstone/apps/observer/tests/test_routes.py:1654-1694`
- `solstone/observe/transcribe/main.py:58`, `solstone/observe/transcribe/main.py:165`, `solstone/observe/transcribe/main.py:594-609`, `solstone/observe/transcribe/main.py:885-889`
- `solstone/observe/sense.py:606-667`, `solstone/observe/sense.py:1096-1100`
- `solstone/think/media.py:6-25`
- `solstone/think/cluster.py:132-136`, `solstone/think/cluster.py:397-405`
- `solstone/apps/transcripts/routes.py:256-258`
- `solstone/think/retention.py:73-100`
