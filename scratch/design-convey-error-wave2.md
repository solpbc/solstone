# Convey error handling — Wave 2 design

See `scratch/recon-convey-error-wave2.md` for the call-site inventory and backend contracts. This doc records the migration choices you will implement.

## Scope (recap)

Wave 2 migrates 17 Tier-2 sites plus 1 prerequisite scaffold upgrade across 8 app files.

D9 Sol updated-days is deferred. Do not touch that client path in Wave 2. See Out-of-scope follow-ups.

## Conventions

- Keep `window.logError(err, { context: ... })` alongside every owner-visible surface. Logging and UI are separate requirements.
- Do not add retry buttons. Owner action is reload unless the site already has its own mutation button.
- Reuse existing Wave 1 helpers in-place. Do not add overlapping helpers for home, settings, or speakers.
- Use `window.apiJson` for every migrated HTTP path. Raw `fetch(...).then(r => r.json())` leaves non-2xx and malformed JSON silent.
- Use `window.SurfaceState.errorCard(...)` for first-paint section failures and `surface-state-refresh-error` only when stale content is intentionally preserved.
- Use `window.appEvents.listen(..., options, handler)` for websocket timeout/drop handling. The overload contract is at `convey/static/websocket.js:300-322`.
- **UI patience, not backend SLA.** Client-side timeouts on long-running async work (websocket listeners, polling loops) are owner-patience thresholds — the point at which the UI stops claiming "still working" and tells the owner to reload. They are not statements about how long the backend should take. Backend work may continue past the UI timeout; the client simply stops lying about its state. This framing is reusable for Wave 3 background-task health.

## Site-by-site decisions

### D1 — Home refreshVitals + refreshNarrative

Targets: `apps/home/workspace.html:1582-1630`

Use `window.apiJson('/app/home/api/pulse')` in both refreshers. Keep `malformedHomeResponse()` for 200-shape failures instead of inventing a second parse helper.

For `refreshVitals`, preserve the existing `#pulse-vitals` content on failure and append a singleton refresh-error sibling after `#pulse-vitals`. Copy: `Couldn't refresh vitals — showing last known state.` Keep the current render path untouched on success.

For `refreshNarrative`, preserve the existing `#pulse-narrative` content on failure and append a singleton refresh-error sibling after `#pulse-narrative`. Copy: `Couldn't refresh narrative — showing last known state.` Do not blank the rendered markdown on refresh failure.

These are refresh-only paths. There is already first-paint content in the HTML, so this is a stale-preserving migration, not a loading-scaffold migration.

### D2 — Link refreshStatus + refreshDevices

Targets: `apps/link/workspace.html:171-199`

Add a stable container id to the existing status row: use `#link-status-panel` on the current `.link-status-row`. `refreshStatus()` should call `window.apiJson('/app/link/api/status', ...)`, shape-validate `typeof data.enrolled === 'boolean'`, and throw `new window.ApiError({ cause: 'parse', status: 200, serverMessage: 'Unexpected status shape' })` on mismatch.

On `refreshStatus()` failure, do not call `setStatus(...)`. Leave the last visible status text and LAN nudge state unchanged. Render a refresh-error sibling after `#link-status-panel` with copy `Can't reach pairing service — reload to try again.` This is PD9.

`refreshDevices()` should also move to `window.apiJson('/app/link/api/devices', ...)`.

Use a split surface for devices:

- First failure before the first successful device render: replace `#link-devices-list` contents with `SurfaceState.errorCard({ heading: "Couldn't load paired devices", desc: "Reload to try again.", serverMessage: err.serverMessage })`.
- Later refresh failure after at least one successful render: leave the stale device list visible and append a `surface-state-refresh-error` sibling after `#link-devices-list` with copy `Couldn't refresh paired devices — showing last known state.`

Track first-success with a small local boolean. Do not treat transport failure as `offline`.

### D3 — Chat listener

Target: `apps/chat/workspace.html:60-167`

Keep the live listener only for `today`, but switch it to the options overload:

- `schema: ['kind']`
- `correlationKey: 'use_id'`
- `timeout: 3 * 60 * 1000`
- `onDrop: window.logError`
- `onTimeout: handleChatTalentTimeout`

PD3 applies. Do not implement an owner-reply watchdog in this wave.

Only track spawned talent cards. Call `cleanup.pending.track(useId)` inside `appendEventFromLive()` when `kind === 'talent_spawned'` and `use_id` exists. Do not track `owner_message`, `sol_message`, or `reflection_ready`.

Add a stalled visual state for talent cards:

- New variant class: `chat-talent-card--stalled`
- New status value: `data-talent-status="stalled"`
- New copy: `Talent stopped responding — reload to retry`

On timeout, find the matching active talent card by `data-talent-use-id`, convert it in place to the stalled variant, and leave the rest of the transcript unchanged. If the card is already finished or errored, the timeout callback is a no-op.

### D4 — Entities cortex listener

Target: `apps/entities/workspace.html:3216-3312`

Switch `window.appEvents.listen('cortex', ...)` to the options overload with:

- `schema: ['event', 'use_id']`
- `timeout: 2 * 60 * 1000`
- `correlationKey: 'use_id'`
- `onDrop: window.logError`
- `onTimeout: handleEntitiesTimeout`

Use 2 minutes as the UI stall threshold. There is no shorter backend contract in `apps/entities/routes.py:498-574`, and cortex still allows much longer work by default (`think/cortex.py:335-339` defaults to 10 minutes). Two minutes is a UI patience threshold for these owner-triggered, single-agent actions, not a backend SLA.

Track ids in both paths:

- `submitEntityAssist()` tracks the real `use_id` immediately after the POST succeeds and the temp id is remapped.
- `listenForAgentCompletion()` tracks the `agentId` when description generation starts.

Timeout behavior must fail both maps cleanly and only once:

- If `pendingEntities.has(useId)`, call the existing failure path with copy `Entity assist timed out — reload to retry`.
- If `pendingAgentCallbacks.has(useId)`, remove the callback entry first, then invoke the callback with a timeout-shaped failure so the textarea/button unlock and the description inline error renders.

Ignore late websocket `finish`/`error` events after timeout by deleting the pending entry before surfacing the timeout.

### D5 — Settings saveField

Targets: `apps/settings/workspace.html:3270-3512` plus all `[data-section][data-key]` autosave controls

Keep `saveField()` as the debounced orchestrator. Do not move debounce into `saveControl()`.

PD5: when `el` is null, or when the caller is the auto-detected `identity.timezone` path, bypass `saveControl()` and call `window.apiJson('api/config', ...)` directly inside the existing debounce. On failure, log with `window.logError(...)`. There is no UI control to revert in that path.

For normal controls, migrate `saveField()` to `prepareFieldErrorHost(el)` plus `window.saveControl(...)`.

PD7 sub-choice: use a local last-known-good snapshot per element. This is required by `saveControl()` internals:

- `convey/static/api.js:208-213` reads `readValue` before the request and uses that as `previousValue`.
- `convey/static/api.js:220` writes `savedControlValues` from the live DOM value after success.
- `convey/static/api.js:227-228` writes `previousValue` back on failure.

That ordering means:

- default `getInitialControlValue()` is wrong for JS-populated settings fields because it falls back to `defaultValue/defaultChecked` (`convey/static/api.js:98-113`)
- a per-call snapshot of the changed DOM value would also be wrong, because `previousValue` is captured before the fetch starts

Design:

- Seed `el.__lastKnownValue` for every `saveField`-managed control during `populateFields()`, after the DOM has been filled from config.
- Pass `readValue: () => el.__lastKnownValue` for all non-env controls.
- Update `el.__lastKnownValue` in `onSuccess` to the control’s current logical value.
- Keep the default `writeValue`. The field types in scope are text, textarea, select, checkbox, and password, so the stock writer is enough.

PD6 env-key nuance (revised): **do not route env fields through `saveControl()` at all.** Reason: pre-clearing the field before `saveControl()` starts is a UX regression — on failure the user's typed secret is lost and they must re-type. Post-success clear is current UX; a pre-save clear is new behavior. Instead, env fields use a direct path:

- `prepareFieldErrorHost(el)` to seed the `<small>` host
- `window.apiJson('api/config', { method: 'PUT', ... })` with the debounced `value`
- On success: clear `el.value = ''` (preserve current post-success UX), run existing key-validation and env-status refresh, call `showFieldStatus(el, 'saved')`
- On failure: render the inline error directly into the `<small>` host using the same `[data-control-save-error]` markup shape that `saveControl()` produces (copy exactly so `.settings-field small .control-save-error` CSS applies), keep the typed secret in `el.value`, call `window.logError(...)`
- No weakmap interaction anywhere

This sidesteps the secret-caching problem entirely while preserving retry-without-retype for env saves. Non-env controls still use the `saveControl()` path per PD7.

Error behavior:

- Success path still calls `showFieldStatus(el, 'saved')`
- Failure path relies on `saveControl()` to render the field-local error in the prepared `<small>`
- Every catch still logs via `window.logError`

### D6 — Entities confirmEntityDelete

Targets: `apps/entities/workspace.html:3436-3465`

Drop the optimistic delete. The order becomes:

1. Disable the confirm button.
2. Call `window.apiJson(...)` on the DELETE route.
3. On success, clear `detected-action-error`, close the modal, then call `removeEntityFromUI(entityName)`.
4. On failure, re-enable the confirm button, close the modal, and show `showInlineError('detected-action-error', err.serverMessage || "Couldn't delete entity")`.

Do not fall back to `loadEntities()` as the primary repair path. Preserve the current local success path by keeping `removeEntityFromUI()` as the post-success side effect.

### D7 — Entities loadEntities + loadJournalEntities

Targets: `apps/entities/workspace.html:1258-1261`, `2636-2703`

PD4 applies. Upgrade the loading scaffold first so `SurfaceState.replaceLoading()` can replace in place.

Before:

```html
<div id="entities-loading" class="entities-loading">
  <div class="spinner"></div>
  <p>Loading entities...</p>
</div>
```

After:

```html
<div id="entities-loading" class="entities-loading">
  <div class="surface-state surface-state--loading" role="status" aria-busy="true">
    <div class="surface-state-spinner" aria-hidden="true"></div>
    <span class="surface-state-text" data-role="loading-status">Loading entities...</span>
  </div>
</div>
```

Keep `#entities-loading` as the outer id so callers do not change. The important change is the inner `.surface-state--loading` child, because `replaceLoading()` only replaces in place when that marker exists (`convey/static/app.js:1606-1635`).

Then migrate both loaders to `window.apiJson(...)` and replace the catch-time scaffold pollution with:

- `window.logError(...)`
- `window.SurfaceState.replaceLoading('entities-loading', window.SurfaceState.errorCard({ heading: "Couldn't load entities", desc: "Reload to try again.", serverMessage: err.serverMessage }))`

Use the same pattern for `loadJournalEntities()` with journal-specific copy only if it materially improves clarity. Otherwise keep one consistent `Couldn't load entities` surface.

### D8 — Settings four loaders

Targets:

- `loadTranscribeBackends()` at `apps/settings/workspace.html:4531-4554`
- `loadObserve()` at `4675-4683`
- `loadSync()` at `4760-4768`
- `loadStorage()` at `5800-5808`

These sections do not have loading scaffolds, so `replaceLoading()` is the wrong primitive. Add one top-of-section status slot per section, immediately under the section description:

- `#transcriptionLoadState` inside `#section-transcription`
- `#observerLoadState` inside `#section-observer`
- `#syncLoadState` inside `#section-sync`
- `#storageLoadState` inside `#section-storage`

Use `SurfaceState.errorCard(...)` in those slots.

Decisions by loader:

- `loadTranscribeBackends()`: targeted transcription-only surface. Render `Couldn't load transcription backends` in `#transcriptionLoadState`, disable the backend selector until success, and do not let this failure abort `loadConfig()`. `loadConfig()` should still populate the rest of the settings page.
- `loadObserve()`: render `Couldn't load observer settings` in `#observerLoadState` and disable `#field-tmux-enabled` plus `#field-tmux-capture-interval` until success.
- `loadSync()`: render `Couldn't load sync settings` in `#syncLoadState` and disable the Plaud, Granola, and Obsidian toggles until success.
- `loadStorage()`: render `Couldn't load storage settings` in `#storageLoadState`. On first paint, disable retention controls until success. On later refreshes after cleanup actions, keep stale storage numbers visible and use the same slot for refresh-error copy instead of blanking the section.

All four loaders still log failures. Clear the section slot on the next successful load.

### D9 — Sol updated-days — DEFERRED

Target: `apps/sol/workspace.html:1363-1385`

PD1 applies. Do not change this client path in Wave 2.

Reason: `apps/sol/routes.py:637-644` converts backend failure into `[]`, so the client cannot distinguish empty from error. Leave the existing `.catch(() => { banner.style.display = 'none'; })` unchanged.

Flag the backend-contract follow-up in Out-of-scope follow-ups and in implementation-stage gate output.

### D10 — Sol loadIdentity

Target: `apps/sol/workspace.html:1197-1231`

PD2 applies.

Move to `window.apiJson('/app/sol/api/identity')`. Remove the dead `if (data.error)` branch, because the route returns real `500 + {error: ...}` on failure and has no 200-side disabled contract.

On failure, replace `#sol-identity` contents with:

- heading: `Couldn't load identity`
- desc: `Reload to try again.`
- server message: `err.serverMessage`

Do not hide the section anymore. A missing identity card should stop meaning “backend failed silently.”

### D11 — Health retry-import

Target: `apps/health/workspace.html:1739-1759`

PD8 applies.

Move the click handler to `window.apiJson('/app/health/api/retry-import', ...)`. Remove the `data.status === 'not_implemented'` branch entirely.

Use the existing row-local importer card as the surface:

- while pending, keep the current button-disabled `Retrying...` state
- on success, clear the row error text and show `Retry sent`
- on failure, restore the button label, re-enable the button, and write `err.serverMessage || 'Retry failed'` into that card’s `.activity-card-error`

This keeps the failure local to the affected import row and lets the server’s 501 message surface unchanged.

## Additional sites

### Speakers checkOwnerStatus + submitOwnerChoice

Targets: `apps/speakers/workspace.html:1259-1364`

Use `window.apiJson(...)` for the owner-status GET, the nested owner-detect POST, and both confirm/reject POSTs.

Keep `resolveSpeakerError()` as the message normalizer. Do not add a second speakers error helper.

Surface choices:

- `checkOwnerStatus()` top-level failure: render an owner-banner error state instead of hiding both banners. Copy baseline: `Couldn't load owner status`.
- nested detect failure: clear `ownerDetectionInFlight`, keep the owner area visible, and render `Couldn't analyze voice patterns` through the same banner surface.
- `submitOwnerChoice()` failure: re-enable the buttons and render the normalized server message in the owner banner.

On success, preserve the existing `checkOwnerStatus()` refresh behavior.

### Speakers loadReview

Target: `apps/speakers/workspace.html:1725-1752`

PD10 applies.

Keep first-paint and refresh separate:

- First-paint path: when `#spkSentences` still contains the `Loading...` placeholder and no `.spk-sentence`, replace that area with `SurfaceState.errorCard({ heading: "Couldn't load review", desc: "Reload to try again.", serverMessage: err.serverMessage })`.
- Refresh path: when `.spk-sentence` rows already exist, preserve them and render one refresh-error slot at the top of the review panel with copy `Couldn't reload review — showing last known state.`

Because there is no existing top-of-panel status slot, add one in the detail render above `#spkSentences`: `#spkReviewStatus`. Clear it on the next successful `loadReview()`.

### Speakers loadUntilFound

Target: `apps/speakers/workspace.html:1461-1474`

Move to `window.apiJson(...)`. Add a catch. Do not let the recursive hash-hunt die as an unhandled rejection.

On failure:

- log the error
- stop recursion
- preserve the current segment list
- render a segment-list refresh error in a new lightweight slot above the list, not a global modal

This is a refresh/deep-link recovery failure, not a first-paint full-panel failure.

### Entities generate-description fetch

Target: `apps/entities/workspace.html:2416-2444`

Move the kickoff POST to `window.apiJson(...)`. Validate that the response includes a usable `use_id`; otherwise throw a parse-style `ApiError`.

On kickoff failure:

- re-enable the textarea
- remove `.generating`
- call `showInlineError('description-save-error', err.serverMessage || "Couldn't start description generation")`

On websocket timeout/error from D4, route the failure through the same inline surface so the owner sees one consistent description-generation error path.

### Entities previewEntityDelete

Target: `apps/entities/workspace.html:3402-3434`

Move the preview GET to `window.apiJson(...)`. Keep `showInlineError('detected-action-error', ...)` as the local surface.

Preserve the existing modal population flow on success. On failure, keep the modal closed and show the server message inline. Do not add a second delete-preview helper.

## Implementation order

1. Entities scaffold upgrade + `loadEntities()` + `loadJournalEntities()`
2. Entities `previewEntityDelete()` + `confirmEntityDelete()` + generate-description kickoff
3. Entities cortex listener timeout/drop handling
4. Settings `saveField()`
5. Settings four loaders
6. Speakers `checkOwnerStatus()` + `submitOwnerChoice()`
7. Speakers `loadReview()` + `loadUntilFound()`
8. Home `refreshVitals()` + `refreshNarrative()`
9. Link `refreshStatus()` + `refreshDevices()`
10. Sol `loadIdentity()`
11. Health retry-import
12. Chat websocket listener
13. Audit and finalize

## Decision log table

| Decision | Choice | Status | Rationale |
| --- | --- | --- | --- |
| PD1 / D9 | Defer Sol updated-days | Shipped 2026-04-23 | Backend collapses failure into `[]`; client cannot distinguish empty from error |
| PD2 / D10 | `apiJson` + identity `errorCard`; remove `data.error` branch | Shipped 2026-04-23 | Route only has success or real 500 failure |
| PD3 / D3 | Track only `talent_spawned`; 3-minute chat stall state | Shipped 2026-04-23 | `owner_message` has no `use_id`; owner watchdog is separate work |
| D4 timeout | 2-minute entities UI stall threshold | Shipped 2026-04-23 | Interactive entity actions should not hang forever; backend still has a longer timeout |
| PD4 / D7 | Retrofit `#entities-loading` with `.surface-state--loading` child | Shipped 2026-04-23 | `replaceLoading()` only replaces in place when that marker exists |
| PD5 / D5 | Keep `saveField()` debounce; null-el timezone uses direct `apiJson` | Shipped 2026-04-23 | `saveControl()` requires an element |
| PD6 / D5 | Env fields bypass `saveControl()`; use `apiJson` + `prepareFieldErrorHost` directly | Shipped 2026-04-23 | Avoids weakmap secret caching and preserves retry-without-retype UX on failure |
| PD7 / D5 | Per-element `__lastKnownValue` snapshots | Shipped 2026-04-23 | Default snapshot is wrong for JS-populated controls |
| PD8 / D11 | Treat 501 like any other error and surface server message | Shipped 2026-04-23 | Simpler and consistent; backend message already has the right copy |
| PD9 / D2 | Preserve stale link status; do not synthesize `offline` on transport failure | Shipped 2026-04-23 | `offline` is not the same as “pairing service failed” |
| PD10 / speakers | Split `loadReview()` by first paint vs refresh | Shipped 2026-04-23 | First-paint can be replaced; mutation refresh should preserve current sentences |
| PD11 / all | Keep `logError` with every visible surface | Shipped 2026-04-23 | Telemetry and owner-facing state serve different purposes |

## As-implemented deviations

- Bundle 4: `saveField()` uses a local `saveConfigValue()` wrapper and passes `fetchArgs` as a function so `saveControl()` treats `200 { success: false, error: ... }` config saves as failures instead of false positives.
- Bundle 4: the request body stays on the live route contract, `{ section, data: { [key]: value } }`, rather than the speculative `{ section, key, value, runtime_env }` shape from the early draft.
- Bundle 7: first-paint vs refresh detection keys off the actual rendered `.spk-sentence` rows plus the literal `Loading...` placeholder; the design draft’s `.spk-sentence-row` selector does not exist in the template.
- Bundle 7: `loadReview()` still converts legacy `200 { error: ... }` payloads into thrown errors so the same first-paint / refresh error surfaces handle both transport failures and envelope failures.
- Bundle 8: home adds a local `clearPulseRefreshError()` helper because `SurfaceState.replaceLoading()` appends refresh-error siblings but does not remove them after a later successful refresh.
- Bundle 11: no behavioral deviation from the approved design; the importer retry handler kept the existing delegated click-handler shape.
- Bundle 12: chat uses a local `<style>` block in `apps/chat/workspace.html` for the stalled-card visuals and normalizes the server-rendered finished-card `data-talent-status` in `apps/chat/_chat_event.html` so the live and initial DOM states match.

## Out-of-scope follow-ups

- D9 Sol updated-days banner. Backend must return a detectable error envelope or non-2xx status before the client can migrate.
- Chat owner-reply watchdog. `owner_message` has no `use_id`, so this needs a separate correlation design.
- Any `convey/static/api.js` primitive expansion. Wave 2 must stay within existing primitives.
- Shared helper consolidation across app files. Reuse local helpers now; consolidate later if the pattern repeats again.

## Open questions needing Jer's confirmation (flag these at gate time)

1. D5 env-field UX (resolved by senior): env fields bypass `saveControl()`. Use `apiJson` + `prepareFieldErrorHost` directly. Clear on success; preserve the typed secret on failure so the owner can retry without re-typing. See revised D5 env-key section.
2. D4 timeout constant: 2 minutes is the proposed UI stall threshold for entity assist and description generation. Owner-facing threshold, not a backend SLA. Senior approves; flag to Jer at gate for visibility only.
