# Convey error handling - Wave 3 design

## Overview

Wave 3 closes the remaining Tier 3 shell, onboarding, and background-task error-handling gaps across 10 sites: `init.html` observers/finalize, shell background registration, chat-bar hydrate, month stats shell, todos background badge/nudges, support background verification, support ticket detail, and support announcements.

This wave is mostly shell-template work. Reuse the Wave 0 primitives, add one tiny `AppServices` helper for non-task background failures, and keep the support badge backend contract explicitly out of scope.

## Conventions carried forward

- Keep `window.logError(err, { context: ... })` alongside every owner-visible failure surface. Logging and UI are separate requirements.
- Use `err.serverMessage` as the primary owner copy when present; fall back to the explicit strings below.
- Do not add retry buttons. Reload is the recovery path unless the site already has its own action button.
- Reuse Wave 0 primitives in place: `window.apiJson`, `window.SurfaceState.errorCard`, `window.SurfaceState.replaceLoading`, and `AppServices.registerTask`.
- UI timeouts and background-task failure thresholds are owner-patience signals, not backend SLAs.

## Decision log

| Decision | Chosen | Rationale | Rejected alternatives | Final copy |
| --- | --- | --- | --- | --- |
| D-init | Shipped 2026-04-23. Add `<script src="error-handler.js">` and `<script src="api.js">` to `convey/templates/init.html`; do not load `app.js` or `websocket.js`; add local `.error-message` styling in `init.html` because onboarding does not load `app.css`. Verified: `convey/static/api.js` is standalone and `convey/static/error-handler.js` is safe without shell DOM or `appEvents`. | Smallest DRY path. Avoids micro-inlining shared helpers into onboarding. | Inline `apiJson`/`logError`; add `app.js`; add `websocket.js`; extract a shared partial just for two scripts. | None. |
| D-finalize | Shipped 2026-04-23. Move `finalize()` to `window.apiJson('/init/finalize', ...)`; add `#finalize-error` next to the CTA; log with context `init-finalize`. Keep the password field-local `showFieldStatus(...)` path for the current backend 400 password validation case and do not double-surface that error in the new slot. | `apiJson` fixes non-2xx and malformed JSON; the new slot fixes the current "button does nothing" failure. Current backend only returns 400 for password length, so field-local handling remains precise. | Password-only surfacing; silent catch; redirect on failure; modal/global error. | Password field on 400: current server string `Password must be at least 8 characters` via `err.serverMessage`. Finalize slot fallback: `Couldn't finalize setup. Check your connection and try again.` |
| D-observers | Shipped 2026-04-23. Add `#observer-error` as a separate node from `#observer-empty`; switch `loadObservers()` to `window.apiJson('/init/observers')`; hide `#observer-error` on success and hide `#observer-empty` while an error is showing; log with context `init-observers`. | Empty state and error state must stop sharing the same element. `apiJson` gives one transport contract. | Reuse `#observer-empty`; inline `fetch(...).json()`; silent keep-empty behavior. | Primary: `err.serverMessage`. Fallback: `Couldn't check for observers - reload to try again.` |
| D-bg-register | Shipped 2026-04-23. Add `AppServices.markBackgroundFailing(appName, error)` beside `registerTask()` / `getTaskHealth()` in `convey/static/app.js`. `convey/templates/app.html` background catch calls that helper plus `window.logError(err, { context: 'app-bg-register', app: '{{ app_name }}' })`. Helper only adds `.menu-item-bg-failing` to `.menu-item[data-app-name="<app>"]` and no-ops if missing. | Centralizes shell-side failing-pip behavior for background scripts that fail before they can register a task. Keeps the app loop running. | Direct DOM mutation inside template catch; fake `registerTask()` records; popup notifications for registration errors. | None - pip only. |
| D-chat-hydrate | Shipped 2026-04-23. On hydrate failure, call `setPendingState(true)`, `setStatus("Couldn't load recent chat session. Reload to try again.", "Couldn't load recent chat session. Reload to try again.")`, and `window.logError(err, { context: 'chat-hydrate' })`. Verified: `setStatus()` only accepts `(text, title)` and has no error variant. Do not add a new chat-bar state. | Reuses the existing disabled affordance on `#chatBarInput` / `#chatBarSend` and keeps the change local to `convey/templates/app.html`. | Invent a chat-bar error variant/class; leave the bar interactive; blank status on error. | `Couldn't load recent chat session. Reload to try again.` |
| D-month-shell | Shipped 2026-04-23. Use a normalized provider contract: `date_nav.html` providers return `{ data, error }`, with `window.apiJson(...)` inside the provider. `convey/static/month-picker.js` caches `{ data, error, facet }`, preserves stale `data` on same-month/same-facet refetch failure, logs with context `month-stats`, adds a warning glyph on `#date-nav-label` via CSS state when `error` is present, and renders a small inline dropdown error above the grid only when the current month has no cached data for the active facet. Clear glyph/title/inline error on next success. | Current picker has no internal header to tint and must remain non-blocking. A normalized `{ data, error }` contract keeps the provider and picker responsibilities explicit. | Block with `SurfaceState.errorCard`; add a new header row inside the picker; keep collapsing failures to empty months. | Label tooltip: `err.serverMessage || "Couldn't load month stats."` Inline first-open copy: `Couldn't load month stats.` plus `err.serverMessage` as secondary text. |
| D-todos-bg | Shipped 2026-04-23. Migrate `updateBadge()` to `AppServices.registerTask('todos', 'update-badge', { intervalMs: 5 * 60 * 1000, run, onSuccess })`. Migrate `checkNudges()` to `AppServices.registerTask('todos', 'check-nudges', { run })` with no `intervalMs`; verified `registerTask()` still performs the initial run when `intervalMs` is absent. Both task `run` functions use the task-scoped `apiJson`; validate `count` and `nudges` shape before mutating badge state or scheduling timers. Keep `_nudgeTimers` dedupe exactly as-is. | Matches support for badge cadence, fixes init-only dark failures, and avoids adding nudge re-fetch dedupe state. The current `registerTask()` contract already supports init-only work cleanly. | Poll nudges; fake init-only with a 24h interval; keep raw `fetch(...)`; add new background-task framework. | No new site-specific copy. Shared `registerTask()` failure notification remains `todos background task failing` with the thrown message. |
| D-support-bg | Shipped 2026-04-23. No client change. Keep `apps/support/background.html` as-is and record that Wave 0 already migrated it to `registerTask(intervalMs=5m)`. Flag the backend contract gap instead. | The client already uses the intended primitive. The missing behavior is server-side: the badge route never returns failure or 403 to the task. | Client-side heuristics on `count === 0`; server fix in this wave. | None. |
| D-open-ticket | Shipped 2026-04-23. Move `openTicket()` to `window.apiJson('/app/support/api/tickets/' + id)`; log with context `support-open-ticket`; on failure render a `support-empty` card into `#ticket-detail` that matches `loadTickets()`'s support-local error shape, but keep the existing back affordance/button. | `loadTickets()` is already the gold-standard pattern in this workspace. Reuse the support-local card instead of mixing in a second visual language. | Keep raw `fetch()` and sparse-ticket rendering; switch to a generic `SurfaceState` card; hide the detail pane on error. | Heading fallback: `Couldn't load ticket.` Hint: `Go back and select it again.` Button: `Back to tickets`. |
| D-announcements | Shipped 2026-04-23. Add local `announcementsFirstPaintDone` and `announcementsLastSuccessAt` state in `apps/support/workspace.html`. Move `loadAnnouncements()` to `window.apiJson(...)`; on first-paint failure render inline error copy in the banner slot and log with context `support-announcements`; on later failure after a successful load, preserve the prior banner content and append a singleton stale-indicator sibling with timestamp text. On success clear stale UI and set the flags. Note: current HEAD only calls `loadAnnouncements()` once, so the stale branch is forward-compatible and does not add a new refresh trigger. | Stops `!ok` from disappearing silently and preserves stale content if the function is ever re-run. The extra flags are the minimum state needed for the split surface. | Silent `!ok` return; always replace banner on failure; add a retry button or periodic polling. | First paint: `Couldn't load announcements. Reload to try again.` Refresh stale: `Couldn't refresh announcements - showing last known state.` Timestamp suffix: `Last updated {local time}.` |

## Implementation order

### Bundle 1

Target commit message: `convey: load api helpers in init shell`

| File | Change |
| --- | --- |
| `convey/templates/init.html` | Add `error-handler.js` and `api.js` includes; add local `.error-message` styles; add `#observer-error` and `#finalize-error` slots. |

### Bundle 2

Target commit message: `convey: surface init observer and finalize failures`

| File | Change |
| --- | --- |
| `convey/templates/init.html` | Migrate `loadObservers()` and `finalize()` to `window.apiJson`; wire the new slots; keep password validation field-local; add `window.logError` calls. |

### Bundle 3

Target commit message: `convey: mark shell background and chat hydrate failures`

| File | Change |
| --- | --- |
| `convey/static/app.js` | Add `AppServices.markBackgroundFailing(appName, error)` beside `registerTask()` / `getTaskHealth()`. |
| `convey/templates/app.html` | Use the new helper in the per-app background catch; migrate chat hydrate failure to disabled input + status copy + `logError`. |

### Bundle 4

Target commit message: `convey: signal month stats failures in date nav`

| File | Change |
| --- | --- |
| `convey/templates/date_nav.html` | Change provider to `window.apiJson(...)` and return `{ data, error }`. |
| `convey/static/month-picker.js` | Normalize/cache `{ data, error, facet }`; preserve stale data on failure; render inline picker error and warning state; log failures. |
| `convey/static/app.css` | Add label warning-glyph state and small picker-error styles. |

### Bundle 5

Target commit message: `todos: migrate badge and nudge background fetches`

| File | Change |
| --- | --- |
| `apps/todos/background.html` | Wrap badge polling and nudge scheduling in `AppServices.registerTask`; keep badge polling at 5 minutes; keep nudges init-only; add light shape validation. |

### Bundle 6

Target commit message: `support: harden ticket detail loads`

| File | Change |
| --- | --- |
| `apps/support/workspace.html` | Move `openTicket()` to `window.apiJson`; render support-local error card with back affordance; add `window.logError`. |
| `apps/support/background.html` | Verify only - no edit expected. |
| `apps/support/routes.py` | Reference only - follow-up, no edit in Wave 3. |

### Bundle 7

Target commit message: `support: split announcements first-paint and stale errors`

| File | Change |
| --- | --- |
| `apps/support/workspace.html` | Add `announcementsFirstPaintDone` / `announcementsLastSuccessAt`; move `loadAnnouncements()` to `window.apiJson`; split first-paint vs stale refresh surfaces; add `window.logError`. |

## Validation plan

- Run `make ci`.
- Run `make test`.
- Grep sweeps after implementation:
  - confirm the migrated sites no longer use raw `fetch(...).json()` at `init observers/finalize`, chat hydrate, ticket detail, announcements, and todos background.
  - confirm `AppServices.markBackgroundFailing` is the only template-side path for non-task background registration failure.
  - confirm month-picker code now references the normalized `{ data, error }` contract and the warning-state selector.
- Screenshot plan:
  - capture `/init` with forced observers failure and forced finalize failure to verify the new inline slots do not collide with `#observer-empty` or the password field.
  - capture one date-nav app with month-picker first-open failure and one stale-month failure to verify the label glyph plus inline picker error.
  - capture `/app/support` with ticket-detail failure and with announcements first-paint failure / stale indicator.
- Manual smoke:
  - open `convey/static/tests/register-task.html` to confirm the shared background-task behavior still passes after adding `markBackgroundFailing`.

## Out-of-scope follow-ups

1. `apps/support/routes.py:256-272` collapses errors to `200 {"count": 0}`. Server contract change needed so the support `registerTask` sees real failures / 403-disabled.
2. `init.html` gains `logError` calls but onboarding has no shell-level `#error-log` sink. Inline slots are the owner-visible surface; logging is telemetry-only at init.
3. Wave 2 D9 - `apps/sol/workspace.html:1371` `sol updated-days` deferred because the server collapses failure to `[]`.

### Also noted during audit

- `apps/activities/_day.html:953` - Wave 1 activity loader missing `window.logError`.
- `apps/settings/workspace.html:3564` - pre-existing undocumented empty catch after `saveControl`; `onError` at `:3562` logs, but the suppressing catch is undocumented.

## Audit evidence

- Grep sweeps passed after audit fixup: migrated Wave 3 sites no longer have raw `fetch(...).json()` at the migrated lines; recursive empty-catch sweep is empty; no `console.error` remains in the touched Wave 3 areas; all expected `logError` contexts are present (`init-observers`, `init-finalize`, `app-bg-register`, `chat-hydrate`, `month-stats`, `support-open-ticket`, `support-announcements`).
- `make test`: `3915 passed, 5 skipped, 1 warning`.
- `make test-app APP=todos`: `85 passed`; `apps/support/tests/` absent.
- `make verify-browser`: not run because `tests/verify_browser.py` does not cover the Wave 3 failure paths.
- Screenshots: not captured; direct `sol screenshot` required a running stack, and the sandbox reported ready but exited before screenshots could connect.
