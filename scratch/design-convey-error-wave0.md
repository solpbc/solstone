# Design: convey-error-handling Wave 0

This file is the lode-local decision log target for Wave 0. The final decision table will also be mirrored into the spec at commit time.

## Decision log

| # | Decision | Chosen | Rationale |
|---|----------|--------|-----------|
| 1 | `api.js` housing | Add `convey/static/api.js`; insert it in `convey/templates/app.html` immediately after `websocket.js` and before the inline sidebar-state script. | That load point runs after `window.appEvents` exists and before any workspace/background consumers, while staying clear of `SurfaceState` and `AppServices` init ordering. |
| 2 | `saveControl` housing | Keep `saveControl` in `convey/static/api.js` beside `ApiError` and `apiJson`. | The save helper depends on the fetch/error contract, so keeping them together avoids split ownership and duplicate imports. |
| 3 | `appEvents.listen` overload | Keep the existing two-arg form and add `(tract, options, fn)` auto-detected by `typeof options === 'object' && typeof fn === 'function'`; the overloaded return stays the cleanup function and is augmented with `pending.track(corrId)` / `pending.clear(corrId)`. | Preserving the cleanup-function return keeps all existing listeners compatible. Attaching `pending` to the returned cleanup function avoids adding a second public API just for correlation tracking. |
| 4 | `onParseError` routing | Add `appEvents.onParseError(fn)`; on websocket parse failure, keep the existing `console.warn`, call every registered parse-error handler, and call `window.logError(error, { context: 'websocket-parse' })` if defined. | This keeps current console visibility, adds explicit subscriber hooks, and routes parse failures into the existing owner-visible error log when available. It also means `error-handler.js` must expose `window.logError`. |
| 5 | `replaceLoading` heuristic | Add `SurfaceState.replaceLoading(container, options)` with the exact heuristic from the scope: if `container.querySelector('.surface-state--loading')` is truthy, treat it as first paint and `replaceChildren(...)`; otherwise manage a singleton `.surface-state-refresh-error` sibling after the last non-error child. | The heuristic matches the audited failure split between first-paint emptiness and refresh-on-stale-content without forcing every caller to hand-roll placement rules. |
| 6 | `registerTask` 403 behavior | On `ApiError.status === 403`, stop the interval, set `health.disabled = true`, do not tint the menu pip, do not notify, and retain the task record in `AppServices.getTaskHealth(appName)`. `registerTask` owns the no-auth-redirect behavior internally so consumers do not pass `noAuthRedirect`. | Support uses `403` as a real disabled state, not a broken-state signal. Centralizing that rule inside `registerTask` keeps background consumers simple and makes diagnostics consistent. |
| 7 | `.menu-item-bg-failing` visual | Apply the class to the existing `<li class="menu-item">` root and render the failing indicator with a `::after` 8px amber pip using `var(--color-warning, #f59e0b)` and a `2px solid var(--facet-bg-primary, #fff)` border. No DOM mutation. | The menu already has a stable root for per-app state, and a pseudo-element avoids template churn while still reading on hover/focus/current states. |
| 8 | Proof-of-adoption set | Adopt the proof set exactly as scoped: `convey/static/pairing.js`, starred-app toggle in `convey/static/app.js`, `apps/tokens/workspace.html`, and `apps/support/background.html`. Ship the `appEvents.listen` overload in Wave 0 without a real-site migration. | This gives one concrete adopter for each new primitive with low blast radius and good audit coverage. Deferring a live websocket-listener migration avoids binding timeout UX decisions into Wave 0 before later waves choose them per surface. |
| 9 | Test form | Add self-contained static smoke pages under `convey/static/tests/` plus `convey/static/tests/README.md`; do not wire them into `tests/verify_browser.py` in this lode. | The repo currently has no `convey/static/tests/` harness. Static HTML pages are enough to prove the primitives without adding a pinchtab dependency to Wave 0. |
| 10 | `ApiError` fields | `ApiError extends Error` with `status`, `statusText`, `serverMessage`, `url`, and optional `cause: 'parse'`; `message === serverMessage`; no `cause: 'network'`. | This keeps `instanceof ApiError` reliable, preserves the server text exactly, and leaves network failures on the existing global error path instead of inventing a second network-error contract. |
| 11 | Error envelope robustness | `apiJson` extracts `payload?.error ?? payload?.message ?? 'Request failed (HTTP ${status})'` and documents that contract in JSDoc. | That catches the dominant server contract plus the existing todos `{"status":"error","message":"..."}` shape without requiring a server-side cleanup before Wave 1-3 adoption. |
| 12 | Menu-item structural precondition | Do not add a new `position: relative` rule: the general `.menu-bar .menu-item` rule already has it in `convey/static/app.css`. | The pip already has a stable positioning anchor at the house-style rule level, so duplicating the property would be noise. |

## Primitive APIs (signatures only)

```js
/**
 * @template T
 * @param {string} url
 * @param {RequestInit & { noAuthRedirect?: boolean }} [opts]
 * @returns {Promise<T>}
 * @throws {ApiError}
 */
window.apiJson = function apiJson(url, opts) {};

/**
 * @extends Error
 * @param {{
 *   status: number,
 *   statusText: string,
 *   serverMessage: string,
 *   url: string,
 *   cause?: 'parse'
 * }} init
 */
class ApiError extends Error {}

/**
 * @template T
 * @param {{
 *   el: HTMLElement,
 *   request: () => Promise<T>,
 *   snapshot?: () => unknown,
 *   revertOnError?: boolean | ((snapshot: unknown, error: Error) => void),
 *   renderError?: (message: string, error: Error) => void,
 *   clearError?: () => void,
 *   onSuccess?: (result: T) => void
 * }} options
 * @returns {Promise<T>}
 */
window.saveControl = function saveControl(options) {};

/**
 * @param {{
 *   icon?: string,
 *   heading?: string,
 *   desc?: string,
 *   action?: string | HTMLElement,
 *   headingLevel?: string
 * }} [options]
 * @returns {HTMLElement}
 */
window.SurfaceState.errorCard = function errorCard(options) {};

/**
 * @param {Element} container
 * @param {{
 *   icon?: string,
 *   heading?: string,
 *   desc?: string,
 *   action?: string | HTMLElement,
 *   headingLevel?: string
 * }} [options]
 * @returns {HTMLElement}
 */
window.SurfaceState.replaceLoading = function replaceLoading(container, options) {};

/**
 * @param {string} tract
 * @param {(msg: any) => void} fn
 * @returns {() => void}
 */
window.appEvents.listen = function listen(tract, fn) {};

/**
 * @param {string} tract
 * @param {{
 *   schema?: (msg: any) => boolean,
 *   timeout?: number,
 *   onDrop?: (msg: any, error: Error) => void,
 *   onTimeout?: (corrId: string) => void,
 *   correlationKey?: string | ((msg: any) => string | null | undefined)
 * }} options
 * @param {(msg: any) => void} fn
 * @returns {(() => void) & { pending: { track(corrId: string): void, clear(corrId: string): void } }}
 */
window.appEvents.listen = function listen(tract, options, fn) {};

/**
 * @param {(error: Error, rawData: string) => void} fn
 * @returns {() => void}
 */
window.appEvents.onParseError = function onParseError(fn) {};

/**
 * @typedef {{
 *   disabled: boolean,
 *   failing: boolean,
 *   lastError: string | null,
 *   lastRunAt: number | null,
 *   lastSuccessAt: number | null,
 *   consecutiveFailures: number
 * }} AppTaskHealth
 */

/**
 * @template T
 * @param {string} appName
 * @param {string} taskName
 * @param {{
 *   intervalMs: number,
 *   run: (task: { apiJson: typeof window.apiJson }) => Promise<T>,
 *   onSuccess?: (result: T) => void,
 *   onError?: (error: Error) => void,
 *   failuresBeforeFailing?: number
 * }} options
 * @returns {{ stop(): void, runNow(): Promise<void>, getHealth(): AppTaskHealth }}
 */
window.AppServices.registerTask = function registerTask(appName, taskName, options) {};

/**
 * @param {string} appName
 * @returns {Record<string, AppTaskHealth>}
 */
window.AppServices.getTaskHealth = function getTaskHealth(appName) {};
```

## Migration plan

1. `convey/static/error-handler.js` — expose `window.logError` while preserving the existing bottom-log behavior so websocket parse failures have an owner-visible sink.
2. `convey/templates/app.html` — insert `api.js` after `websocket.js`; no other load-order change.
3. `convey/static/api.js` — add `ApiError`, `apiJson`, and `saveControl`.
4. `convey/static/websocket.js` — add the `listen` overload, parse-error fanout, `onParseError`, and correlation timeout plumbing.
5. `convey/static/app.js` — add `SurfaceState.errorCard`, `SurfaceState.replaceLoading`, `AppServices.registerTask`, `AppServices.getTaskHealth`, starred-app toggle adoption, and menu failing-state class plumbing.
6. `convey/static/app.css` — add `.surface-state-refresh-error` and `.menu-item-bg-failing::after` styling. Do not add a new `.menu-item { position: relative; }` rule because it already exists.
7. `convey/static/pairing.js` — replace the local `fetchJson` implementation with a thin `apiJson(..., { noAuthRedirect: true })` wrapper and update both existing callers in that file.
8. `apps/tokens/workspace.html` — switch the load-failure catch block to `SurfaceState.errorCard` / `replaceLoading` and remove the Retry button per founder direction.
9. `apps/support/background.html` — migrate badge polling to `AppServices.registerTask` and delete the silent-swallow comment.
10. `convey/static/tests/README.md` plus `convey/static/tests/api.html`, `surface-state.html`, `ws-listen.html`, `register-task.html` — add static smoke coverage for the new primitives.

No `convey/templates/menu_bar.html` change is needed; the existing `<li class="menu-item">` root is the anchor for the failing pip.

## Test plan

- `convey/static/tests/api.html` — smoke `ApiError`, `apiJson`, error-envelope extraction, redirect suppression, and `saveControl` revert behavior with monkeypatched `window.fetch`.
- `convey/static/tests/surface-state.html` — smoke `SurfaceState.errorCard` and `SurfaceState.replaceLoading` for both first-paint and refresh placement.
- `convey/static/tests/ws-listen.html` — smoke `appEvents.listen` overload, pending tracking, timeout callbacks, parse-error fanout, and `onParseError`.
- `convey/static/tests/register-task.html` — smoke `registerTask` lifecycle, success/failure transitions, 403 disable behavior, and `getTaskHealth`.

## Risks / open questions

- `window.logError` is not currently public, so Wave 0 must expose it without regressing the existing `window.error` / `unhandledrejection` surfaces.
- The websocket overload ships without a production adopter in Wave 0; the static smoke page is the only proof until later waves migrate a real listener.
- Static smoke pages are manual in this wave and are not wired into CI or `verify_browser`, so enforcement still depends on reviewer discipline.
