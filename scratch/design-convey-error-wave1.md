# Wave 1 design — convey error handling migration

Scope: design only. No app/template implementation in this stage. Wave 0 shipped primitives are the ground truth. This wave migrates the 19 Tier-1 sites plus the importer listener contract.

## Files in scope

- `apps/settings/workspace.html`
- `apps/speakers/workspace.html`
- `apps/home/workspace.html`
- `apps/activities/_day.html`
- `apps/import/workspace.html`
- `apps/import/_detail.html`
- `apps/observer/workspace.html`

## Implementation order

1. Add local CSS/helpers needed by multiple sites.
2. Migrate the 7 settings saves to `saveControl(...)`.
3. Migrate speakers to `apiJson(...)` + shared resolver.
4. Upgrade first-paint loaders and refresh errors in home / activities / observer / import.
5. Migrate importer WS listener to the options overload; add stalled state + load-more banner.
6. Split import detail error surfaces.
7. Audit with grep.

## D1. Settings `errorHost`

**Decision**

- Use `.settings-field small` as `saveControl`’s `errorHost`.
- Add the amber override in `apps/settings/workspace.html`’s inline `<style>` block, not in `convey/static/app.css`.
- Pre-clear the `<small>` before calling `saveControl(...)`.

**Rationale**

- The helper `<small>` is already where owners look.
- `saveControl` only removes `[data-control-save-error]`; it does not clear arbitrary host text, so `Saved` / helper copy would otherwise coexist with the injected span.

**Sketch**

```css
.settings-field small .control-save-error {
  color: var(--color-warning, #f59e0b);
  margin-left: 0;
  font-size: inherit;
  display: inline;
}
```

```js
function prepareFieldErrorHost(el) {
  const small = el.closest('.settings-field')?.querySelector('small');
  if (!small) return null;
  if (!fieldHelperText.has(small)) fieldHelperText.set(small, small.textContent);
  const existingTimeout = fieldStatusTimeouts.get(small);
  if (existingTimeout) {
    clearTimeout(existingTimeout);
    fieldStatusTimeouts.delete(small);
  }
  small.textContent = '';
  small.classList.remove('status-saved', 'status-error', 'status-fade');
  return small;
}
```

## D2. Settings `onSuccess`

**Decision**

- Every settings `saveControl(...)` call includes `onSuccess: () => showFieldStatus(el, 'saved')` or the equivalent bound element.

**Rationale**

- Preserve the current success flash instead of making saves silent.

**Sketch**

```js
onSuccess: () => showFieldStatus(el, 'saved')
```

## D3. Speakers `friendlyError`

**Decision**

- Keep `friendlyError(...)`.
- Add local `resolveSpeakerError(err)` beside the three handlers.
- If the mapping returns the generic fallback for an unknown server message, show the raw server message instead.
- Keep the generic fallback only for the two intentional generic mappings: `Sentence embedding not found` and `No speaker labels found`.

**Rationale**

- Known failures stay owner-friendly.
- Unknown failures remain legible instead of being flattened to the generic message.

**Sketch**

```js
function resolveSpeakerError(err) {
  const raw = err?.serverMessage || err?.message || '';
  const friendly = friendlyError(raw);
  const GENERIC = 'something went wrong — try again';
  const preserveGeneric = /Sentence embedding not found|No speaker labels found/i.test(raw);
  if (friendly === GENERIC && raw && !preserveGeneric) {
    return raw;
  }
  return friendly;
}
```

Network/HTTP failures use the same resolver; `ApiError.serverMessage` is the canonical field.

## D4. Home refresh behavior

**Decision**

- `refreshSkills` and `refreshRoutines`: replace the content inside their container with `SurfaceState.errorCard(...)`.
- `refreshBriefing`: use `SurfaceState.replaceLoading('pulse-briefing', ...)` so refresh failures render as a sibling `.surface-state-refresh-error` and the existing card stays visible.

**Rationale**

- Skills/routines are volatile sections; blanking them is acceptable.
- Briefing is persistent owner-facing copy; preserve it in stale state.

**Sketch**

```js
container.innerHTML = window.SurfaceState.errorCard({
  heading: "Couldn't refresh skills",
  desc: 'Reload the page to try again.',
  serverMessage: err?.serverMessage ?? err?.message,
});
```

```js
window.SurfaceState.replaceLoading('pulse-briefing', window.SurfaceState.errorCard({
  heading: "Couldn't refresh briefing",
  desc: 'Reload the page to try again.',
  serverMessage: err?.serverMessage ?? err?.message,
}));
```

## D5. Custom loading scaffolds

**Decision**

- Replace the bespoke loading HTML at:
  - `apps/observer/workspace.html:512`
  - `apps/activities/_day.html:498`
  - `apps/import/workspace.html:610`
- Hard-code the same DOM structure emitted by `SurfaceState.loading(...)` so `replaceLoading(...)` detects first-paint loading via `.surface-state--loading`.

**Rationale**

- Consistent first-paint markup lets the shared replacement heuristic work everywhere.

**Sketch**

```html
<div class="surface-state surface-state--loading" role="status" aria-busy="true">
  <div class="surface-state-spinner" aria-hidden="true"></div>
  <span class="surface-state-text" data-role="loading-status">Loading observers...</span>
</div>
```

Use the same structure with text adjusted to:

- `Loading observers...`
- `Loading activities...`
- `Loading imports...`

## D6. Importer WS timeout

Note: the `appEvents.listen` options-overload default `correlationKey` is `'use_id'` (confirmed at `convey/static/websocket.js:310`). Importer opts into `'import_id'` explicitly. This correction was logged at gate approval.

**Decision**

- Add `const IMPORT_STALL_TIMEOUT_MS = 10 * 60 * 1000;` near the top of `apps/import/workspace.html`.
- Use the `appEvents.listen` options overload with `correlationKey: 'import_id'`.

**Rationale**

- Ten minutes is the pragmatic stall threshold.
- The overload defaults to `use_id`; importer must opt into `import_id`.

**Sketch**

```js
const IMPORT_STALL_TIMEOUT_MS = 10 * 60 * 1000;
const IMPORT_ROW_EVENTS = new Set(['started', 'status', 'completed', 'error']);
const IMPORT_TERMINAL_EVENTS = new Set(['completed', 'error']);
```

```js
const importEventsCleanup = window.appEvents.listen('importer', {
  correlationKey: 'import_id',
  schema: ['import_id', 'event'],
  timeout: IMPORT_STALL_TIMEOUT_MS,
  onTimeout: markRowStalled,
}, (eventData) => {
  if (!IMPORT_ROW_EVENTS.has(eventData.event)) return;
  updateImportRow(eventData.import_id, eventData);
  if (!IMPORT_TERMINAL_EVENTS.has(eventData.event)) {
    importEventsCleanup.pending.track(eventData.import_id);
  }
});
```

Also arm the timer after local synthetic starts and after `loadImports()` discovers already-running rows.

## D7. Importer stalled UI

**Decision**

- Add `import-row--stalled`.
- Replace the row’s blue running badge with a non-spinning amber `stalled` badge.
- Surface the `import_id` in the row.
- Mirror the stalled state into `importEvents[importId]` so the progress route can show it too.

**Rationale**

- Stalled is non-terminal and must be visually distinct from running, failed, and completed.

**Sketch**

```css
.import-row--stalled { background: rgba(245, 158, 11, 0.08); }
.import-row--stalled:hover { background: rgba(245, 158, 11, 0.12); }
.import-status.stalled { background: rgba(245, 158, 11, 0.16); color: #92400e; }
.import-stalled-meta { color: #92400e; }
```

```js
function markRowStalled(importId) {
  const row = document.querySelector(`tr[data-import-id="${importId}"]`);
  if (!row) return;
  row.classList.add('import-row--stalled');
  row.querySelector('.status-cell').innerHTML =
    `<span class="import-status stalled">Stalled</span><div class="progress-detail import-stalled-meta">Stalled — no updates in 10 minutes. Import ID: ${escapeHtml(importId)}. Reload to retry.</div>`;
  importEvents[importId] = { ...(importEvents[importId] || {}), import_id: importId, event: 'stalled', stalled: true };
  refreshInlineProgress(importId, importEvents[importId]);
}
```

`updateImportRow(...)` clears the stalled row class when a later `started` or `status` arrives.

## D8. `loadMoreImports` banner

**Decision**

- On `loadMoreImports()` failure, insert a singleton `.surface-state-refresh-error` banner between the table and `#importLoadMore`.
- Preserve existing rows and keep the load-more control.

**Rationale**

- The rows already on screen remain useful state.

**Sketch**

```js
function renderLoadMoreError(err) {
  document.getElementById('importLoadMoreError')?.remove();
  const loadMore = document.getElementById('importLoadMore');
  if (!loadMore?.parentNode) return;
  const banner = document.createElement('div');
  banner.id = 'importLoadMoreError';
  banner.className = 'surface-state-refresh-error';
  banner.innerHTML = `<strong>Couldn't load more imports.</strong> ${escapeHtml(err?.serverMessage ?? err?.message ?? 'Reload the page to try again.')}`;
  loadMore.parentNode.insertBefore(banner, loadMore);
}
```

Remove `#importLoadMoreError` on the next successful page append.

## D9. `_detail.html` failure split

**Decision**

- `#importMeta` becomes the primary error surface with a prominent `.surface-state-refresh-error` banner that includes `err.serverMessage`.
- `#overviewContent`, `#importJsonContent`, and `#importedJsonContent` become neutral unavailable/no-data copy.
- Do not use the literal word `Pending` in the error path.

**Rationale**

- `Pending` is a legitimate server state and must stay reserved for that state.
- The meta strip is the most visible slot on the page.

**Sketch**

```js
document.getElementById('importMeta').innerHTML =
  `<div class="surface-state-refresh-error" role="alert"><strong>Couldn't load import details.</strong> ${escapeHtml(err?.serverMessage ?? err?.message ?? 'Reload the page to try again.')}</div>`;
document.getElementById('overviewContent').innerHTML = '<div class="no-data">Import details are unavailable right now.</div>';
document.getElementById('importJsonContent').innerHTML = '<span class="no-data">Import metadata unavailable.</span>';
document.getElementById('importedJsonContent').innerHTML = '<span class="no-data">Processed metadata unavailable.</span>';
```

## D10. Observer `loadObservers`

**Decision**

- Migrate `loadObservers()` to `window.apiJson(...)`.
- On failure, use `window.SurfaceState.replaceLoading('observersList', window.SurfaceState.errorCard(...))`.
- Remove `showLocalError(..., { retry: true })` from this loader only.
- Leave `showLocalError(...)` itself intact; do not delete the retry branch in this wave.

**Rationale**

- First-paint and refresh failures should look like tokens-style data errors, not like local action errors.

**Sketch**

```js
window.SurfaceState.replaceLoading('observersList', window.SurfaceState.errorCard({
  heading: "Couldn't load observers",
  desc: 'Reload the page to try again.',
  serverMessage: err?.serverMessage ?? err?.message,
}));
```

## Per-site code sketches

### 1. `apps/settings/workspace.html:4769` Plaud sync toggle

**Before**
```js
const response = await fetch('api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plaud: { enabled } }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
showFieldStatus(this, 'saved');
```

**After**
```js
const el = this;
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plaud: { enabled } }) }],
    onSuccess: () => showFieldStatus(el, 'saved'),
  });
} catch (_) {}
```

### 2. `apps/settings/workspace.html:4786` Granola sync toggle

**Before**
```js
const response = await fetch('api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ granola: { enabled } }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
showFieldStatus(this, 'saved');
```

**After**
```js
const el = this;
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ granola: { enabled } }) }],
    onSuccess: () => showFieldStatus(el, 'saved'),
  });
} catch (_) {}
```

### 3. `apps/settings/workspace.html:4803` Obsidian sync toggle

**Before**
```js
const response = await fetch('api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ obsidian: { enabled } }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
showFieldStatus(this, 'saved');
```

**After**
```js
const el = this;
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/sync', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ obsidian: { enabled } }) }],
    onSuccess: () => showFieldStatus(el, 'saved'),
  });
} catch (_) {}
```

### 4. `apps/settings/workspace.html:5897` `saveRetentionConfig(data)`

**Before**
```js
const response = await fetch('api/storage', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
const result = await response.json();
if (result.error) throw new Error(result.error);
showFieldStatus(el, 'saved');
```

**After**
```js
const el = document.getElementById('retentionModeField');
const errorHost = el ? prepareFieldErrorHost(el) : null;
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/storage', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }],
    onSuccess: () => showFieldStatus(el, 'saved'),
  });
} catch (_) {}
```

### 5. `apps/settings/workspace.html:4310` provider/tier/backup loop

**Before**
```js
const response = await fetch('api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [type]: { [field]: value } }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
providersData = result;
showFieldStatus(el, 'saved');
```

**After**
```js
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [type]: { [field]: value } }) }],
    onSuccess: (result) => { providersData = result; /* existing provider/tier side-effects */ showFieldStatus(el, 'saved'); },
  });
} catch (_) {}
```

### 6. `apps/settings/workspace.html:4342` cogitate auth change

**Before**
```js
const response = await fetch('api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auth: { [provider]: authSelect.value } }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
providersData = result;
showFieldStatus(authSelect, 'saved');
```

**After**
```js
const el = this;
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auth: { [provider]: el.value } }) }],
    onSuccess: (result) => { providersData = result; updateTypeProviderKeyWarning('cogitate', provider, result.api_keys, result.auth); showFieldStatus(el, 'saved'); },
  });
} catch (_) {}
```

### 7. `apps/settings/workspace.html:4364` google backend change

**Before**
```js
const response = await fetch('api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ google_backend: value }) });
const result = await response.json();
if (result.error) throw new Error(result.error);
providersData = result;
showFieldStatus(this, 'saved');
```

**After**
```js
const el = this;
const errorHost = prepareFieldErrorHost(el);
try {
  await window.saveControl({
    el, errorHost,
    fetchArgs: ['api/providers', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ google_backend: value }) }],
    onSuccess: (result) => { providersData = result; /* existing vertex/google field toggles */ showFieldStatus(el, 'saved'); },
  });
} catch (_) {}
```

### 8. `apps/speakers/workspace.html:1985` `confirmAttribution`

**Before**
```js
fetch('/app/speakers/api/confirm-attribution', { ... })
  .then(r => r.json())
  .then(data => { if (data.error) { showStatusBySentence(sentenceId, friendlyError(data.error), 'error'); return; } ... })
  .catch(() => { showStatusBySentence(sentenceId, 'Failed to confirm attribution — try again', 'error'); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/speakers/api/confirm-attribution', { ... });
  if (data.error) { showStatusBySentence(sentenceId, resolveSpeakerError({ serverMessage: data.error, message: data.error }), 'error'); return; }
  ...
} catch (err) {
  showStatusBySentence(sentenceId, resolveSpeakerError(err), 'error');
}
```

### 9. `apps/speakers/workspace.html:2020` `correctAttribution`

**Before**
```js
fetch('/app/speakers/api/correct-attribution', { ... })
  .then(r => r.json())
  .then(data => { if (data.error) { showStatusBySentence(sentenceId, friendlyError(data.error), 'error'); return; } ... })
  .catch(() => { showStatusBySentence(sentenceId, 'Failed to correct attribution — try again', 'error'); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/speakers/api/correct-attribution', { ... });
  if (data.error) { showStatusBySentence(sentenceId, resolveSpeakerError({ serverMessage: data.error, message: data.error }), 'error'); return; }
  ...
} catch (err) {
  showStatusBySentence(sentenceId, resolveSpeakerError(err), 'error');
}
```

### 10. `apps/speakers/workspace.html:2047` `assignAttribution`

**Before**
```js
fetch('/app/speakers/api/assign-attribution', { ... })
  .then(r => r.json())
  .then(data => { if (data.error) { showStatusBySentence(sentenceId, friendlyError(data.error), 'error'); return; } ... })
  .catch(() => { showStatusBySentence(sentenceId, 'Failed to assign attribution — try again', 'error'); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/speakers/api/assign-attribution', { ... });
  if (data.error) { showStatusBySentence(sentenceId, resolveSpeakerError({ serverMessage: data.error, message: data.error }), 'error'); return; }
  ...
} catch (err) {
  showStatusBySentence(sentenceId, resolveSpeakerError(err), 'error');
}
```

### 11. `apps/home/workspace.html:1639` `refreshRoutines`

**Before**
```js
fetch('/app/home/api/pulse')
  .then(r => r.json())
  .then(data => { ... })
  .catch(function(err) { console.error('home: refreshRoutines failed', err); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/home/api/pulse');
  ...
} catch (err) {
  console.error('home: refreshRoutines failed', err);
  document.getElementById('pulse-routines')?.replaceChildren();
  document.getElementById('pulse-routines')?.insertAdjacentHTML('afterbegin', window.SurfaceState.errorCard({ heading: "Couldn't refresh routines", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
}
```

### 12. `apps/home/workspace.html:1702` `refreshSkills`

**Before**
```js
fetch('/app/home/api/pulse')
  .then(r => r.json())
  .then(data => { ... })
  .catch(function(err) { console.error('home: refreshSkills failed', err); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/home/api/pulse');
  ...
} catch (err) {
  console.error('home: refreshSkills failed', err);
  document.getElementById('pulse-skills')?.replaceChildren();
  document.getElementById('pulse-skills')?.insertAdjacentHTML('afterbegin', window.SurfaceState.errorCard({ heading: "Couldn't refresh skills", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
}
```

### 13. `apps/home/workspace.html:1772` `refreshBriefing`

**Before**
```js
fetch('/app/home/api/briefing')
  .then(function(r) { return r.json(); })
  .then(function(data) { ... })
  .catch(function(err) { console.error('home: refreshBriefing failed', err); });
```

**After**
```js
try {
  const data = await window.apiJson('/app/home/api/briefing');
  ...
} catch (err) {
  console.error('home: refreshBriefing failed', err);
  window.SurfaceState.replaceLoading('pulse-briefing', window.SurfaceState.errorCard({ heading: "Couldn't refresh briefing", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
}
```

### 14. `apps/activities/_day.html:896` `loadData`

**Before**
```js
fetch(`/app/activities/api/day/${day}/activities`).then(r => r.json()).then(acts => {
  allActivities = acts || [];
  ...
}).catch(() => {
  document.getElementById('timelineView').innerHTML = '<div class="timeline-empty">Failed to load data.</div>';
});
```

**After**
```js
try {
  const acts = await window.apiJson(`/app/activities/api/day/${day}/activities`);
  allActivities = acts || [];
  ...
} catch (err) {
  window.SurfaceState.replaceLoading('timelineView', window.SurfaceState.errorCard({ heading: "Couldn't load activities", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
}
```

### 15. `apps/import/workspace.html:978` `loadImports`

**Before**
```js
const response = await fetch('/app/import/api/list?page=1&per_page=25');
const data = await response.json();
...
document.getElementById('importListContent').innerHTML = `...Retry...`;
```

**After**
```js
const data = await window.apiJson('/app/import/api/list?page=1&per_page=25');
...
document.getElementById('importLoadMoreError')?.remove();
document.querySelectorAll('.import-table tbody tr').forEach(row => {
  if (row.querySelector('.import-status.running')) importEventsCleanup.pending.track(row.dataset.importId);
});
// catch:
window.SurfaceState.replaceLoading('importListContent', window.SurfaceState.errorCard({ heading: "Couldn't load import history", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
```

### 16. `apps/import/workspace.html:1052` `loadMoreImports`

**Before**
```js
const response = await fetch(`/app/import/api/list?page=${currentPage}&per_page=25`);
const data = await response.json();
...
console.error('Failed to load more imports:', err);
```

**After**
```js
const data = await window.apiJson(`/app/import/api/list?page=${currentPage}&per_page=25`);
document.getElementById('importLoadMoreError')?.remove();
...
// catch:
console.error('Failed to load more imports:', err);
renderLoadMoreError(err);
```

### 17. `apps/import/workspace.html:1791` importer listener

**Before**
```js
appEvents.listen('importer', eventData => updateImportRow(eventData.import_id, eventData));
```

**After**
```js
const importEventsCleanup = window.appEvents.listen('importer', {
  correlationKey: 'import_id',
  schema: ['import_id', 'event'],
  timeout: IMPORT_STALL_TIMEOUT_MS,
  onTimeout: markRowStalled,
}, (eventData) => {
  if (!IMPORT_ROW_EVENTS.has(eventData.event)) return;
  updateImportRow(eventData.import_id, eventData);
  if (!IMPORT_TERMINAL_EVENTS.has(eventData.event)) importEventsCleanup.pending.track(eventData.import_id);
});
```

### 18. `apps/import/_detail.html:437` detail bootstrap load

**Before**
```js
fetch('/app/import/api/{{ timestamp }}')
  .then(r => r.json())
  .then(data => { ... })
  .catch(err => {
    document.getElementById('importMeta').innerHTML = '<span style="color:red;">Error loading import details</span>';
    document.getElementById('overviewContent').innerHTML = '<div class="no-data">Error loading data</div>';
    document.getElementById('importJsonContent').innerHTML = '<span class="no-data">Error loading data</span>';
    document.getElementById('importedJsonContent').innerHTML = '<span class="no-data">Error loading data</span>';
  });
```

**After**
```js
window.apiJson('/app/import/api/{{ timestamp }}')
  .then(data => { ... })
  .catch(err => {
    document.getElementById('importMeta').innerHTML = `<div class="surface-state-refresh-error" role="alert"><strong>Couldn't load import details.</strong> ${escapeHtml(err?.serverMessage ?? err?.message ?? 'Reload the page to try again.')}</div>`;
    document.getElementById('overviewContent').innerHTML = '<div class="no-data">Import details are unavailable right now.</div>';
    document.getElementById('importJsonContent').innerHTML = '<span class="no-data">Import metadata unavailable.</span>';
    document.getElementById('importedJsonContent').innerHTML = '<span class="no-data">Processed metadata unavailable.</span>';
  });
```

### 19. `apps/observer/workspace.html:743` `loadObservers`

**Before**
```js
const response = await fetch('/app/observer/api/list');
const payload = await response.json();
...
showLocalError('couldn\'t load observers — the server may be unreachable. it will retry automatically, or you can retry now.', { retry: true });
```

**After**
```js
const payload = await window.apiJson('/app/observer/api/list');
...
// catch:
window.SurfaceState.replaceLoading('observersList', window.SurfaceState.errorCard({ heading: "Couldn't load observers", desc: 'Reload the page to try again.', serverMessage: err?.serverMessage ?? err?.message }));
```

## Audit grep checklist

Repo-local scope doc with the original audit regexes is not present in this worktree. Use this derived checklist in the audit stage.

**Legacy patterns that must disappear**

- `apps/settings/workspace.html`: `fetch\('api/(sync|providers|storage)'`
- `apps/speakers/workspace.html`: `fetch\('/app/speakers/api/(confirm-attribution|correct-attribution|assign-attribution)'`
- `apps/home/workspace.html`: `console\.error\('home: refresh(Routines|Skills|Briefing) failed'`
- `apps/activities/_day.html`: `fetch\(`/app/activities/api/day/\$\{day\}/activities`\)`
- `apps/import/workspace.html`: `fetch\('/app/import/api/list\?page=1&per_page=25'`
- `apps/import/workspace.html`: `fetch\(`/app/import/api/list\?page=\$\{currentPage\}&per_page=25`\)`
- `apps/import/workspace.html`: `appEvents\.listen\('importer', eventData => updateImportRow\(eventData\.import_id, eventData\)\)`
- `apps/import/_detail.html`: `fetch\('/app/import/api/\{\{ timestamp \}\}'`
- `apps/observer/workspace.html`: `showLocalError\([^)]*\{ retry: true \}\)`
- `apps/observer/workspace.html`: `fetch\('/app/observer/api/list'`

**New patterns that must appear**

- `apps/settings/workspace.html`: `prepareFieldErrorHost\(`
- `apps/settings/workspace.html`: `errorHost`
- `apps/settings/workspace.html`: `showFieldStatus\(.*'saved'`
- `apps/speakers/workspace.html`: `function resolveSpeakerError\(err\)`
- `apps/home/workspace.html`: `window\.SurfaceState\.errorCard`
- `apps/home/workspace.html`: `window\.SurfaceState\.replaceLoading\('pulse-briefing'`
- `apps/activities/_day.html`: `window\.SurfaceState\.replaceLoading\('timelineView'`
- `apps/import/workspace.html`: `const IMPORT_STALL_TIMEOUT_MS = 10 \* 60 \* 1000`
- `apps/import/workspace.html`: `correlationKey: 'import_id'`
- `apps/import/workspace.html`: `onTimeout: markRowStalled`
- `apps/import/workspace.html`: `import-row--stalled`
- `apps/import/workspace.html`: `surface-state-refresh-error`
- `apps/import/_detail.html`: `surface-state-refresh-error`
- `apps/observer/workspace.html`: `window\.SurfaceState\.replaceLoading\('observersList'`

## Risks and open questions

- Importer dedup can emit `started` without a terminal event; Wave 1 only surfaces that as stalled UI.
- `file_imported` / `enrichment_ready` must be ignored by row-state caching or they will clobber stage/elapsed fields.
- `prepareFieldErrorHost(...)` must seed `fieldHelperText` before clearing the `<small>`, or the original helper text is lost after the first save.
- Skills/routines containers are conditionally server-rendered; when absent, the refresh-error path no-ops instead of synthesizing a new section.
- `showLocalError(...)` may end this wave with no `{ retry: true }` callers; keep the retry branch for a later cleanup wave.

## CPO follow-up

The importer dedup path at `think/importers/cli.py:726-741` can emit `importer:started` without a terminal `completed`/`error` event. Wave 1 surfaces this client-side as the stalled UI. The server-side fix (always emit a terminal event, or a distinct `deduped` short-circuit event) is out of scope for this wave — flag to CPO as a separate ticket after Wave 1 ships.
