# Sunstone App Development Guide

**Complete guide for building apps in the `apps/` directory.**

Apps are the primary way to extend Sunstone's web interface (Convey). Each app is a self-contained module discovered automatically using **convention over configuration**—no base classes or manual registration required.

---

## Quick Start

Create a minimal app in two steps:

```bash
# 1. Create app directory (use underscores, not hyphens!)
mkdir apps/my_app

# 2. Create workspace template
touch apps/my_app/workspace.html
```

**Minimal `workspace.html`:**
```html
<h1>Hello from My App!</h1>
```

**That's it!** Restart Convey and your app is automatically available at `/app/my_app`.

All apps are served via a shared route handler at `/app/{app_name}`. You only need `routes.py` if your app requires custom routes beyond the index page (e.g., API endpoints, form handlers, or navigation routes).

---

## Directory Structure

```
apps/my_app/
├── workspace.html     # Required: Main content template
├── routes.py          # Optional: Flask blueprint (only if custom routes needed)
├── app.json          # Optional: Metadata (icon, label, facet support)
├── hooks.py          # Optional: Dynamic submenu and badge logic
├── app_bar.html      # Optional: Bottom bar controls (forms, buttons)
└── background.html   # Optional: Background JavaScript service
```

### File Purposes

| File | Required | Purpose |
|------|----------|---------|
| `workspace.html` | **Yes** | Main app content (rendered in container) |
| `routes.py` | No | Flask blueprint for custom routes (API endpoints, forms, etc.) |
| `app.json` | No | Icon, label, facet support overrides |
| `hooks.py` | No | Submenu items and facet badge counts |
| `app_bar.html` | No | Bottom fixed bar for app controls |
| `background.html` | No | Background service (WebSocket listeners) |

---

## Naming Conventions

**Critical for auto-discovery:**

1. **App directory**: Use `snake_case` (e.g., `my_app`, **not** `my-app`)
2. **Blueprint variable** (if using routes.py): Must be `{app_name}_bp` (e.g., `my_app_bp`)
3. **Blueprint name** (if using routes.py): Must be `app:{app_name}` (e.g., `"app:my_app"`)
4. **URL prefix**: Convention is `/app/{app_name}` (e.g., `/app/my_app`)

**Index route**: All apps are automatically served at `/app/{app_name}` via a shared handler. You don't need to define an index route in `routes.py`.

See `apps/__init__.py` for discovery logic and `convey/__init__.py` for the shared route handler.

---

## Required Files

### 1. `workspace.html` - Main Content

The workspace template is included inside the app container (`app.html`).

**Available Template Context:**
- `app` - Current app name
- `facets` - List of active facet dicts: `[{name, title, color, emoji}, ...]`
- `selected_facet` - Currently selected facet name (string or None)
- `app_registry` - Registry with all apps (usually not needed directly)
- `state.journal_root` - Path to journal directory
- Any variables passed from route handler via `render_template(...)`

**Note:** The server-side `selected_facet` is also available client-side as `window.selectedFacet` (see JavaScript APIs below).

**Vendor Libraries:**
- Use `{{ vendor_lib('marked') }}` for markdown rendering
- See `convey/static/vendor/VENDOR.md` for available libraries

**Reference implementations:**
- Minimal: `apps/home/workspace.html` (simple content)
- Styled: `apps/dev/workspace.html` (custom CSS, forms, interactive JS)
- Data-driven: `apps/todos/workspace.html` (facet sections, dynamic rendering)

---

## Optional Files

### 2. `routes.py` - Flask Blueprint

Define custom routes for your app (API endpoints, form handlers, navigation routes).

**Key Points:**
- **Not needed for simple apps** - the shared handler at `/app/{app_name}` serves your workspace automatically
- Only create `routes.py` if you need custom routes beyond the index page
- Blueprint variable must be named `{app_name}_bp`
- Blueprint name must be `"app:{app_name}"`
- URL prefix convention: `/app/{app_name}`
- Access journal root via `state.journal_root` (always available)
- Import utilities from `convey.utils` (see [Flask Utilities](#flask-utilities))

**Reference implementations:**
- API endpoints: `apps/search/routes.py` (search APIs, no index route)
- Form handlers: `apps/todos/routes.py` (POST handlers, validation, flash messages)
- Navigation: `apps/calendar/routes.py` (date-based routes with custom context)
- Redirects: `apps/todos/routes.py` index route (redirects `/` to today's date)



### 3. `app.json` - Metadata

Override default icon, label, and facet support.

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `icon` | string | "📦" | Emoji icon for menu bar |
| `label` | string | Title-cased name | Display label in menu |
| `facets` | boolean | `true` | Enable facet integration |

**Defaults:**
- Icon: "📦"
- Label: `app_name.replace("_", " ").title()` (e.g., "my_app" → "My App")
- Facets: `true` (facet pills shown, selection enabled)

**When to disable facets:** Set `"facets": false` for apps that don't use facet-based organization (e.g., system settings, dev tools).

**Examples:** `apps/home/app.json`, `apps/todos/app.json`, `apps/inbox/app.json`

### 4. `hooks.py` - Dynamic Logic

Provide submenu items and facet badge counts that update dynamically.

**Functions:**
- `get_submenu_items(facets, selected_facet)` - Returns list of submenu item dicts
  - Keys: `label` (required), `path` (required), `count` (optional), `facet` (optional)
  - `facet` attribute enables facet selection on click (same as facet pills)
- `get_facet_counts(facets, selected_facet)` - Returns dict mapping facet name to count
  - Used for badge counts on facet pills

**Submenu Items:**
- Appear below your app in the menu bar when expanded
- Optional `count` shows badge next to label
- Optional `facet` attribute enables facet selection on click
- Without `facet`, items navigate directly to `path`

**Facet Counts:**
- Show badge on facet pills in the top bar
- Useful for showing counts per facet (e.g., pending todos, unread messages)

**Reference implementation:** `apps/inbox/hooks.py` - Shows both functions with real badge logic

### 5. `app_bar.html` - Bottom Bar Controls

Fixed bottom bar for forms, buttons, date pickers, search boxes.

**Key Points:**
- App bar is fixed to bottom when present
- Page body gets `has-app-bar` class (adjusts content margin)
- Only rendered when app provides this template
- Great for persistent input controls across views

**Reference implementations:**
- Form input: `apps/todos/app_bar.html` (date navigation, todo input)
- Controls: `apps/tokens/app_bar.html` (action buttons)

### 6. `background.html` - Background Service

JavaScript service that runs globally, even when app is not active.

**AppServices API:**

**Core Methods:**
- `AppServices.register(appName, service)` - Register background service
- `AppServices.updateBadge(appName, facetName, count)` - Update facet badge
- `AppServices.updateSubmenu(appName, items)` - Update submenu items

**Notification Methods:**
- `AppServices.notifications.show(options)` - Show persistent notification card
- `AppServices.notifications.dismiss(id)` - Dismiss specific notification
- `AppServices.notifications.dismissApp(appName)` - Dismiss all for app
- `AppServices.notifications.dismissAll()` - Dismiss all notifications
- `AppServices.notifications.count()` - Get active notification count
- `AppServices.notifications.update(id, options)` - Update existing notification

**Notification Options:**
```javascript
{
  app: 'my_app',          // App name (required)
  icon: '📬',             // Emoji icon (optional)
  title: 'New Message',   // Title (required)
  message: 'You have...', // Message body (optional)
  action: '/app/inbox',   // Click action URL (optional)
  facet: 'work',          // Auto-select facet on click (optional)
  badge: 5,               // Badge count (optional)
  dismissible: true,      // Show X button (default: true)
  autoDismiss: 10000      // Auto-dismiss ms (optional)
}
```

**WebSocket Events (`window.appEvents`):**
- `listen(tract, callback)` - Listen to specific tract ('cortex', 'indexer', 'observe', etc.)
- `listen('*', callback)` - Listen to all events
- Messages have structure: `{tract: 'cortex', event: 'agent_complete', ...data}`
- See **CALLOSUM.md** for event protocol details

**Reference implementation:** `apps/home/background.html` - Shows WebSocket event listening, notification handling, and service pattern

**Implementation source:** `convey/static/app.js` - AppServices framework, `convey/static/websocket.js` - WebSocket API

---

## Flask Utilities

Available in `convey/utils.py`:

### Route Helpers
- `error_response(message, code=400)` - Standard JSON error response
- `success_response(data=None, code=200)` - Standard JSON success response
- `parse_pagination_params(default_limit, max_limit, min_limit)` - Extract and validate limit/offset from request.args

### Date Navigation
- `adjacent_days(journal, day)` - Get previous and next day folders (returns tuple or None)
- `format_date(date_str)` - Format YYYYMMDD as "Wednesday January 14th"

### Agent Spawning
- `spawn_agent(prompt, persona, backend, config)` - Spawn Cortex agent, returns agent_id

### JSON Utilities
- `load_json(path)` - Load JSON file with error handling (returns None on error)
- `save_json(path, data, indent, add_newline)` - Save JSON with formatting (returns bool)

**See source:** `convey/utils.py` for full signatures and documentation

---

## Think Module Integration

Available functions from the `think` module:

### Facets
`think/facets.py`: `get_facets()` - Returns dict of facet configurations

### Todos
`think/todo.py`:
- `get_todos(day, facet)` - Get todo list for day and facet
- `TodoChecklist` class - Load and manipulate todo markdown files

### Messages
`think/messages.py`:
- `get_unread_count()` - Count unread messages
- `get_messages(limit, offset, status)` - Query messages with pagination

### Entities
`think/entities.py`: `get_entities(facet)` - Get entities for facet

See **JOURNAL.md**, **CORTEX.md**, **CALLOSUM.md** for subsystem details.

---

## JavaScript APIs

### Global Variables

Defined in `convey/templates/app.html` (lines 13-17):
- `window.facetsData` - Array of facet objects `[{name, title, color, emoji}, ...]`
- `window.selectedFacet` - Current facet name or null (see Facet Selection below)
- `window.appFacetCounts` - Badge counts for current app `{"work": 5, "personal": 3}`

### Facet Selection

Apps can access and control facet selection through a uniform API:
- `window.selectedFacet` - Current facet name or null (initialized by server, updated on change)
- `window.selectFacet(name)` - Change selection programmatically
- `facet.switch` CustomEvent - Dispatched when selection changes
  - Event detail: `{facet: 'work' or null, facetData: {name, title, color, emoji} or null}`

**Facet Modes:**
- **all-facet mode**: `window.selectedFacet === null`, show content from all facets
- **specific-facet mode**: `window.selectedFacet === "work"`, show only that facet's content
- Selection persisted via cookie, synchronized across pills and submenu items

**See implementation:** `convey/static/app.js` - Facet switching logic and event dispatch

### WebSocket Events

`window.appEvents` API defined in `convey/static/websocket.js`:
- `listen(tract, callback)` - Subscribe to specific tract or '*' for all events
- Messages structure: `{tract: 'cortex', event: 'agent_complete', ...data}`

**Common tracts:** `cortex`, `indexer`, `observe`, `task`

See **CALLOSUM.md** for complete event protocol.

---

## CSS Styling

### CSS Variables

Dynamic variables based on selected facet (update automatically on facet change):

```css
:root {
  --facet-color: #3b82f6;      /* Selected facet color */
  --facet-bg: #3b82f61a;       /* 10% opacity background */
  --facet-border: #3b82f6;     /* Border color */
}
```

Use these in your app-specific styles to respond to facet theme.

### App-Specific Styles

**Best practice:** Scope styles with unique class prefix to avoid conflicts.

**Example:** `apps/dev/workspace.html` (lines 1-145) shows scoped `.dev-*` classes for all custom styles.

### Global Styles

Main stylesheet `convey/static/app.css` provides base components. Review for available classes and patterns.

---

## Common Patterns

### Date-Based Navigation
See `apps/todos/routes.py:todos_day()` - Shows date validation, `adjacent_days()`, `format_date()`, and passing navigation to template.

### AJAX Endpoints
See `apps/todos/routes.py:move_todo()` - Shows JSON parsing, validation, `error_response()`, `success_response()`.

### Form Handling with Flash Messages
See `apps/todos/routes.py:todos_day()` POST handler - Shows form processing, validation, flash messages, redirects.

### Facet-Aware Queries
See `apps/todos/routes.py:todos_day()` - Loads data per-facet when selected, or all facets when null.

---

## Debugging Tips

### Check Discovery

```bash
# Start Convey with debug logging
FLASK_DEBUG=1 convey

# Look for log lines:
# "Discovered app: my_app"
# "Registered blueprint: app:my_app"
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| App not discovered | Missing `workspace.html` | Ensure workspace.html exists |
| Blueprint not found (with routes.py) | Wrong variable name | Use `{app_name}_bp` exactly |
| Import error (with routes.py) | Blueprint name mismatch | Use `"app:{app_name}"` exactly |
| Hyphens in name | Directory uses hyphens | Rename to use underscores |
| Custom routes don't work | URL prefix mismatch | Check `url_prefix` matches pattern |

### Logging

Use `current_app.logger` from Flask for debugging. See `apps/todos/routes.py` for examples.

---

## Best Practices

1. **Use underscores** in directory names (`my_app`, not `my-app`)
2. **Scope CSS** with unique class names to avoid conflicts
3. **Validate input** on all POST endpoints (use `error_response`)
4. **Check facet selection** when loading facet-specific data
5. **Use state.journal_root** for journal path (always available)
6. **Provide hooks** if app has submenu or facet counts
7. **Handle errors gracefully** with flash messages or JSON errors
8. **Test facet switching** to ensure content updates correctly
9. **Use background services** for WebSocket event handling
10. **Follow Flask patterns** for blueprints, url_for, etc.

---

## Example Apps

Study these reference implementations:

- **`apps/home/`** - Minimal app (no routes.py, just workspace + background service)
- **`apps/dev/`** - Simple app (no routes.py, custom styling and notifications)
- **`apps/live/`** - Minimal app (no routes.py, event dashboard)
- **`apps/todos/`** - Full-featured (custom routes with date navigation, forms, AJAX)
- **`apps/inbox/`** - API-driven (custom routes, submenu with badges via hooks)
- **`apps/search/`** - API-only (custom routes for search, no index route)
- **`apps/tokens/`** - Navigation (index redirects to today, app bar with controls)

---

## Additional Resources

- **`apps/__init__.py`** - App discovery and registry implementation
- **`convey/apps.py`** - Context processors and vendor library helper
- **`convey/templates/app.html`** - Main app container template
- **`convey/static/app.js`** - AppServices framework
- **`convey/static/websocket.js`** - WebSocket event system
- **CLAUDE.md** - Project development guidelines and standards
- **JOURNAL.md** - Journal directory structure and data organization
- **CORTEX.md** - Agent system architecture and spawning agents
- **CALLOSUM.md** - Message bus protocol and WebSocket events
- **CRUMBS.md** - Transcript format specification

For Flask documentation, see [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
