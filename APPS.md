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
├── app_bar.html      # Optional: Bottom bar controls (forms, buttons)
└── background.html   # Optional: Background JavaScript service
```

### File Purposes

| File | Required | Purpose |
|------|----------|---------|
| `workspace.html` | **Yes** | Main app content (rendered in container) |
| `routes.py` | No | Flask blueprint for custom routes (API endpoints, forms, etc.) |
| `app.json` | No | Icon, label, facet support overrides |
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

### 4. `app_bar.html` - Bottom Bar Controls

Fixed bottom bar for forms, buttons, date pickers, search boxes.

**Key Points:**
- App bar is fixed to bottom when present
- Page body gets `has-app-bar` class (adjusts content margin)
- Only rendered when app provides this template
- Great for persistent input controls across views

**Date Navigator Component:**

For apps with day-based navigation, include the shared date navigator:

```html
{% include 'date_nav.html' %}
```

This provides a unified `← 🗓️ Today →` control with:
- Previous/next day buttons
- Native date picker (calendar icon)
- Today button (disabled when viewing current day)
- Keyboard shortcuts: ←/→ arrows, `t` for today

The component reads `day` and `app` from template context to construct navigation URLs.

**Reference implementations:**
- Date navigation: `apps/todos/app_bar.html`, `apps/tokens/app_bar.html`, `apps/calendar/app_bar.html`

**Implementation source:** `convey/templates/date_nav.html`

### 5. `background.html` - Background Service

JavaScript service that runs globally, even when app is not active.

**AppServices API:**

**Core Methods:**
- `AppServices.register(appName, service)` - Register background service

**Badge Methods:**

App icon badges (menu bar):
- `AppServices.badges.app.set(appName, count)` - Set app icon badge count
- `AppServices.badges.app.clear(appName)` - Remove app icon badge
- `AppServices.badges.app.get(appName)` - Get current badge count

Facet pill badges (facet bar):
- `AppServices.badges.facet.set(facetName, count)` - Set facet badge count
- `AppServices.badges.facet.clear(facetName)` - Remove facet badge
- `AppServices.badges.facet.get(facetName)` - Get current badge count

Both badge types appear as red notification counts.

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

**Submenu Methods:**
- `AppServices.submenus.set(appName, items)` - Set all submenu items
- `AppServices.submenus.upsert(appName, item)` - Add or update single item
- `AppServices.submenus.remove(appName, itemId)` - Remove item by id
- `AppServices.submenus.clear(appName)` - Clear all items

Submenus appear as hover pop-outs on menu bar icons. Items support `id`, `label`, `icon`, `href`, `facet`, `badge`, and `order` properties.

**See implementation:** `convey/static/app.js` - Submenu rendering and positioning

**WebSocket Events (`window.appEvents`):**
- `listen(tract, callback)` - Listen to specific tract ('cortex', 'indexer', 'observe', etc.)
- `listen('*', callback)` - Listen to all events
- Messages have structure: `{tract: 'cortex', event: 'agent_complete', ...data}`
- See **CALLOSUM.md** for event protocol details

**Reference implementations:**
- `apps/todos/background.html` - App icon badge with API fetch
- `apps/dev/background.html` - Submenu quick-links with dynamic badges

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

### App Storage

Apps can persist journal-specific configuration and data in `<journal>/apps/<app_name>/`:

```python
from apps.utils import get_app_storage_path, load_app_config, save_app_config
```

- `get_app_storage_path(app_name, *sub_dirs, ensure_exists)` - Get Path to app storage directory
- `load_app_config(app_name, default)` - Load app config from `config.json`
- `save_app_config(app_name, config)` - Save app config to `config.json`

**See source:** `apps/utils.py` for implementation details

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
- `window.appFacetCounts` - Badge counts for current app `{"work": 5, "personal": 3}` (set via route's `facet_counts`)

### Facet Selection

Apps can access and control facet selection through a uniform API:
- `window.selectedFacet` - Current facet name or null (initialized by server, updated on change)
- `window.selectFacet(name)` - Change selection programmatically
- `facet.switch` CustomEvent - Dispatched when selection changes
  - Event detail: `{facet: 'work' or null, facetData: {name, title, color, emoji} or null}`

**Facet Modes:**
- **all-facet mode**: `window.selectedFacet === null`, show content from all facets
- **specific-facet mode**: `window.selectedFacet === "work"`, show only that facet's content
- Selection persisted via cookie, synchronized across facet pills

**See implementation:** `convey/static/app.js` - Facet switching logic and event dispatch

### WebSocket Events

`window.appEvents` API defined in `convey/static/websocket.js`:
- `listen(tract, callback)` - Subscribe to specific tract or '*' for all events
- Messages structure: `{tract: 'cortex', event: 'agent_complete', ...data}`

**Common tracts:** `cortex`, `indexer`, `observe`, `task`

See **CALLOSUM.md** for complete event protocol.

---

## CSS Styling

### Workspace Containers

**Always wrap your workspace content** in one of these standardized containers for consistent spacing and layout:

**For readable content** (forms, lists, messages, text):
```html
<div class="workspace-content">
  <!-- Your app content here -->
</div>
```

**For data-heavy content** (tables, grids, calendars):
```html
<div class="workspace-content-wide">
  <!-- Your app content here -->
</div>
```

**Key differences:**
- `.workspace-content` - Centered with 1200px max-width, ideal for readability
- `.workspace-content-wide` - Full viewport width, ideal for data tables and grids
- Both include consistent padding and mobile responsiveness

**See:** `convey/static/app.css` for implementation details

**Examples:**
- Standard: `apps/home/workspace.html`, `apps/todos/workspace.html`, `apps/inbox/workspace.html`
- Wide: `apps/search/workspace.html`, `apps/calendar/_month.html`, `apps/import/workspace.html`

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

### Facet Pill Badges
Pass `facet_counts` dict to `render_template()` to show initial badge counts on facet pills:
```python
facet_counts = {"work": 5, "personal": 3}
return render_template("app.html", app="my_app", facet_counts=facet_counts)
```
For client-side updates (e.g., after completing a todo), use `AppServices.badges.facet.set(facetName, count)`.

See `apps/todos/routes.py:todos_day()` - Computes pending counts from already-loaded data.

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
2. **Wrap workspace content** in `.workspace-content` or `.workspace-content-wide`
3. **Scope CSS** with unique class names to avoid conflicts
4. **Validate input** on all POST endpoints (use `error_response`)
5. **Check facet selection** when loading facet-specific data
6. **Use state.journal_root** for journal path (always available)
7. **Pass facet_counts** from routes if app has per-facet counts
8. **Handle errors gracefully** with flash messages or JSON errors
9. **Test facet switching** to ensure content updates correctly
10. **Use background services** for WebSocket event handling
11. **Follow Flask patterns** for blueprints, url_for, etc.

---

## Example Apps

Study these reference implementations:

- **`apps/home/`** - Minimal app (no routes.py, just workspace)
- **`apps/dev/`** - Simple app (no routes.py, custom styling, submenu)
- **`apps/live/`** - Minimal app (no routes.py, event dashboard)
- **`apps/todos/`** - Full-featured (routes, forms, AJAX, icon badge, facet badges)
- **`apps/inbox/`** - API-driven (custom routes, message management)
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
