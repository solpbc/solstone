# Convey Design Language

This document defines the standardized design language and terminology for the Convey web application UI components.

## Design Principles

- **Contextual Prefixes**: Component names use prefixes that indicate their parent container or context (e.g., `menu-`, `facet-`, `app-`)
- **Consistent Naming**: Related components follow the same naming pattern (e.g., `facet-pill`, `facet-badge`, `facet-section`)
- **Clear Hierarchy**: Names reflect the structural relationship between components
- **Plugin Architecture**: Apps are self-contained modules with their own routes, templates, and behavior

---

## App System

Apps are dynamically discovered from the `apps/` directory using **convention over configuration**. Each app is a directory with required and optional files—no base class needed.

**App Directory Structure**
```
apps/{name}/
├── routes.py            # Required: Flask blueprint
├── workspace.html       # Required: Main content template
├── app.json            # Optional: Metadata (icon, label)
├── hooks.py            # Optional: Dynamic logic (submenu, badges)
├── app_bar.html        # Optional: Bottom bar controls
└── service.html        # Optional: Background service (JavaScript)
```

**Naming Conventions**
- App directory: `snake_case` (e.g., `my_app`, not `my-app`)
- Blueprint variable: `{name}_bp` (e.g., `home_bp`, `inbox_bp`)
- Blueprint name: Should match app name for `url_for()` consistency
- URL prefix: `/app/{name}` (e.g., `/app/home`, `/app/inbox`)

**Required Files**

1. **`routes.py`** - Flask blueprint with app routes:
   ```python
   from flask import Blueprint, render_template

   home_bp = Blueprint(
       "home",                    # Blueprint name (matches app name)
       __name__,
       url_prefix="/app/home",    # URL prefix
   )

   @home_bp.route("/")
   def index():
       return render_template("app.html", app="home")
   ```

2. **`workspace.html`** - Main content template (included in `app.html` container)

**Optional Files**

3. **`app.json`** - Metadata overrides (defaults: icon="📦", label=title-cased name):
   ```json
   {
     "icon": "🏠",
     "label": "Home"
   }
   ```

4. **`hooks.py`** - Dynamic behavior for submenu and badges:
   ```python
   def get_submenu_items(facets, selected_facet):
       """Return list of submenu items."""
       return [
           {"label": "Item", "path": "/app/home", "count": 5, "facet": "work"}
       ]

   def get_facet_counts(facets, selected_facet):
       """Return dict of badge counts per facet."""
       return {"work": 3, "personal": 7}
   ```

5. **`app_bar.html`** - Bottom bar controls (forms, buttons, search)

6. **`service.html`** - Background JavaScript service for WebSocket events

**App Discovery**
- Apps are automatically discovered on startup
- Each app is registered as an `App` dataclass in the `AppRegistry`
- Apps automatically receive facet integration (pills, theme, selection state)

**Submenu Integration**
- Submenu items can include `data-facet` attribute for facet-based navigation
- Clicking submenu items with facets uses same selection mechanism as facet pills
- Non-facet submenu items navigate directly (e.g., date ranges, search scopes)

---

## App Services (Background Services)

Apps can register background services that run globally, even when the app is not active. This is similar to iOS background notification handlers.

**Service File Structure**
```
apps/{name}/templates/
└── service.html       # Background service template (JavaScript)
```

**Service Capabilities**
- Listen to WebSocket events (Callosum tracts)
- Update badge counts dynamically
- Show persistent notification cards
- Update submenu items in real-time
- Run custom background logic

**AppServices API**

Core Methods:
- `AppServices.register(appName, service)` - Register background service
- `AppServices.updateBadge(appName, facetName, count)` - Update badge counts
- `AppServices.updateSubmenu(appName, items)` - Update submenu items
- `AppServices.requestNotificationPermission()` - Request browser notifications

Notification System:
- `AppServices.notifications.show(options)` - Show persistent notification card
- `AppServices.notifications.dismiss(id)` - Dismiss specific notification
- `AppServices.notifications.dismissApp(appName)` - Dismiss all for app
- `AppServices.notifications.dismissAll()` - Dismiss all notifications
- `AppServices.notifications.count()` - Get active notification count
- `AppServices.notifications.update(id, options)` - Update notification

**Notification Options**
```javascript
{
  app: 'inbox',           // App name (required)
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

**Notification Cards**
- Persistent until dismissed or clicked
- Stack vertically in top-right (max 5 visible)
- Click card → auto-select facet (if specified) + navigate to action URL
- Click X → dismiss immediately
- Auto-dismiss optional per notification
- Relative timestamps (e.g., "5m ago")
- Also triggers browser notification if permitted
- Facet selection works like submenu navigation (updates cookie, applies theme)

**Service Registration**
Services are automatically loaded and executed on page load. Each service runs in an isolated function scope and can listen to WebSocket events via `window.appEvents`.

**Example Service** (apps/home/templates/service.html)
```javascript
window.AppServices.register('home', {
  initialize() {
    if (!window.appEvents) return;
    window.appEvents.listen('cortex', this.handleCortexEvent.bind(this));
  },

  handleCortexEvent(msg) {
    if (msg.event === 'agent_complete') {
      // Show persistent notification card
      window.AppServices.notifications.show({
        app: 'home',
        icon: '🤖',
        title: 'Agent Complete',
        message: `${msg.agent} finished processing`,
        action: '/app/home',
        autoDismiss: 10000
      });
    }
  }
});
```

---

## Structural Components

### Navigation & Layout

**`menu-bar`**
- Left sidebar navigation containing all apps
- Contains `menu-item` and `submenu-item` components
- Toggles open/closed via hamburger menu

**`menu-item`**
- Individual app row in the menu-bar
- Displays app icon and label
- May contain `submenu` for apps with secondary navigation

**`submenu-item`**
- Secondary navigation within an app's menu section
- Can include `submenu-badge` for counts
- May use `data-facet` attribute for facet selection on click

**`submenu-badge`**
- Count indicator on submenu items
- Shows item counts for that view (e.g., todos per facet)

**`facet-bar`**
- Top horizontal bar with hamburger, app-icon, and facet pills
- Fixed position, responds to selected facet with color theming

**`facet-pill`**
- Individual selectable facet in the facet-bar
- Shows facet emoji and title
- Can display `facet-badge` with app-specific counts
- Collapses to icon-only when space constrained

**`facet-badge`**
- Count indicator on facet pills
- App provides counts via `get_facet_counts()` method

**`facet-section`**
- Collapsible content group representing a facet's items
- Used to organize workspace content by facet

**`app-bar`**
- Bottom fixed bar for app-specific controls
- Only rendered when app provides template via `get_app_bar_template()`
- Contains action inputs, date controls, search boxes, etc.

**`app-icon`**
- Displays currently active app icon in facet-bar
- Clickable to clear facet selection and return to default view

**`status-icon`**
- System status indicator in facet-bar (aligned right)
- Clickable to toggle status-pane visibility
- Visual feedback with hover state

**`status-pane`**
- Dropdown panel anchored below status-icon
- Displays system health and service status information
- Closes on outside click or icon re-click

**`notification-center`**
- Fixed position top-right below facet-bar
- Displays persistent notification cards from apps
- Cards stack vertically (max 5 visible)
- Animated slide-in from right
- Each card shows app icon, title, message, timestamp
- Click card to navigate, click X to dismiss
- Auto-dismiss optional per notification

**`notification-card`**
- Individual notification in notification-center
- Contains header (icon, app name, close button)
- Body with title, message, optional badge
- Footer with relative timestamp
- Hover effect shifts card left slightly
- Dismissible or persistent based on options

**`workspace`**
- Main content area between facet-bar and app-bar
- Renders app-specific workspace template
- Bottom margin adjusts based on app-bar presence

---

## State & Behavior

**Journal Path**
- `state.journal_root` is always available and is the authoritative reference for the journal path
- Use it directly in views without safety checks or environment variable lookups

**Facet Selection**
- Persisted via cookie, synchronized between facet pills and submenu items
- Clicking app-icon clears selection
- Updates CSS variables for theming (color, background, border)
- Single source of truth via `selectFacet()` JavaScript function
- Dispatches `facet.switch` CustomEvent when selection changes
  - Apps can listen: `window.addEventListener('facet.switch', (e) => { ... })`
  - Event detail includes: `{ facet: string|null, facetData: object|null }`

**Active App**
- Highlighted in menu-bar with `.current` class
- Determines workspace content, app-bar, and available submenu

**Responsive Facet Pills**
- Pills collapse to icon-only from right-to-left when space constrained
- Uses ResizeObserver to detect container width changes

---

## Shared Resources

### Vendor Libraries

Third-party JavaScript libraries are organized in `convey/static/vendor/` for centralized access across all apps.

**Directory Structure**:
```
convey/static/vendor/
├── marked/
│   └── marked.min.js         # v15.0.12 - Markdown parser
└── VENDOR.md                 # Library manifest and documentation
```

**Template Helper**:

The `vendor_lib()` helper function provides convenient access to vendor libraries:

```html
<!-- Recommended: Using helper function -->
<script src="{{ vendor_lib('marked') }}"></script>

<!-- Alternative: Explicit path -->
<script src="{{ url_for('static', filename='vendor/marked/marked.min.js') }}"></script>
```

**Available Libraries**:

1. **marked (v15.0.12)** - Markdown parsing and rendering
   - Source: https://github.com/markedjs/marked
   - License: MIT
   - Usage: `{{ vendor_lib('marked') }}`

**Adding New Libraries**:
1. Create subdirectory in `convey/static/vendor/{library_name}/`
2. Add minified `.min.js` file
3. Update `vendor/VENDOR.md` with version and usage documentation
4. Use `{{ vendor_lib('library_name') }}` in templates

See `convey/static/vendor/VENDOR.md` for detailed documentation.

---

## Implementation Status

### Phase 0: Foundation (Complete)

**Infrastructure Created:**
- `apps/__init__.py` - `App` dataclass and `AppRegistry` for convention-based discovery
- `convey/templates/app.html` - Main app container template with all UI components
- Context processors for facet data and app registry injection
- Package configuration in `pyproject.toml` to include apps package
- `convey/static/app.js` - AppServices framework for badges, notifications, and services

**Reference Apps:**
- `apps/home/` - Minimal app with background service example
- `apps/dev/` - Developer utilities app
- Both demonstrate the convention-based pattern
- Both systems (legacy + new) coexist during migration

**Routes:**
- Legacy views: `/`, `/facets`, `/calendar`, etc. (existing `convey/views/`)
- New apps: `/app/{name}/` (plugin system via `apps/`)

**Next Steps:**
- Migrate existing views to apps one at a time
- Each app follows the standardized structure with routes.py and templates
- Legacy views can be deprecated after all apps are migrated
