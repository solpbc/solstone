# Convey Design Language

This document defines the standardized design language and terminology for the Convey web application UI components.

## Design Principles

- **Contextual Prefixes**: Component names use prefixes that indicate their parent container or context (e.g., `menu-`, `facet-`, `app-`)
- **Consistent Naming**: Related components follow the same naming pattern (e.g., `facet-pill`, `facet-badge`, `facet-section`)
- **Clear Hierarchy**: Names reflect the structural relationship between components
- **Plugin Architecture**: Apps are self-contained modules with their own routes, templates, and behavior

---

## App System

Apps are dynamically discovered from the top-level `apps/` directory. Each app is a Python module implementing the `BaseApp` interface.

**App File Structure**
```
apps/{name}/
├── __init__.py              # App class with metadata
├── routes.py                # Flask blueprint and route handlers
└── templates/
    ├── workspace.html       # Main content (required)
    └── app_bar.html         # Bottom bar controls (optional)
```

**App Metadata**
- Each app provides an icon, label, Flask blueprint, and workspace template
- Apps optionally define submenu items, facet counts, and custom app-bar template
- Apps automatically receive facet integration (facet pills and selection state)

**App Methods**
- `get_blueprint()` - Flask routes for the app (from routes.py)
- `get_workspace_template()` - Main content template path (default: relative to blueprint template_folder)
- `get_app_bar_template()` - Optional bottom bar template (return None to hide app-bar)
- `get_submenu_items()` - Optional submenu with custom logic per app
- `get_facet_counts()` - Optional badge counts for facet pills

**Submenu Integration**
- Submenu items can include `data-facet` attribute for facet-based navigation
- Clicking submenu items with facets uses same selection mechanism as facet pills
- Non-facet submenu items navigate directly (e.g., date ranges, search scopes)

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

**`workspace`**
- Main content area between facet-bar and app-bar
- Renders app-specific workspace template
- Bottom margin adjusts based on app-bar presence

---

## State & Behavior

**Facet Selection**
- Persisted via cookie, synchronized between facet pills and submenu items
- Clicking app-icon clears selection
- Updates CSS variables for theming (color, background, border)
- Single source of truth via `selectFacet()` JavaScript function

**Active App**
- Highlighted in menu-bar with `.current` class
- Determines workspace content, app-bar, and available submenu

**Responsive Facet Pills**
- Pills collapse to icon-only from right-to-left when space constrained
- Uses ResizeObserver to detect container width changes

---

## Implementation Status

### Phase 0: Foundation (Complete)

**Infrastructure Created:**
- `apps/__init__.py` - BaseApp class and AppRegistry for plugin discovery
- `convey/templates/app.html` - Main app container template with all UI components
- Context processors for facet data and app registry injection
- Package configuration in `pyproject.toml` to include apps package

**Example App:**
- `apps/home/` - Reference implementation demonstrating the pattern
- Routes at `/app/home/` (new system) vs `/` (legacy system)
- Both systems coexist during migration

**Routes:**
- Legacy views: `/`, `/facets`, `/calendar`, etc. (existing `convey/views/`)
- New apps: `/app/{name}/` (plugin system via `apps/`)

**Next Steps:**
- Migrate existing views to apps one at a time
- Each app follows the standardized structure with routes.py and templates
- Legacy views can be deprecated after all apps are migrated
