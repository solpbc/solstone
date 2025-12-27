# Third-Party Vendor Libraries

This directory contains third-party JavaScript libraries used across Sunstone apps.

## Purpose

The `vendor/` directory provides:
- Centralized location for third-party libraries
- Version tracking and documentation
- Consistent access pattern for all apps
- Local copies for reliability (no CDN dependencies)

## Available Libraries

### marked (v15.0.12)

**Purpose**: Markdown parsing and rendering

**License**: MIT (included in file header)

**Source**: https://github.com/markedjs/marked

**CDN Alternative**: `https://cdn.jsdelivr.net/npm/marked/marked.min.js`

**Usage in App Templates**:
```html
<!-- Using helper function (recommended) -->
<script src="{{ vendor_lib('marked') }}"></script>

<!-- Using explicit path -->
<script src="{{ url_for('static', filename='vendor/marked/marked.min.js') }}"></script>
```

**Example**:
```javascript
// Basic markdown rendering
const html = marked.parse('# Hello World');

// With options
const html = marked.parse(markdown, {
  breaks: true,      // Convert \n to <br>
  gfm: true,        // GitHub Flavored Markdown
  headerIds: false, // Disable auto-generated header IDs
  mangle: false     // Disable email mangling
});
```

**Currently Used By** (legacy references):
- `convey/templates/chat.html` (via `convey/static/marked.min.js`)
- `convey/templates/facet_detail.html` (via CDN)
- `convey/templates/agents.html` (via CDN)

## Adding New Libraries

When adding a new third-party library:

1. **Create subdirectory**: `vendor/{library_name}/`
2. **Add minified file**: Copy production-ready `.min.js` file
3. **Check license**: Ensure license is MIT-compatible and included
4. **Update this manifest**: Add entry with version, purpose, and usage
5. **Test**: Verify library loads and works in development

## Updating Libraries

When updating a library version:

1. **Replace file** in vendor directory
2. **Update version** in this manifest
3. **Test all apps**: Check apps listed in "Currently Used By"
4. **Commit**: Use message format: `chore: update {library} to v{version}`

## Guidelines

- **Prefer local copies** over CDN for reliability and offline development
- **Use minified versions** for production-ready performance
- **Include licenses** either in file headers or separate LICENSE files
- **Document usage patterns** in this manifest
- **Track versions** to enable security updates
- **One library per subdirectory** for clean organization
