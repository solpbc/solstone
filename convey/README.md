# sunstone-convey

Web-based journal review interface built with Flask. It exposes a few small views for exploring daily summaries and entity data stored inside a **journal** folder.

## Installation

```bash
pip install -e .
```

## Usage

Run the server with:

```bash
convey
```

### Authentication

Password authentication is configured through the journal config at `config/journal.json`:

```json
{
  "convey": {
    "password": "your-password-here"
  }
}
```

A password must be configured to use the application. If no password is set, the login page will display an error with configuration instructions.

## Architecture

Convey uses an **app plugin system** where all functional views are implemented as independent apps in the `/apps/` directory. The core `convey/` package provides authentication, WebSocket communication, and the app loading infrastructure.

```
convey/
  __init__.py        - Flask app factory, app registry, context processors
  state.py           - global state (journal_root, chat_backend)
  bridge.py          - Callosum WebSocket bridge for real-time events
  utils.py           - shared helpers (format_date, spawn_agent, etc.)
  screenshot.py      - screenshot utility for testing
  views/
      __init__.py    - blueprint registration
      home.py        - authentication (login/logout) and root redirect
  templates/
      app.html       - main app container template
      menu_bar.html  - dynamic left sidebar menu
      status_pane.html - WebSocket status indicator
      login.html     - login page
      macros.html    - Jinja macros
  static/            - shared CSS and JavaScript
      app.css        - app system styles
      app.js         - facet pills, services, notification center
      websocket.js   - WebSocket connection handler
      error-handler.js - global error handling
      colors.js      - color palette
      vendor/        - third-party libraries (marked.js)

apps/                - App plugin directory (see apps/README.md)
  {app_name}/
    app.json         - metadata (icon, label)
    routes.py        - Flask blueprint with routes
    workspace.html   - main UI template
    background.html  - (optional) background service script
```

### App System

All functional views are implemented as apps in `/apps/`. Each app:
- Has its own directory with `app.json`, `routes.py`, and `workspace.html`
- Uses blueprint name `app:{name}` with URL prefix `/app/{name}/`
- Is automatically discovered and registered by `AppRegistry`
- Can provide facet-scoped views and background services

**Available apps:** home, todos, inbox, chat, agents, search, calendar, entities, news, stats, tokens, settings, import, live, dev

### Core Routes

The `convey/views/home.py` module provides essential routes:

- `/` - Redirects to `/app/home/`
- `/login` - Authentication page
- `/logout` - Clear session and redirect to login
- `/favicon.ico` - Serve favicon

All functional views are accessed at `/app/{name}/` URLs.

### Adding a New App

See `/apps/README.md` for detailed instructions on creating new apps in the plugin system.
