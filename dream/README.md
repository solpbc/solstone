# sunstone-dream

Web-based journal review interface built with Flask. It exposes a few small views for exploring daily summaries and entity data stored inside a **journal** folder.

## Installation

```bash
pip install -e .
```

## Usage

Run the server with:

```bash
dream-review --password YOURPASSWORD
```

Open the printed URL in your browser and login with the password.

## Layout

```
dream/
  __init__.py        - application factory and public API
  state.py           - global state shared by views
  utils.py           - helpers for parsing and building indices
  views/             - one module per view
      home.py        - home page and login/logout routes
      entities.py    - entity review UI and related APIs
      calendar.py    - meeting calendar UI and APIs
      chat.py        - simple Gemini chat interface
  templates/         - HTML templates, one file per view
  static/            - shared CSS
```

Each file inside `views/` defines a Flask `Blueprint` with all routes for that page. Front‑end JavaScript lives in the corresponding template file. Views are independent so working on one does not affect the others.

### Adding a view

1. Create `views/<name>.py` defining a `Blueprint`.
2. Add the HTML template under `templates/`.
3. Update `views/__init__.py` to register the new blueprint.
4. Expose any helper functions via `__init__.py` if needed.

This small structure keeps dependencies clear and makes it easy to focus on a single page.

