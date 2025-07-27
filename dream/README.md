# sunstone-dream

Web-based journal review interface built with Flask. It exposes a few small views for exploring daily summaries and entity data stored inside a **journal** folder.

## Installation

```bash
pip install -e .
```

## Usage

Run the server with:

```bash
dream --password YOURPASSWORD
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
      chat.py        - simple chat interface for Google or OpenAI LLMs
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

## Links and navigation

- `/calendar` shows the month view. Clicking a day opens `/calendar/YYYYMMDD`.
- Individual markdown files for a day are displayed as tabs. Each tab can be
  linked directly via an anchor, e.g. `/calendar/20240101#meetings`.
- Events in the calendar hour view modal also link to these anchors so you can
  jump straight from an event to the relevant markdown section.


### Admin view

Navigate to `/admin` for index and reindex options. A specific day can be opened at `/admin/YYYYMMDD` to run repairs, ponder prompts, entity roll and screen reduction.
Holding **Shift** while clicking a task button runs it with `--force` if the command supports that flag.

### Tasks view

The `/tasks` page lists all running tasks and recently completed ones. Logs are
cached under `JOURNAL_PATH/tasks` with one `<id>.json` metadata file and a
matching `<id>.jsonl` stream of log messages. Use the
"Log" button to follow live output or "Kill" to stop a running task. The
"Clear Old" button removes cached entries older than seven days.
