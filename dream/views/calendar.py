from __future__ import annotations

import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from .. import state
from ..utils import (
    DATE_RE,
    adjacent_days,
    build_occurrence_index,
    format_date,
    list_day_folders,
)

bp = Blueprint("calendar", __name__, template_folder="../templates")


@bp.route("/calendar")
def calendar_page() -> str:
    return render_template("calendar.html", active="calendar")


@bp.route("/calendar/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    day_dir = os.path.join(state.journal_root, day)
    if not os.path.isdir(day_dir):
        return "", 404
    from think.utils import get_topics

    topics = get_topics()
    files = []
    topics_dir = os.path.join(day_dir, "topics")
    if os.path.isdir(topics_dir):
        for name in sorted(os.listdir(topics_dir)):
            base, ext = os.path.splitext(name)
            if ext != ".md" or base not in topics:
                continue
            path = os.path.join(topics_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            try:
                import markdown  # type: ignore

                html = markdown.markdown(text, extensions=["extra"])
            except Exception:
                html = "<p>Error loading file.</p>"
            label = base.replace("_", " ").title()
            files.append(
                {
                    "label": label,
                    "html": html,
                    "topic": base,
                    "color": topics[base]["color"],
                }
            )
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "calendar_day.html",
        active="calendar",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
        day=day,
    )


@bp.route("/calendar/<day>/transcript")
def calendar_transcript_page(day: str) -> str:
    """Render transcript viewer for a specific day."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "calendar_transcript.html",
        active="calendar",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@bp.route("/calendar/<day>/todos", methods=["GET", "POST"])
def calendar_todos_page(day: str) -> Any:
    """Render TODO viewer or handle TODO updates for a specific day."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import get_todos

    # Handle POST requests for TODO updates
    if request.method == "POST":
        from pathlib import Path

        data = request.get_json()
        action = data.get("action")

        day_dir = Path(os.path.join(state.journal_root, day))
        todo_path = day_dir / "TODO.md"

        # Create directory if it doesn't exist
        day_dir.mkdir(exist_ok=True)

        if action == "add":
            section = data.get("section", "today")
            text = data.get("text", "").strip()

            if not text:
                return jsonify({"success": False, "error": "Empty text"}), 400

            # Parse the text for type prefix (either **Type**: prefix or Type: prefix)
            type_match_bold = re.match(r"^\*\*(\w+)\*\*:\s*(.+)", text)
            type_match_plain = re.match(r"^(\w+):\s*(.+)", text)
            if type_match_bold:
                todo_type = type_match_bold.group(1).capitalize()
                description = type_match_bold.group(2)
                has_type = True
            elif type_match_plain:
                todo_type = type_match_plain.group(1).capitalize()
                description = type_match_plain.group(2)
                has_type = True
            else:
                has_type = False
                description = text

            # Get current time for new items
            from datetime import datetime

            current_time = datetime.now().strftime("%H:%M")

            # Format the new line
            if has_type:
                if section == "today":
                    new_line = (
                        f"- [ ] **{todo_type}**: {description} ({current_time})\n"
                    )
                else:  # future
                    new_line = f"- [ ] **{todo_type}**: {description}\n"
            else:
                if section == "today":
                    new_line = f"- [ ] {description} ({current_time})\n"
                else:  # future
                    new_line = f"- [ ] {description}\n"

            # Read existing content or create new
            if todo_path.exists():
                content = todo_path.read_text()
            else:
                content = "# Today\n\n# Future\n"

            # Find the section and append
            lines = content.splitlines(keepends=True)
            new_lines = []
            in_section = False
            added = False

            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.strip() == f"# {section.capitalize()}":
                    in_section = True
                elif in_section and not added:
                    # Add before next section or at end of current section
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("#"):
                        new_lines.insert(-1, new_line)
                        added = True
                    elif i == len(lines) - 1:
                        new_lines.append(new_line)
                        added = True

            if not added:
                # Section might be at the end
                new_lines.append(new_line)

            todo_path.write_text("".join(new_lines))
            return jsonify({"success": True})

        elif action == "update":
            line_number = data.get("line_number")  # 1-based line number
            field = data.get("field")
            value = data.get("value")

            if not todo_path.exists():
                return jsonify({"success": False, "error": "TODO.md not found"}), 404

            if not line_number or line_number < 1:
                return jsonify({"success": False, "error": "Invalid line number"}), 400

            content = todo_path.read_text()
            lines = content.splitlines(keepends=True)

            # Validate line number is within bounds
            if line_number > len(lines):
                return (
                    jsonify({"success": False, "error": "Line number out of range"}),
                    400,
                )

            # Get the line to update (convert to 0-based index)
            line_index = line_number - 1
            line = lines[line_index]

            # Ensure it's a todo line
            if not line.strip().startswith("- ["):
                return jsonify({"success": False, "error": "Not a todo line"}), 400

            # Get current time for timestamp update
            from datetime import datetime

            current_time = datetime.now().strftime("%H:%M")

            # Determine if this is a "today" item by checking section
            # Find which section this line belongs to
            is_today_item = False
            for i in range(line_index, -1, -1):
                if lines[i].strip() == "# Today":
                    is_today_item = True
                    break
                elif lines[i].strip() == "# Future":
                    is_today_item = False
                    break

            # Check if timestamp already exists
            has_timestamp = re.search(r"\(\d{1,2}:\d{2}\)\s*$", line) is not None

            # For 'today' items with timestamps, remove old timestamp before any modification
            if has_timestamp and field != "cancelled":
                lines[line_index] = (
                    re.sub(r"\s*\(\d{1,2}:\d{2}\)\s*$", "", lines[line_index].rstrip())
                    + "\n"
                )

            # Apply the field update
            if field == "completed":
                if value:
                    lines[line_index] = lines[line_index].replace("- [ ]", "- [x]", 1)
                else:
                    lines[line_index] = lines[line_index].replace("- [x]", "- [ ]", 1)
            elif field == "cancelled":
                # Toggle strikethrough
                if value:
                    # Add strikethrough after checkbox
                    match = re.match(r"(- \[.\] )(.*)", lines[line_index])
                    if match and not match.group(2).startswith("~~"):
                        lines[line_index] = (
                            match.group(1) + "~~" + match.group(2).rstrip() + "~~\n"
                        )
                else:
                    # Remove strikethrough
                    lines[line_index] = re.sub(
                        r"(- \[.\] )~~(.+?)~~", r"\1\2", lines[line_index]
                    )
            elif field == "text":
                # Update the entire text after checkbox
                match = re.match(r"(- \[.\] )(.*)", lines[line_index])
                if match:
                    lines[line_index] = match.group(1) + value + "\n"
            elif field == "date":
                # Update just the date for future items
                match = re.match(r"(- \[.\] )(.*)", lines[line_index])
                if match:
                    content = match.group(2).rstrip()
                    # Remove any existing date (MM/DD/YYYY format) at the end
                    content = re.sub(r"\s+\d{1,2}/\d{1,2}/\d{4}\s*$", "", content)
                    # Add the new date
                    lines[line_index] = match.group(1) + content + " " + value + "\n"

            # Re-add timestamp for 'today' items (unless cancelled or date update)
            if is_today_item and field not in ["cancelled", "date"]:
                lines[line_index] = lines[line_index].rstrip() + f" ({current_time})\n"

            todo_path.write_text("".join(lines))
            return jsonify({"success": True})

        return jsonify({"success": False, "error": "Invalid action"}), 400

    # GET request - render the page
    todos = get_todos(day)
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "calendar_todos.html",
        active="calendar",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
        todos=todos,
    )


@bp.route("/calendar/<day>/todos/generate", methods=["POST"])
def generate_todos(day: str) -> Any:
    """Generate TODO list for a specific day using the TODO persona."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    # Import cortex client
    from ..cortex_client import get_global_cortex_client

    client = get_global_cortex_client()
    if not client:
        return jsonify({"error": "Cortex service not available"}), 503

    # Get yesterday's TODO if it exists
    from datetime import datetime, timedelta
    from pathlib import Path

    day_date = datetime.strptime(day, "%Y%m%d")
    yesterday = (day_date - timedelta(days=1)).strftime("%Y%m%d")
    yesterday_path = Path(state.journal_root) / yesterday / "TODO.md"

    yesterday_content = ""
    if yesterday_path.exists():
        try:
            yesterday_content = yesterday_path.read_text()
        except Exception:
            pass

    # Prepare the prompt for the TODO persona
    prompt = f"""Generate a TODO list for {day_date.strftime('%Y-%m-%d')}.

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Target day: {day_date.strftime('%Y-%m-%d')}
Target day folder: {day}

Yesterday's TODO.md content:
{yesterday_content if yesterday_content else "(No TODO.md from yesterday)"}

Please analyze recent journal entries and generate an appropriate TODO.md file.
Write the generated TODO list to the file at: {day}/TODO.md"""

    # Spawn agent with TODO persona
    agent_id = client.spawn_agent(
        prompt=prompt,
        persona="todo",
        backend="openai",  # Use configured backend
        config={},  # Use default config for persona
    )

    if not agent_id:
        return jsonify({"error": "Failed to spawn agent"}), 500

    # Store agent ID for this day's generation (in memory for now)
    if not hasattr(state, "todo_generation_agents"):
        state.todo_generation_agents = {}
    state.todo_generation_agents[day] = agent_id

    return jsonify({"agent_id": agent_id, "status": "started"})


@bp.route("/calendar/<day>/todos/generation-status")
def todo_generation_status(day: str) -> Any:
    """Check the status of TODO generation for a specific day."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    # Check for specific agent_id in query params
    agent_id = request.args.get("agent_id")

    # Or get from state if tracking
    if not agent_id and hasattr(state, "todo_generation_agents"):
        agent_id = state.todo_generation_agents.get(day)

    if not agent_id:
        return jsonify({"status": "none", "agent_id": None})

    # Check agent status
    import os

    from ..cortex_client import get_global_cortex_client

    # First check if agent exists in journal (finished)
    agents_dir = os.path.join(state.journal_root, "agents")
    agent_file = os.path.join(agents_dir, f"{agent_id}.jsonl")

    if os.path.exists(agent_file):
        # Agent has finished, check if TODO was created
        from pathlib import Path

        todo_path = Path(state.journal_root) / day / "TODO.md"

        if todo_path.exists():
            # Clear from state tracking
            if (
                hasattr(state, "todo_generation_agents")
                and day in state.todo_generation_agents
            ):
                del state.todo_generation_agents[day]
            return jsonify(
                {"status": "finished", "agent_id": agent_id, "todo_created": True}
            )
        else:
            return jsonify(
                {"status": "finished", "agent_id": agent_id, "todo_created": False}
            )

    # Check if still running via cortex
    client = get_global_cortex_client()
    if client:
        response = client.list_agents(limit=100, offset=0)
        if response:
            agents = response.get("agents", [])
            for agent in agents:
                if agent.get("id") == agent_id:
                    return jsonify({"status": "running", "agent_id": agent_id})
            # Agent not in running list, likely finished or errored
            return jsonify({"status": "unknown", "agent_id": agent_id})

    # No cortex client available, can't check status
    return jsonify({"status": "unknown", "agent_id": agent_id})


@bp.route("/calendar/api/transcript_ranges/<day>")
def calendar_transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for ``day``."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.cluster import cluster_scan

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@bp.route("/calendar/api/transcript/<day>")
def calendar_transcript_range(day: str) -> Any:
    """Return transcript markdown HTML for the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    # Get checkbox states from query params
    audio_enabled = request.args.get("audio", "true").lower() == "true"
    screen_enabled = request.args.get("screen", "true").lower() == "true"

    if not audio_enabled and not screen_enabled:
        markdown_text = "*Please select at least one source (Audio or Screen)*"
    else:
        from think.cluster import cluster_range

        # Call cluster_range with appropriate parameters based on checkboxes
        if audio_enabled and screen_enabled:
            # Both sources selected - get full content
            markdown_text = cluster_range(day, start, end, audio=True, screen="summary")
        elif audio_enabled and not screen_enabled:
            # Only audio selected - custom logic to exclude screen content
            from datetime import datetime

            from think.cluster import (
                _date_str,
                _group_entries,
                _groups_to_markdown,
                _load_entries,
                day_path,
            )

            day_dir = day_path(day)
            date_str = _date_str(day_dir)
            start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
            end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

            # Load with audio=True and screen="raw" to get entries
            entries = _load_entries(day_dir, True, "raw")
            # Filter to only audio entries within time range
            entries = [
                e
                for e in entries
                if e.get("prefix") == "audio" and start_dt <= e["timestamp"] < end_dt
            ]
            groups = _group_entries(entries)
            markdown_text = _groups_to_markdown(groups)
        elif not audio_enabled and screen_enabled:
            # Only screen selected - get screen without audio
            markdown_text = cluster_range(
                day, start, end, audio=False, screen="summary"
            )
        else:
            # This case is already handled above
            markdown_text = ""
    try:
        import markdown  # type: ignore

        html_output = markdown.markdown(markdown_text, extensions=["extra"])
    except Exception:  # pragma: no cover - fallback
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"
    return jsonify({"html": html_output})


@bp.route("/calendar/api/raw_files/<day>")
def calendar_raw_files(day: str) -> Any:
    """Return raw file timestamps for the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    file_type = request.args.get("type", None)  # 'audio', 'screen', or None for both

    from datetime import datetime

    from think.cluster import _date_str, _load_entries, day_path

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Load entries based on type filter
    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:  # Load both
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    # Filter to time range and extract timestamps
    files = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            # Convert timestamp to minutes since midnight for easier rendering
            minutes = e["timestamp"].hour * 60 + e["timestamp"].minute
            # For screen files, prefix could be "screen" (summary) or source name (raw)
            file_type_str = "audio" if e["prefix"] == "audio" else "screen"
            files.append(
                {
                    "minute": minutes,
                    "type": file_type_str,
                    "time": e["timestamp"].strftime("%H:%M:%S"),
                }
            )

    return jsonify({"files": files})


@bp.route("/calendar/api/media_files/<day>")
def calendar_media_files(day: str) -> Any:
    """Return actual media files for embedding in the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    file_type = request.args.get("type", None)  # 'audio', 'screen', or None for both

    from datetime import datetime

    from think.cluster import _date_str, _load_entries, day_path
    from think.utils import get_raw_file

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        return jsonify({"media": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Load entries based on type filter
    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:  # Load both
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    # Filter to time range and get raw file info
    media = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            # Get the raw file info using get_raw_file
            try:
                rel_path, mime_type, metadata = get_raw_file(day, e["name"])

                # Create a URL path for serving the file
                file_url = (
                    f"/calendar/api/serve_file/{day}/{rel_path.replace('/', '__')}"
                )

                # For screen files, prefix could be "screen" (summary) or source name (raw)
                file_type_str = "audio" if e["prefix"] == "audio" else "screen"
                human_time = e["timestamp"].strftime("%I:%M:%S %p").lstrip("0")

                media.append(
                    {
                        "url": file_url,
                        "type": file_type_str,
                        "mime_type": mime_type,
                        "time": e["timestamp"].strftime("%H:%M:%S"),
                        "human_time": human_time,
                        "timestamp": e["timestamp"].isoformat(),
                        "metadata": metadata,
                    }
                )
            except Exception:
                # Skip files that can't be processed
                continue

    # Sort by timestamp
    media.sort(key=lambda x: x["timestamp"])

    return jsonify({"media": media})


@bp.route("/calendar/api/serve_file/<day>/<path:encoded_path>")
def serve_media_file(day: str, encoded_path: str) -> Any:
    """Serve actual media files for embedding."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    try:
        # Decode the path (replace '__' back to '/')
        rel_path = encoded_path.replace("__", "/")

        # Construct the full file path
        full_path = os.path.join(state.journal_root, day, rel_path)

        # Security check: ensure the path is within the day directory
        day_dir = os.path.join(state.journal_root, day)
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return "", 403

        # Check if file exists
        if not os.path.isfile(full_path):
            return "", 404

        from flask import send_file

        return send_file(full_path)

    except Exception:
        return "", 404


@bp.route("/calendar/api/occurrences")
def calendar_occurrences() -> Any:
    if not state.occurrences_index and state.journal_root:
        state.occurrences_index = build_occurrence_index(state.journal_root)
    return jsonify(state.occurrences_index)


@bp.route("/calendar/api/days")
def calendar_days() -> Any:
    """Return list of available day folders."""

    days = list_day_folders(state.journal_root)
    return jsonify(days)
