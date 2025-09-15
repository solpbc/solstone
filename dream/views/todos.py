from __future__ import annotations

import json
import os
import re
from typing import Any
from datetime import date

from flask import Blueprint, jsonify, render_template, request, redirect, url_for

from .. import state
from ..utils import (
    DATE_RE,
    adjacent_days,
    format_date,
)

bp = Blueprint("todos", __name__, template_folder="../templates")


@bp.route("/todos")
def todos_page() -> Any:
    """Redirect to today's todos."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for('todos.todos_day', day=today))


@bp.route("/todos/<day>", methods=["GET", "POST"])
def todos_day(day: str) -> Any:
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
        active="todos",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
        todos=todos,
    )


@bp.route("/todos/<day>/generate", methods=["POST"])
def generate_todos(day: str) -> Any:
    """Generate TODO list for a specific day using the TODO persona."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    # Import cortex functions
    # Get yesterday's TODO if it exists
    from datetime import datetime, timedelta
    from pathlib import Path

    from think.cortex_client import cortex_request

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

    # Spawn agent with TODO persona using cortex_request
    try:
        active_file = cortex_request(
            prompt=prompt,
            persona="todo",
            backend="openai",  # Use configured backend
            config={},  # Use default config for persona
        )
        # Extract agent ID from filename
        from pathlib import Path as PathLib

        agent_id = PathLib(active_file).stem.replace("_active", "")
    except Exception as e:
        return jsonify({"error": f"Failed to spawn agent: {str(e)}"}), 500

    # Store agent ID for this day's generation (in memory for now)
    if not hasattr(state, "todo_generation_agents"):
        state.todo_generation_agents = {}
    state.todo_generation_agents[day] = agent_id

    return jsonify({"agent_id": agent_id, "status": "started"})


@bp.route("/todos/<day>/generation-status")
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
    from pathlib import Path

    from think.cortex_client import cortex_agents

    # First check if agent exists in journal (finished)
    agents_dir = os.path.join(state.journal_root, "agents")
    agent_file = os.path.join(agents_dir, f"{agent_id}.jsonl")

    if os.path.exists(agent_file):
        # Agent has finished, check if TODO was created
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
    try:
        response = cortex_agents(limit=100, offset=0)
        if response:
            agents = response.get("agents", [])
            for agent in agents:
                if agent.get("id") == agent_id:
                    return jsonify({"status": "running", "agent_id": agent_id})
            # Agent not in running list, likely finished or errored
            return jsonify({"status": "unknown", "agent_id": agent_id})
    except Exception:
        pass

    # Can't check status
    return jsonify({"status": "unknown", "agent_id": agent_id})