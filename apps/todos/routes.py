# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from apps.todos.todo import (
    TodoChecklist,
    TodoEmptyTextError,
    TodoLineNumberError,
    get_todos,
)
from apps.utils import log_app_action
from convey import state
from convey.config import get_selected_facet
from convey.utils import DATE_RE, format_date
from think.facets import get_facets

todos_bp = Blueprint("app:todos", __name__, url_prefix="/app/todos")


def _compute_badge_counts(day: str, facet: str) -> dict:
    """Compute badge counts for a specific facet and total for today.

    Returns dict with 'facet_count' and 'total_count'.
    Excludes cancelled todos from counts.
    """
    today = date.today().strftime("%Y%m%d")

    # Get count for the specific facet (exclude cancelled)
    facet_todos = get_todos(day, facet)
    facet_count = 0
    if facet_todos:
        facet_count = sum(
            1 for t in facet_todos if not t.get("completed") and not t.get("cancelled")
        )

    # Get total count across all facets for today (for app icon badge)
    total_count = 0
    if day == today:
        try:
            facet_map = get_facets()
        except Exception:
            facet_map = {}

        for facet_name in facet_map.keys():
            todos = get_todos(today, facet_name)
            if todos:
                total_count += sum(
                    1
                    for t in todos
                    if not t.get("completed") and not t.get("cancelled")
                )

    return {"facet_count": facet_count, "total_count": total_count}


@todos_bp.route("/api/badge-count")
def badge_count():
    """Get total pending todo count for today across all facets."""
    today = date.today().strftime("%Y%m%d")
    total = 0

    try:
        facet_map = get_facets()
    except Exception:
        facet_map = {}

    for facet_name in facet_map.keys():
        facet_todos = get_todos(today, facet_name)
        if facet_todos:
            total += sum(
                1
                for todo in facet_todos
                if not todo.get("completed") and not todo.get("cancelled")
            )

    return jsonify({"count": total})


@todos_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return todo counts per facet for a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to facet counts dict.
        Count is number of non-cancelled todos for that day.
    """
    import re

    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    try:
        facet_map = get_facets()
    except Exception:
        facet_map = {}

    stats: dict[str, dict[str, int]] = {}
    journal_root = Path(state.journal_root)

    for facet_name in facet_map.keys():
        todos_dir = journal_root / "facets" / facet_name / "todos"
        if not todos_dir.exists():
            continue

        for todo_file in todos_dir.glob(f"{month}*.jsonl"):
            day = todo_file.stem
            if not DATE_RE.fullmatch(day):
                continue

            # Count non-cancelled todos in file
            facet_todos = get_todos(day, facet_name)
            if facet_todos:
                count = sum(1 for t in facet_todos if not t.get("cancelled"))
                if count > 0:
                    if day not in stats:
                        stats[day] = {}
                    stats[day][facet_name] = count

    return jsonify(stats)


def _todo_path(day: str, facet: str) -> Path:
    return Path(state.journal_root) / "facets" / facet / "todos" / f"{day}.jsonl"


@todos_bp.route("/")
def todos_page() -> str:
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:todos.todos_day", day=today))


@todos_bp.route("/<day>", methods=["GET", "POST"])
def todos_day(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    if request.method == "POST":
        action = request.form.get("action")

        if action == "add":
            text = request.form.get("text", "").strip()
            is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
            error_message = None

            if not text:
                error_message = "Cannot add an empty todo"
            else:
                # Extract facet from hashtag (e.g., "#work" -> "work")
                import re

                facet_match = re.search(r"#([a-z][a-z0-9_-]*)", text, re.IGNORECASE)
                if facet_match:
                    facet = facet_match.group(1).lower()
                    # Remove the hashtag from the text
                    text = re.sub(
                        r"\s*#" + re.escape(facet_match.group(1)) + r"\b",
                        "",
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()

                    # Validate facet exists
                    try:
                        facet_map = get_facets()
                    except Exception:
                        facet_map = {}

                    if facet not in facet_map:
                        error_message = f"Facet #{facet} does not exist"
                else:
                    # Use selected facet as default, fall back to personal
                    selected = get_selected_facet()
                    facet = selected if selected else "personal"

                if not error_message and not text:
                    error_message = "Cannot add an empty todo"

                if not error_message:
                    try:
                        checklist = TodoChecklist.load(day, facet)
                        item = checklist.append_entry(text)

                        log_app_action(
                            app="todos",
                            facet=facet,
                            action="todo_add",
                            params={"text": item.text, "line_number": item.index},
                            day=day,
                        )

                        # If AJAX request, return JSON with new todo data
                        if is_ajax:
                            counts = _compute_badge_counts(day, facet)
                            return jsonify(
                                {
                                    "status": "ok",
                                    "todo": {
                                        "facet": facet,
                                        "index": item.index,
                                        "text": item.text,
                                        "time": item.time,
                                        "completed": False,
                                    },
                                    **counts,
                                }
                            )
                    except (TodoEmptyTextError, RuntimeError) as exc:
                        current_app.logger.debug(
                            "Failed to append todo for %s/%s: %s", facet, day, exc
                        )
                        error_message = "Unable to add todo right now"

            # Handle errors
            if error_message:
                if is_ajax:
                    return (
                        jsonify({"status": "error", "message": error_message}),
                        400,
                    )
                flash(error_message, "error")

            return redirect(url_for("app:todos.todos_day", day=day))

        # Get facet and index for other actions
        facet = request.form.get("facet", "personal")
        index_str = request.form.get("index")

        try:
            index = int(index_str) if index_str else None
        except ValueError:
            index = None

        if not index:
            flash("Missing todo index", "error")
            return redirect(url_for("app:todos.todos_day", day=day))

        try:
            checklist = TodoChecklist.load(day, facet)
        except RuntimeError as exc:
            current_app.logger.debug(
                "Failed to load checklist for %s/%s: %s", facet, day, exc
            )
            flash("Todo list changed, please refresh and try again", "error")
            return redirect(url_for("app:todos.todos_day", day=day))

        try:
            if action == "complete":
                item = checklist.mark_done(index)
                log_app_action(
                    app="todos",
                    facet=facet,
                    action="todo_complete",
                    params={"line_number": index, "text": item.text},
                    day=day,
                )
            elif action == "uncomplete":
                item = checklist.mark_undone(index)
                log_app_action(
                    app="todos",
                    facet=facet,
                    action="todo_uncomplete",
                    params={"line_number": index, "text": item.text},
                    day=day,
                )
            elif action == "cancel":
                item = checklist.cancel_entry(index)
                log_app_action(
                    app="todos",
                    facet=facet,
                    action="todo_cancel",
                    params={"line_number": index, "text": item.text},
                    day=day,
                )
            elif action == "edit":
                import re

                text = request.form.get("text", "").strip()

                # Check if text contains a facet hashtag
                facet_match = re.search(r"#([a-z][a-z0-9_-]*)", text, re.IGNORECASE)
                if facet_match:
                    new_facet = facet_match.group(1).lower()
                    # Remove the hashtag from the text
                    text = re.sub(
                        r"\s*#" + re.escape(facet_match.group(1)) + r"\b",
                        "",
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()

                    # Validate new facet exists
                    try:
                        facet_map = get_facets()
                    except Exception:
                        facet_map = {}

                    if new_facet not in facet_map:
                        flash(f"Facet #{new_facet} does not exist", "error")
                        return redirect(url_for("app:todos.todos_day", day=day))

                    # If facet changed, move the todo (cancel source, add to target)
                    if new_facet != facet:
                        source_item = checklist.get_item(index)
                        old_text = source_item.text

                        # Add to new facet, preserving original created_at
                        new_checklist = TodoChecklist.load(day, new_facet)
                        new_item = new_checklist.append_entry(
                            text, created_at=source_item.created_at
                        )

                        # Preserve completed status
                        if source_item.completed:
                            new_checklist.mark_done(new_item.index)

                        # Cancel from old facet
                        checklist.cancel_entry(index)

                        log_app_action(
                            app="todos",
                            facet=facet,
                            action="todo_edit",
                            params={
                                "line_number": index,
                                "old_text": old_text,
                                "new_text": text,
                                "old_facet": facet,
                                "new_facet": new_facet,
                            },
                            day=day,
                        )

                        return redirect(url_for("app:todos.todos_day", day=day))

                # No facet change, just update text
                old_text = checklist.get_item(index).text
                checklist.update_entry_text(index, text)
                log_app_action(
                    app="todos",
                    facet=facet,
                    action="todo_edit",
                    params={
                        "line_number": index,
                        "old_text": old_text,
                        "new_text": text,
                    },
                    day=day,
                )
            else:
                flash("Unknown action", "error")
                return redirect(url_for("app:todos.todos_day", day=day))
        except TodoEmptyTextError:
            flash("Cannot update todo to empty text", "error")
        except (TodoLineNumberError, IndexError):
            flash("Todo list changed, please refresh and try again", "error")

        # If AJAX request, return JSON with updated counts
        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.accept_mimetypes.accept_json
        ):
            counts = _compute_badge_counts(day, facet)
            return jsonify({"status": "ok", **counts})

        return redirect(url_for("app:todos.todos_day", day=day))

    # Load todos from all facets
    try:
        facet_map = get_facets()
    except Exception as exc:  # pragma: no cover - metadata is optional
        current_app.logger.debug("Failed to load facet metadata: %s", exc)
        facet_map = {}

    # Collect todos from each facet (excluding cancelled, including empty facets)
    todos_by_facet = {}
    for facet_name in facet_map.keys():
        facet_todos = get_todos(day, facet_name)
        if facet_todos:
            # Filter out cancelled todos and add facet info
            facet_todos = [t for t in facet_todos if not t.get("cancelled")]
            for todo in facet_todos:
                todo["facet"] = facet_name
        else:
            facet_todos = []
        todos_by_facet[facet_name] = facet_todos

    # Sort facets for initial page load:
    # 1. Facets with incomplete items first, sorted by incomplete count (descending)
    # 2. Fully completed facets next, sorted alphabetically
    # 3. Empty facets last, sorted alphabetically
    def facet_sort_key(item):
        facet_name, facet_todos = item
        incomplete_count = sum(1 for todo in facet_todos if not todo.get("completed"))
        has_no_todos = len(facet_todos) == 0
        all_complete = incomplete_count == 0 and len(facet_todos) > 0
        # Return tuple: (has_no_todos, all_complete, -incomplete_count, facet_name)
        # has_no_todos=False sorts before has_no_todos=True (empty facets last)
        # all_complete=False sorts before all_complete=True (incomplete before complete)
        # -incomplete_count sorts higher counts first
        # facet_name for alphabetical tie-breaking
        return (has_no_todos, all_complete, -incomplete_count, facet_name)

    sorted_todos_by_facet = dict(sorted(todos_by_facet.items(), key=facet_sort_key))

    today_day = date.today().strftime("%Y%m%d")

    # Compute facet counts for facet pill badges
    facet_counts = {}
    for facet_name, facet_todos in todos_by_facet.items():
        pending = sum(1 for t in facet_todos if not t.get("completed"))
        if pending > 0:
            facet_counts[facet_name] = pending

    return render_template(
        "app.html",
        title=format_date(day),
        today_day=today_day,
        todos_by_facet=sorted_todos_by_facet,
        facet_map=facet_map,
        facet_counts=facet_counts,
    )


@todos_bp.route("/<day>/move", methods=["POST"])
def move_todo(day: str):  # type: ignore[override]
    """Move a todo to a different day by cancelling source and adding to target."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    payload = request.get_json(silent=True) or {}
    target_day = (payload.get("target_day") or "").strip()
    facet = (payload.get("facet") or "personal").strip()
    index_value = payload.get("index")

    if not DATE_RE.fullmatch(target_day):
        return jsonify({"error": "Please pick a valid target day."}), 400

    try:
        index = int(index_value)
    except (TypeError, ValueError):
        return jsonify({"error": "Missing todo index."}), 400

    if index <= 0:
        return jsonify({"error": "Todo index must be positive."}), 400

    if target_day == day:
        return jsonify(
            {
                "status": "noop",
                "message": "Todo is already on that day.",
                "redirect": url_for("app:todos.todos_day", day=day),
            }
        )

    try:
        source_checklist = TodoChecklist.load(day, facet)
    except RuntimeError as exc:
        current_app.logger.debug(
            "Failed to load source todo list for %s/%s: %s", facet, day, exc
        )
        return (
            jsonify({"error": "Todo list changed, please refresh and try again."}),
            409,
        )

    try:
        target_checklist = TodoChecklist.load(target_day, facet)
    except RuntimeError as exc:
        current_app.logger.debug(
            "Failed to load target todo list for %s/%s: %s", facet, target_day, exc
        )
        return jsonify({"error": "Unable to access target day."}), 500

    try:
        source_item = source_checklist.get_item(index)
    except (IndexError, TodoLineNumberError) as exc:
        current_app.logger.debug("Failed to locate todo %s on %s: %s", index, day, exc)
        return (
            jsonify({"error": "Todo list changed, please refresh and try again."}),
            409,
        )

    # Add to target day
    try:
        # Reconstruct text with time if present
        text = source_item.text
        if source_item.time:
            text = f"{text} ({source_item.time})"
        # Preserve original created_at timestamp when moving
        new_item = target_checklist.append_entry(
            text, created_at=source_item.created_at
        )
    except TodoEmptyTextError as exc:
        current_app.logger.debug("Failed to append todo to %s: %s", target_day, exc)
        return jsonify({"error": "Unable to move todo to the selected day."}), 400

    # Preserve completed status
    if source_item.completed:
        target_checklist.mark_done(new_item.index)

    # Cancel from source day
    source_checklist.cancel_entry(index)

    log_app_action(
        app="todos",
        facet=facet,
        action="todo_move",
        params={
            "source_day": day,
            "target_day": target_day,
            "line_number": index,
            "text": source_item.text,
            "completed": source_item.completed,
        },
        day=day,
    )

    redirect_url = url_for("app:todos.todos_day", day=target_day)
    counts = _compute_badge_counts(day, facet)
    return jsonify(
        {
            "status": "ok",
            "redirect": redirect_url,
            "target_day": target_day,
            **counts,
        }
    )


@todos_bp.route("/<day>/generate", methods=["POST"])
def generate_todos(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    payload = request.get_json(silent=True) or {}
    facet = (payload.get("facet") or "personal").strip()

    day_date = datetime.strptime(day, "%Y%m%d")
    yesterday = (day_date - timedelta(days=1)).strftime("%Y%m%d")
    yesterday_path = _todo_path(yesterday, facet)

    yesterday_content = ""
    if yesterday_path.exists():
        try:
            yesterday_content = yesterday_path.read_text(encoding="utf-8")
        except OSError:
            yesterday_content = ""

    prompt = f"""Generate a TODO checklist for {day_date.strftime('%Y-%m-%d')} in the {facet} facet.

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Target day: {day_date.strftime('%Y-%m-%d')}
Target facet: {facet}
Target file: facets/{facet}/todos/{day}.jsonl

Yesterday's todos content:
{yesterday_content if yesterday_content else "(No todos recorded yesterday)"}

Write the generated checklist to facets/{facet}/todos/{day}.jsonl"""

    try:
        from convey.utils import spawn_agent

        agent_id = spawn_agent(
            prompt=prompt,
            name="todos:todo",
            provider="openai",
            config={},
        )
    except Exception as exc:  # pragma: no cover - network/agent failure
        return jsonify({"error": f"Failed to spawn agent: {exc}"}), 500

    if not hasattr(state, "todo_generation_agents"):
        state.todo_generation_agents = {}
    state.todo_generation_agents[day] = agent_id

    return jsonify({"agent_id": agent_id, "status": "started"})


@todos_bp.route("/<day>/generation-status")
def todo_generation_status(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    facet = request.args.get("facet", "personal")
    agent_id = request.args.get("agent_id")
    if not agent_id and hasattr(state, "todo_generation_agents"):
        agent_id = state.todo_generation_agents.get(day)

    if not agent_id:
        return jsonify({"status": "none", "agent_id": None})

    from think.cortex_client import cortex_agents

    todo_path = _todo_path(day, facet)

    agents_dir = Path(state.journal_root) / "agents"
    agent_file = agents_dir / f"{agent_id}.jsonl"

    if agent_file.exists():
        if todo_path.exists():
            if (
                hasattr(state, "todo_generation_agents")
                and day in state.todo_generation_agents
            ):
                del state.todo_generation_agents[day]
            return jsonify(
                {"status": "finished", "agent_id": agent_id, "todo_created": True}
            )
        return jsonify(
            {"status": "finished", "agent_id": agent_id, "todo_created": False}
        )

    try:
        response = cortex_agents(limit=100, offset=0)
        if response:
            agents = response.get("agents", [])
            for agent in agents:
                if agent.get("id") == agent_id:
                    return jsonify({"status": "running", "agent_id": agent_id})
            return jsonify({"status": "unknown", "agent_id": agent_id})
    except Exception:  # pragma: no cover - external call failure
        pass

    return jsonify({"status": "unknown", "agent_id": agent_id})


@todos_bp.route("/<day>/generate-weekly/<facet>", methods=["POST"])
def generate_weekly_todos(day: str, facet: str):  # type: ignore[override]
    """Spawn todo_weekly agent for a specific facet."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    day_date = datetime.strptime(day, "%Y%m%d")

    prompt = f"""Review the past week and generate high-impact todos for {facet} facet.

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Target day: {day_date.strftime('%Y-%m-%d')}
Target facet: {facet}
Target file: facets/{facet}/todos/{day}.jsonl

Focus on surfacing the most important unfinished work from the past 7 days."""

    try:
        from convey.utils import spawn_agent

        agent_id = spawn_agent(
            prompt=prompt,
            name="todos:weekly",
            provider="openai",
            config={},
        )
    except Exception as exc:  # pragma: no cover - network/agent failure
        return jsonify({"error": f"Failed to spawn agent: {exc}"}), 500

    return jsonify({"agent_id": agent_id, "status": "started"})
