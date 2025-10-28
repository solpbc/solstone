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

from think.domains import get_domains
from think.todo import (
    TodoChecklist,
    TodoEmptyTextError,
    TodoGuardMismatchError,
    TodoLineNumberError,
    get_todos,
)

from .. import state
from ..utils import DATE_RE, adjacent_days, format_date

bp = Blueprint("todos", __name__, template_folder="../templates")


def _todo_path(day: str, domain: str) -> Path:
    return Path(state.journal_root) / "domains" / domain / "todos" / f"{day}.md"


@bp.route("/todos")
def todos_page() -> str:
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("todos.todos_day", day=today))


@bp.route("/todos/<day>", methods=["GET", "POST"])
def todos_day(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    if request.method == "POST":
        action = request.form.get("action")

        if action == "add":
            text = request.form.get("text", "").strip()
            if not text:
                flash("Cannot add an empty todo", "error")
            else:
                # Extract domain from hashtag (e.g., "#work" -> "work")
                import re

                domain_match = re.search(r"#([a-z][a-z0-9_-]*)", text, re.IGNORECASE)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    # Remove the hashtag from the text
                    text = re.sub(
                        r"\s*#" + re.escape(domain_match.group(1)) + r"\b",
                        "",
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()

                    # Validate domain exists
                    try:
                        domain_map = get_domains()
                    except Exception:
                        domain_map = {}

                    if domain not in domain_map:
                        flash(f"Domain #{domain} does not exist", "error")
                        return redirect(url_for("todos.todos_day", day=day))
                else:
                    # Default to personal if no hashtag
                    domain = "personal"

                if not text:
                    flash("Cannot add an empty todo", "error")
                else:
                    try:
                        checklist = TodoChecklist.load(day, domain)
                        checklist.append_entry(text)
                    except (TodoEmptyTextError, RuntimeError) as exc:
                        current_app.logger.debug(
                            "Failed to append todo for %s/%s: %s", domain, day, exc
                        )
                        flash("Unable to add todo right now", "error")
            return redirect(url_for("todos.todos_day", day=day))

        # Get domain and index for other actions
        domain = request.form.get("domain", "personal")
        index_str = request.form.get("index")
        guard = request.form.get("guard", "").strip()

        try:
            index = int(index_str) if index_str else None
        except ValueError:
            index = None

        if not index:
            flash("Missing todo index", "error")
            return redirect(url_for("todos.todos_day", day=day))

        try:
            checklist = TodoChecklist.load(day, domain)
        except RuntimeError as exc:
            current_app.logger.debug(
                "Failed to load checklist for %s/%s: %s", domain, day, exc
            )
            flash("Todo list changed, please refresh and try again", "error")
            return redirect(url_for("todos.todos_day", day=day))

        try:
            if action == "complete":
                checklist.mark_done(index, guard)
            elif action == "uncomplete":
                checklist.mark_undone(index, guard)
            elif action == "remove":
                checklist.remove_entry(index, guard)
            elif action == "edit":
                import re

                text = request.form.get("text", "").strip()

                # Check if text contains a domain hashtag
                domain_match = re.search(r"#([a-z][a-z0-9_-]*)", text, re.IGNORECASE)
                if domain_match:
                    new_domain = domain_match.group(1).lower()
                    # Remove the hashtag from the text
                    text = re.sub(
                        r"\s*#" + re.escape(domain_match.group(1)) + r"\b",
                        "",
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()

                    # Validate new domain exists
                    try:
                        domain_map = get_domains()
                    except Exception:
                        domain_map = {}

                    if new_domain not in domain_map:
                        flash(f"Domain #{new_domain} does not exist", "error")
                        return redirect(url_for("todos.todos_day", day=day))

                    # If domain changed, move the todo
                    if new_domain != domain:
                        # Get the completed status before moving
                        _, source_entry, completed, _ = checklist._entry_components(
                            index, guard
                        )

                        # Add to new domain
                        new_checklist = TodoChecklist.load(day, new_domain)
                        new_checklist.append_entry(text)
                        new_index = len(new_checklist.entries)
                        new_guard = new_checklist.entries[new_index - 1]

                        # Preserve completed status
                        if completed:
                            new_checklist.mark_done(new_index, new_guard)

                        # Remove from old domain
                        checklist.remove_entry(index, source_entry)

                        return redirect(url_for("todos.todos_day", day=day))

                # No domain change, just update text
                checklist.update_entry_text(index, guard, text)
            else:
                flash("Unknown action", "error")
                return redirect(url_for("todos.todos_day", day=day))
        except TodoEmptyTextError:
            flash("Cannot update todo to empty text", "error")
        except (TodoGuardMismatchError, TodoLineNumberError, IndexError, ValueError):
            flash("Todo list changed, please refresh and try again", "error")

        # If AJAX request, return JSON
        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.accept_mimetypes.accept_json
        ):
            return jsonify({"status": "ok"})

        return redirect(url_for("todos.todos_day", day=day))

    # Load todos from all domains
    try:
        domain_map = get_domains()
    except Exception as exc:  # pragma: no cover - metadata is optional
        current_app.logger.debug("Failed to load domain metadata: %s", exc)
        domain_map = {}

    # Collect todos from each domain
    todos_by_domain = {}
    for domain_name in domain_map.keys():
        domain_todos = get_todos(day, domain_name)
        if domain_todos:
            # Add domain info to each todo
            for todo in domain_todos:
                todo["domain"] = domain_name
            todos_by_domain[domain_name] = domain_todos

    # Sort domains for initial page load:
    # 1. Domains with incomplete items first, sorted by incomplete count (descending)
    # 2. Fully completed domains last, sorted alphabetically
    def domain_sort_key(item):
        domain_name, domain_todos = item
        incomplete_count = sum(1 for todo in domain_todos if not todo.get("completed"))
        all_complete = incomplete_count == 0
        # Return tuple: (all_complete, -incomplete_count, domain_name)
        # all_complete=False sorts before all_complete=True
        # -incomplete_count sorts higher counts first
        # domain_name for alphabetical tie-breaking
        return (all_complete, -incomplete_count, domain_name)

    sorted_todos_by_domain = dict(sorted(todos_by_domain.items(), key=domain_sort_key))

    prev_day, next_day = adjacent_days(state.journal_root, day)
    today_day = date.today().strftime("%Y%m%d")

    return render_template(
        "todos.html",
        active="todos",
        title=format_date(day),
        day=day,
        prev_day=prev_day,
        next_day=next_day,
        today_day=today_day,
        todos_by_domain=sorted_todos_by_domain,
        domain_map=domain_map,
    )


@bp.route("/todos/<day>/move", methods=["POST"])
def move_todo(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    payload = request.get_json(silent=True) or {}
    target_day = (payload.get("target_day") or "").strip()
    domain = (payload.get("domain") or "personal").strip()
    index_value = payload.get("index")
    guard = (payload.get("guard") or "").strip()

    if not DATE_RE.fullmatch(target_day):
        return jsonify({"error": "Please pick a valid target day."}), 400

    try:
        index = int(index_value)
    except (TypeError, ValueError):
        return jsonify({"error": "Missing todo index."}), 400

    if index <= 0:
        return jsonify({"error": "Todo index must be positive."}), 400

    if not guard:
        return jsonify({"error": "Todo guard value is required."}), 400

    if target_day == day:
        return jsonify(
            {
                "status": "noop",
                "message": "Todo is already on that day.",
                "redirect": url_for("todos.todos_day", day=day),
            }
        )

    try:
        source_checklist = TodoChecklist.load(day, domain)
    except RuntimeError as exc:
        current_app.logger.debug(
            "Failed to load source todo list for %s/%s: %s", domain, day, exc
        )
        return (
            jsonify({"error": "Todo list changed, please refresh and try again."}),
            409,
        )

    try:
        target_checklist = TodoChecklist.load(target_day, domain)
    except RuntimeError as exc:
        current_app.logger.debug(
            "Failed to load target todo list for %s/%s: %s", domain, target_day, exc
        )
        return jsonify({"error": "Unable to access target day."}), 500

    try:
        _, source_entry, completed, body = source_checklist._entry_components(
            index, guard
        )
    except (TodoLineNumberError, TodoGuardMismatchError, IndexError, ValueError) as exc:
        current_app.logger.debug("Failed to locate todo %s on %s: %s", index, day, exc)
        return (
            jsonify({"error": "Todo list changed, please refresh and try again."}),
            409,
        )

    try:
        target_checklist.append_entry(body)
    except TodoEmptyTextError as exc:
        current_app.logger.debug("Failed to append todo to %s: %s", target_day, exc)
        return jsonify({"error": "Unable to move todo to the selected day."}), 400

    new_index = len(target_checklist.entries)
    new_guard = target_checklist.entries[new_index - 1]

    if completed:
        try:
            target_checklist.mark_done(new_index, new_guard)
            new_guard = target_checklist.entries[new_index - 1]
        except (TodoGuardMismatchError, TodoLineNumberError, IndexError) as exc:
            current_app.logger.debug(
                "Failed to mark moved todo complete on %s: %s", target_day, exc
            )

    try:
        source_checklist.remove_entry(index, source_entry)
    except (TodoGuardMismatchError, TodoLineNumberError, IndexError) as exc:
        current_app.logger.debug(
            "Failed to remove todo %s from %s after move: %s", index, day, exc
        )
        try:
            target_checklist.remove_entry(new_index, new_guard)
        except Exception:  # pragma: no cover - best effort cleanup
            current_app.logger.debug("Failed to roll back moved todo on %s", target_day)
        return (
            jsonify({"error": "Todo list changed, please refresh and try again."}),
            409,
        )

    redirect_url = url_for("todos.todos_day", day=target_day)
    return jsonify({"status": "ok", "redirect": redirect_url, "target_day": target_day})


@bp.route("/todos/<day>/generate", methods=["POST"])
def generate_todos(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    payload = request.get_json(silent=True) or {}
    domain = (payload.get("domain") or "personal").strip()

    from muse.cortex_client import cortex_request

    day_date = datetime.strptime(day, "%Y%m%d")
    yesterday = (day_date - timedelta(days=1)).strftime("%Y%m%d")
    yesterday_path = _todo_path(yesterday, domain)

    yesterday_content = ""
    if yesterday_path.exists():
        try:
            yesterday_content = yesterday_path.read_text(encoding="utf-8")
        except OSError:
            yesterday_content = ""

    prompt = f"""Generate a TODO checklist for {day_date.strftime('%Y-%m-%d')} in the {domain} domain.

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Target day: {day_date.strftime('%Y-%m-%d')}
Target domain: {domain}
Target file: domains/{domain}/todos/{day}.md

Yesterday's todos content:
{yesterday_content if yesterday_content else "(No todos recorded yesterday)"}

Write the generated checklist to domains/{domain}/todos/{day}.md"""

    try:
        active_file = cortex_request(
            prompt=prompt,
            persona="todo",
            backend="openai",
            config={},
        )
        agent_id = Path(active_file).stem.replace("_active", "")
    except Exception as exc:  # pragma: no cover - network/agent failure
        return jsonify({"error": f"Failed to spawn agent: {exc}"}), 500

    if not hasattr(state, "todo_generation_agents"):
        state.todo_generation_agents = {}
    state.todo_generation_agents[day] = agent_id

    return jsonify({"agent_id": agent_id, "status": "started"})


@bp.route("/todos/<day>/generation-status")
def todo_generation_status(day: str):  # type: ignore[override]
    if not DATE_RE.fullmatch(day):
        return "", 404

    domain = request.args.get("domain", "personal")
    agent_id = request.args.get("agent_id")
    if not agent_id and hasattr(state, "todo_generation_agents"):
        agent_id = state.todo_generation_agents.get(day)

    if not agent_id:
        return jsonify({"status": "none", "agent_id": None})

    from muse.cortex_client import cortex_agents

    todo_path = _todo_path(day, domain)

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
