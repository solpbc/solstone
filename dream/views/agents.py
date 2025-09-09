from __future__ import annotations

import asyncio
import json
import os

from flask import Blueprint, jsonify, render_template, request

from .. import state

bp = Blueprint("agents", __name__, template_folder="../templates")


@bp.route("/agents")
def agents_page() -> str:
    """Render the Agents view."""
    return render_template("agents.html", active="agents")


def _agents_dir() -> str:
    if not state.journal_root:
        return ""
    path = os.path.join(state.journal_root, "agents")
    os.makedirs(path, exist_ok=True)
    return path


def _list_items(item_type: str) -> list[dict[str, object]]:
    """Generic function to list items from think/{item_type}/.

    Args:
        item_type: Either 'agents' or 'topics'

    Returns:
        List of items with id, title, and description (and color for topics)
    """
    # Special handling for topics to get colors from get_topics()
    if item_type == "topics":
        from think.utils import get_topics

        topics = get_topics()
        items: list[dict[str, object]] = []

        for name, info in topics.items():
            item = {
                "id": name,
                "title": info.get("title", name),
                "description": info.get("description", ""),
                "color": info.get("color", "#007bff"),
            }
            items.append(item)

        items.sort(key=lambda x: x.get("title", ""))
        return items

    # Standard handling for agents
    items_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", item_type
    )
    items: list[dict[str, object]] = []

    if os.path.isdir(items_path):
        # Find all .txt files and match with optional .json files
        txt_files = [f for f in os.listdir(items_path) if f.endswith(".txt")]

        for txt_file in txt_files:
            base_name = txt_file[:-4]  # Remove .txt extension
            json_file = base_name + ".json"

            txt_path = os.path.join(items_path, txt_file)
            json_path = os.path.join(items_path, json_file)

            try:
                # Read first line of .txt file as description
                with open(txt_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    # Remove trailing period if present for consistency
                    description = first_line.rstrip(".")

                # Read title from .json if it exists, otherwise use base_name
                title = base_name
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            title = config.get("title", base_name)
                    except Exception:
                        pass

                items.append(
                    {
                        "id": base_name,
                        "title": title,
                        "description": description,
                    }
                )
            except Exception:
                continue

    items.sort(key=lambda x: x.get("title", ""))
    return items


@bp.route("/agents/api/available")
def available_agents() -> object:
    """Return list of available agent definitions from think/agents/."""
    return jsonify(_list_items("agents"))


def _get_item_content(item_type: str, item_id: str) -> tuple[dict, int]:
    """Generic function to get item content from think/{item_type}/{item_id}.txt.

    Args:
        item_type: Either 'agents' or 'topics'
        item_id: The item identifier

    Returns:
        Tuple of (response dict, status code)
    """
    items_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", item_type
    )
    txt_path = os.path.join(items_path, f"{item_id}.txt")

    if not os.path.isfile(txt_path):
        return {"error": f"{item_type[:-1].title()} not found"}, 404

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}, 200
    except Exception as e:
        return {"error": str(e)}, 500


@bp.route("/agents/api/content/<agent_id>")
def agent_content(agent_id: str) -> object:
    """Return the .txt content for an agent."""
    response, status = _get_item_content("agents", agent_id)
    return jsonify(response), status


@bp.route("/agents/api/live")
def agents_live() -> object:
    """Get list of only live/running agents."""
    return _get_agents_list("live")


@bp.route("/agents/api/historical")
def agents_historical() -> object:
    """Get list of only historical/completed agents."""
    return _get_agents_list("historical")


@bp.route("/agents/api/list")
def agents_list() -> object:
    # Get type parameter (live, historical, or all)
    agent_type = request.args.get("type", "all")
    return _get_agents_list(agent_type)


def _get_agents_list(agent_type: str) -> object:
    """Internal helper to get agents list."""
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))

    # Validate parameters - cortex_agents already does this but let's be explicit
    limit = max(1, min(limit, 100))  # Limit between 1-100
    offset = max(0, offset)

    # Get agents directly from cortex_agents function
    from ..utils import time_since
    from think.cortex_client import cortex_agents
    from think.utils import get_personas

    # Get all agents using cortex_agents
    response = cortex_agents(limit=limit, offset=offset, agent_type=agent_type)

    # Load persona titles for display
    personas = get_personas()
    persona_titles = {pid: p["title"] for pid, p in personas.items()}

    # Format agents for display
    agents = response.get("agents", [])
    for agent in agents:
        persona_id = agent.get("persona", "default")
        agent["persona_title"] = persona_titles.get(persona_id, persona_id)
        # Convert milliseconds to seconds for time_since
        agent["since"] = time_since(agent["start"] / 1000) if agent.get("start") else "unknown"
        # Keep backward compatibility
        agent["pid"] = None  # We don't track PIDs in the new system

    return jsonify(response)


@bp.route("/agents/api/plan", methods=["POST"])
def create_plan() -> object:
    """Create a plan from user input using the planner agent."""
    data = request.get_json()
    if not data or not data.get("request"):
        return jsonify({"error": "Request is required"}), 400

    user_request = data["request"]
    model = data.get("model", "")

    try:
        # Import planner module
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
        )
        from think.planner import generate_plan

        # Generate the plan (synchronous)
        if model:
            plan = generate_plan(user_request, model=model)
        else:
            plan = generate_plan(user_request)
        return jsonify({"plan": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _update_item(item_type: str, item_id: str, data: dict) -> tuple[dict, int]:
    """Generic function to update or create an item.

    Args:
        item_type: Either 'agents' or 'topics'
        item_id: The item identifier
        data: Request data with title and content

    Returns:
        Tuple of (response dict, status code)
    """
    if not data:
        return {"error": "No data provided"}, 400

    new_title = data.get("title", "").strip()
    new_content = data.get("content", "").strip()

    if not new_title or not new_content:
        return {"error": "Title and content are required"}, 400

    # Path to item files
    items_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", item_type
    )
    json_path = os.path.join(items_path, f"{item_id}.json")
    txt_path = os.path.join(items_path, f"{item_id}.txt")

    # Check if this is update or create
    is_new = not os.path.isfile(json_path)

    try:
        if is_new:
            # Create new item config
            item_config = {
                "title": new_title,
                "description": "",
            }
            # Add model field for agents only
            if item_type == "agents":
                item_config["model"] = "gemini-2.0-flash-exp"
        else:
            # Update existing JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                item_config = json.load(f)
            item_config["title"] = new_title

        # Update description (first line of content)
        first_line = new_content.split("\n")[0] if new_content else ""
        item_config["description"] = (
            first_line[:100] + "..." if len(first_line) > 100 else first_line
        )

        # Write JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(item_config, f, indent=2, ensure_ascii=False)

        # Write TXT file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        action = "created" if is_new else "updated"
        item_name = item_type[:-1].title()  # 'agents' -> 'Agent', 'topics' -> 'Topic'
        return {"success": True, "message": f"{item_name} {action} successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500


@bp.route("/agents/api/update/<agent_id>", methods=["PUT"])
def update_agent(agent_id: str) -> object:
    """Update an agent's title and content or create a new one."""
    response, status = _update_item("agents", agent_id, request.get_json())
    return jsonify(response), status


@bp.route("/agents/api/start", methods=["POST"])
def start_agent() -> object:
    """Start a new agent with the given plan and configuration."""
    data = request.get_json()
    if not data or not data.get("plan"):
        return jsonify({"error": "Plan is required"}), 400

    plan = data["plan"]
    backend = data.get("backend", "openai")
    persona = data.get("persona", "default")
    config = data.get("config", {})

    try:
        # Import agents module
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
        )
        from think import agents

        # Run the agent async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                agents.run_agent(
                    plan,
                    backend=backend,
                    config=config,
                    persona=persona,
                )
            )
            return jsonify({"success": True, "result": result})
        finally:
            loop.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Topics API endpoints
@bp.route("/agents/api/topics")
def available_topics() -> object:
    """Return list of available topic definitions from think/topics/."""
    return jsonify(_list_items("topics"))


@bp.route("/agents/api/topics/content/<topic_id>")
def topic_content(topic_id: str) -> object:
    """Return the .txt content for a topic."""
    response, status = _get_item_content("topics", topic_id)
    return jsonify(response), status


@bp.route("/agents/api/topics/update/<topic_id>", methods=["PUT"])
def update_topic(topic_id: str) -> object:
    """Update a topic's title and content or create a new one."""
    response, status = _update_item("topics", topic_id, request.get_json())
    return jsonify(response), status
