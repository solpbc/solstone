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
                "disabled": info.get("disabled", False),
            }
            items.append(item)

        items.sort(key=lambda x: x.get("title", ""))
        return items

    # Standard handling for agents
    # Agents are in muse/agents/, topics are in think/topics/
    if item_type == "agents":
        items_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "muse", item_type
        )
    else:
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

                # Load full metadata from JSON if available
                metadata = {}
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except Exception:
                        pass

                items.append(
                    {
                        "id": base_name,
                        "title": title,
                        "description": description,
                        "schedule": metadata.get("schedule"),
                        "priority": metadata.get("priority"),
                        "multi_domain": metadata.get("multi_domain", False),
                        "tools": metadata.get("tools"),
                        "backend": metadata.get("backend"),
                        "model": metadata.get("model"),
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
    """Generic function to get item content from {item_type}/{item_id}.txt.

    Args:
        item_type: Either 'agents' or 'topics'
        item_id: The item identifier

    Returns:
        Tuple of (response dict, status code)
    """
    # Agents are in muse/agents/, topics are in think/topics/
    if item_type == "agents":
        items_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "muse", item_type
        )
    else:
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
    # Default limit depends on agent type - 20 for historical, 10 for live
    default_limit = 20 if agent_type == "historical" else 10
    limit = int(request.args.get("limit", default_limit))
    offset = int(request.args.get("offset", 0))

    # Validate parameters - cortex_agents already does this but let's be explicit
    limit = max(1, min(limit, 100))  # Limit between 1-100
    offset = max(0, offset)

    # Get agents directly from cortex_agents function
    from muse.cortex_client import cortex_agents
    from think.utils import get_agents

    from ..utils import time_since

    # Get all agents using cortex_agents
    response = cortex_agents(limit=limit, offset=offset, agent_type=agent_type)

    # Load agent titles for display
    agents_meta = get_agents()
    persona_titles = {aid: a["title"] for aid, a in agents_meta.items()}

    # Format agents for display
    agents = response.get("agents", [])
    for agent in agents:
        persona_id = agent.get("persona", "default")
        agent["persona_title"] = persona_titles.get(persona_id, persona_id)
        # Convert milliseconds to seconds for time_since
        agent["since"] = (
            time_since(agent["start"] / 1000) if agent.get("start") else "unknown"
        )
        # Keep backward compatibility
        agent["pid"] = None  # We don't track PIDs in the new system

    return jsonify(response)


@bp.route("/agents/api/plan", methods=["POST"])
def create_plan() -> object:
    """Create a prompt from user input using the planner agent."""
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

        # Generate the prompt (synchronous)
        if model:
            prompt_text = generate_plan(user_request, model=model)
        else:
            prompt_text = generate_plan(user_request)
        return jsonify({"prompt": prompt_text, "plan": prompt_text})
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
    schedule = data.get("schedule")  # Can be None, "daily", etc.
    priority = data.get("priority")  # Can be None or 0-99
    tools = data.get("tools")  # Can be None or comma-separated string
    multi_domain = data.get("multi_domain")  # Can be None or boolean
    backend = data.get("backend")  # Can be None or backend name
    model = data.get("model")  # Can be None or model name

    if not new_title or not new_content:
        return {"error": "Title and content are required"}, 400

    # Path to item files
    # Agents are in muse/agents/, topics are in think/topics/
    if item_type == "agents":
        items_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "muse", item_type
        )
    else:
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
            }
            # Add optional fields if provided (for agents only)
            if item_type == "agents":
                if schedule:
                    item_config["schedule"] = schedule
                if priority is not None:
                    item_config["priority"] = priority
                if tools:
                    item_config["tools"] = tools
                if multi_domain is not None and multi_domain:
                    item_config["multi_domain"] = True
                if backend:
                    item_config["backend"] = backend
                if model:
                    item_config["model"] = model
        else:
            # Update existing JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                item_config = json.load(f)
            item_config["title"] = new_title
            # Update or remove optional fields for agents
            if item_type == "agents":
                # Schedule
                if schedule:
                    item_config["schedule"] = schedule
                elif "schedule" in item_config:
                    del item_config["schedule"]

                # Priority
                if priority is not None:
                    item_config["priority"] = priority
                elif "priority" in item_config:
                    del item_config["priority"]

                # Tools
                if tools:
                    item_config["tools"] = tools
                elif "tools" in item_config:
                    del item_config["tools"]

                # Multi-domain (boolean field)
                if multi_domain is not None:
                    if multi_domain:
                        item_config["multi_domain"] = True
                    elif "multi_domain" in item_config:
                        del item_config["multi_domain"]
                # Don't delete if multi_domain is None (not provided)

                # Backend
                if backend:
                    item_config["backend"] = backend
                elif "backend" in item_config:
                    del item_config["backend"]

                # Model
                if model:
                    item_config["model"] = model
                elif "model" in item_config:
                    del item_config["model"]

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
    data = request.get_json()

    # Handle topics-specific fields (color, disabled)
    if "color" in data or "disabled" in data:
        # This is actually a topic update, not an agent
        return jsonify({"error": "Invalid fields for agent update"}), 400

    response, status = _update_item("agents", agent_id, data)
    return jsonify(response), status


@bp.route("/agents/api/start", methods=["POST"])
def start_agent() -> object:
    """Start a new agent with the given prompt and configuration."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Prompt is required"}), 400

    plan = (data.get("plan") or "").strip()
    prompt = (data.get("prompt") or "").strip()
    prompt_value = prompt or plan

    if not prompt_value:
        return jsonify({"error": "Prompt is required"}), 400
    backend = data.get("backend", "openai")
    persona = data.get("persona", "default")
    config = data.get("config", {})

    try:
        # Use cortex_client to spawn agent
        from muse.cortex_client import cortex_request

        # Create the agent request
        request_file = cortex_request(
            prompt=prompt_value,
            persona=persona,
            backend=backend,
            config=config,
        )

        # Extract agent_id from the filename
        from pathlib import Path

        agent_id = Path(request_file).stem.replace("_active", "")

        return jsonify({"success": True, "agent_id": agent_id})
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
    data = request.get_json()

    # Handle topic-specific fields
    if "color" in data:
        # Save color in JSON config
        topics_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "think", "topics"
        )
        json_path = os.path.join(topics_path, f"{topic_id}.json")

        topic_config = {}
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                topic_config = json.load(f)

        topic_config["color"] = data["color"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topic_config, f, indent=4)

    if "disabled" in data:
        # Save disabled state in JSON config
        topics_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "think", "topics"
        )
        json_path = os.path.join(topics_path, f"{topic_id}.json")

        topic_config = {}
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                topic_config = json.load(f)

        topic_config["disabled"] = data["disabled"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topic_config, f, indent=4)

    response, status = _update_item("topics", topic_id, data)
    return jsonify(response), status


@bp.route("/agents/api/topics/toggle/<topic_id>", methods=["POST"])
def toggle_topic(topic_id: str) -> object:
    """Toggle the disabled state of a topic."""
    topics_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", "topics"
    )
    json_path = os.path.join(topics_path, f"{topic_id}.json")

    if not os.path.isfile(json_path):
        return jsonify({"error": "Topic not found"}), 404

    try:
        # Read existing JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            topic_config = json.load(f)

        # Toggle disabled state
        topic_config["disabled"] = not topic_config.get("disabled", False)

        # Write back to file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topic_config, f, indent=4)

        return jsonify({"success": True, "disabled": topic_config["disabled"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/agents/api/tools")
def available_tools() -> object:
    """Return list of available MCP tools from muse.mcp_tools."""
    try:
        from muse.mcp_tools import TOOL_PACKS, mcp

        # Get all tools asynchronously
        async def get_all_tools():
            tools = await mcp.get_tools()
            return tools

        tools = asyncio.run(get_all_tools())

        # Transform tools into a structure suitable for the frontend
        tools_list = []

        # Add tool packs information
        packs_info = {}
        for pack_name, tool_names in TOOL_PACKS.items():
            packs_info[pack_name] = {
                "name": pack_name,
                "tools": tool_names,
                "description": _get_pack_description(pack_name),
            }

        # Process individual tools
        for name, tool in tools.items():
            # Find which packs contain this tool
            containing_packs = [
                pack for pack, tool_list in TOOL_PACKS.items() if name in tool_list
            ]

            # Extract input schema if available
            input_schema = None
            if hasattr(tool, "input_schema"):
                try:
                    input_schema = tool.input_schema
                    # If it's a pydantic model, convert to dict
                    if hasattr(input_schema, "model_json_schema"):
                        input_schema = input_schema.model_json_schema()
                    elif hasattr(input_schema, "dict"):
                        input_schema = input_schema.dict()
                except Exception:
                    pass

            tools_list.append(
                {
                    "name": name,
                    "description": tool.description or "No description available",
                    "packs": containing_packs,
                    "input_schema": input_schema,
                }
            )

        # Sort tools alphabetically
        tools_list.sort(key=lambda x: x["name"])

        return jsonify(
            {"tools": tools_list, "packs": packs_info, "total": len(tools_list)}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_pack_description(pack_name: str) -> str:
    """Return a human-friendly description for each tool pack."""
    descriptions = {
        "journal": "Core journal operations for searching and managing content",
        "todo": "Todo list management with add, remove, and complete operations",
        "domains": "Domain-specific news and information retrieval",
    }
    return descriptions.get(pack_name, f"Tools for {pack_name} operations")
