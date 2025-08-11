from __future__ import annotations

import asyncio
import json
import os

from flask import Blueprint, jsonify, render_template, request

from .. import state
from ..utils import time_since

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


@bp.route("/agents/api/available")
def available_agents() -> object:
    """Return list of available agent definitions from think/agents/."""
    agents_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", "agents"
    )
    items: list[dict[str, object]] = []

    if os.path.isdir(agents_path):
        # Find all .json files and match with corresponding .txt files
        json_files = [f for f in os.listdir(agents_path) if f.endswith(".json")]

        for json_file in json_files:
            base_name = json_file[:-5]  # Remove .json extension
            txt_file = base_name + ".txt"

            json_path = os.path.join(agents_path, json_file)
            txt_path = os.path.join(agents_path, txt_file)

            # Skip if no corresponding .txt file
            if not os.path.isfile(txt_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    agent_config = json.load(f)

                items.append(
                    {
                        "id": base_name,
                        "title": agent_config.get("title", base_name),
                        "description": agent_config.get("description", ""),
                    }
                )
            except Exception:
                continue

    items.sort(key=lambda x: x.get("title", ""))
    return jsonify(items)


@bp.route("/agents/api/content/<agent_id>")
def agent_content(agent_id: str) -> object:
    """Return the .txt content for an agent."""
    agents_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", "agents"
    )
    txt_path = os.path.join(agents_path, f"{agent_id}.txt")

    if not os.path.isfile(txt_path):
        return jsonify({"error": "Agent not found"}), 404

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/agents/api/list")
def agents_list() -> object:
    """Get list of agents via cortex WebSocket API with pagination."""
    # Get pagination parameters
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))

    # Validate parameters
    limit = max(1, min(limit, 100))  # Limit between 1-100
    offset = max(0, offset)

    # Try to get cortex client
    from ..cortex_client import get_global_cortex_client

    client = get_global_cortex_client()

    if not client:
        return jsonify({"error": "Could not connect to cortex server"}), 503

    # Use cortex WebSocket API
    response = client.list_agents(limit=limit, offset=offset)
    if not response:
        return jsonify({"error": "Failed to get response from cortex server"}), 503

    agents = response.get("agents", [])
    pagination_info = response.get("pagination", {})

    # Transform cortex format to match expected frontend format
    items = []
    for agent in agents:
        start_ms = agent.get("started_at", 0)
        start = start_ms / 1000
        metadata = agent.get("metadata", {})

        items.append(
            {
                "id": agent.get("id", ""),
                "start": start,
                "since": time_since(start),
                "model": metadata.get("model", ""),
                "persona": metadata.get("persona", ""),
                "prompt": metadata.get("prompt", ""),
                "status": agent.get("status", "unknown"),
                "pid": agent.get("pid"),
            }
        )

    return jsonify({"agents": items, "pagination": pagination_info})


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


@bp.route("/agents/api/start", methods=["POST"])
def start_agent() -> object:
    """Start a new agent with the given plan and configuration."""
    data = request.get_json()
    if not data or not data.get("plan"):
        return jsonify({"error": "Plan is required"}), 400

    plan = data["plan"]
    backend = data.get("backend", "openai")
    model = data.get("model", "")
    max_tokens = data.get("max_tokens", 0)
    persona = data.get("persona", "default")

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
                    model=model,
                    max_tokens=max_tokens,
                    persona=persona,
                )
            )
            return jsonify({"success": True, "result": result})
        finally:
            loop.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
