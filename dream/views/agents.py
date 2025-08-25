from __future__ import annotations

import asyncio
import json
import os
import time

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
        # Find all .txt files and match with optional .json files
        txt_files = [f for f in os.listdir(agents_path) if f.endswith(".txt")]

        for txt_file in txt_files:
            base_name = txt_file[:-4]  # Remove .txt extension
            json_file = base_name + ".json"

            txt_path = os.path.join(agents_path, txt_file)
            json_path = os.path.join(agents_path, json_file)

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
                            agent_config = json.load(f)
                            title = agent_config.get("title", base_name)
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
    """Get list of live agents from Cortex and historical agents from journal."""
    # Get type parameter (live, historical, or all)
    agent_type = request.args.get("type", "all")
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))

    # Validate parameters
    limit = max(1, min(limit, 100))  # Limit between 1-100
    offset = max(0, offset)

    live_agents = []
    historical_agents = []

    # Load persona metadata once for efficiency
    from ..utils import get_personas

    personas = get_personas()
    persona_titles = {pid: p["title"] for pid, p in personas.items()}

    # Get live agents from Cortex if requested
    if agent_type in ["live", "all"]:
        from ..cortex_client import get_global_cortex_client

        client = get_global_cortex_client()

        if client:
            response = client.list_agents(limit=100, offset=0)  # Get all live agents
            if response:
                agents = response.get("agents", [])
                for agent in agents:
                    start_ms = agent.get("started_at", 0)
                    start = start_ms / 1000
                    metadata = agent.get("metadata", {})

                    persona_id = metadata.get("persona", "default")
                    # Calculate runtime in seconds for live agents (ongoing)
                    runtime_seconds = time.time() - start if start > 0 else 0
                    live_agents.append(
                        {
                            "id": agent.get("id", ""),
                            "start": start,
                            "since": time_since(start),
                            "runtime_seconds": runtime_seconds,
                            "model": metadata.get("model", ""),
                            "persona": persona_id,
                            "persona_title": persona_titles.get(persona_id, persona_id),
                            "prompt": metadata.get("prompt", ""),
                            "status": agent.get("status", "running"),
                            "pid": agent.get("pid"),
                            "is_live": True,
                        }
                    )

    # Get historical agents from journal if requested
    if agent_type in ["historical", "all"]:
        agents_dir = _agents_dir()
        if agents_dir and os.path.exists(agents_dir):
            # Get all .jsonl files
            jsonl_files = [f for f in os.listdir(agents_dir) if f.endswith(".jsonl")]

            for jsonl_file in jsonl_files:
                agent_id = jsonl_file[:-6]  # Remove .jsonl extension

                # Skip if this is a live agent
                if any(a["id"] == agent_id for a in live_agents):
                    continue

                agent_path = os.path.join(agents_dir, jsonl_file)
                try:
                    # Read first and last lines to get metadata
                    with open(agent_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            first_event = json.loads(lines[0])
                            last_event = (
                                json.loads(lines[-1]) if len(lines) > 1 else first_event
                            )

                            # Extract metadata from events
                            start_ms = first_event.get("ts", int(agent_id))
                            start = start_ms / 1000

                            # Get end time from last event
                            end_ms = last_event.get("ts", start_ms)
                            end = end_ms / 1000
                            # Calculate runtime in seconds
                            runtime_seconds = end - start if end >= start else 0

                            # Determine status from last event
                            status = "finished"
                            if last_event.get("event") == "error":
                                status = "error"
                            elif last_event.get("event") != "finish":
                                status = "interrupted"

                            persona_id = first_event.get("persona", "default")
                            historical_agents.append(
                                {
                                    "id": agent_id,
                                    "start": start,
                                    "since": time_since(start),
                                    "runtime_seconds": runtime_seconds,
                                    "model": first_event.get("model", ""),
                                    "persona": persona_id,
                                    "persona_title": persona_titles.get(
                                        persona_id, persona_id
                                    ),
                                    "prompt": first_event.get("prompt", ""),
                                    "status": status,
                                    "pid": None,
                                    "is_live": False,
                                }
                            )
                except Exception:
                    # Skip malformed files
                    continue

    # Combine and sort by timestamp (newest first)
    all_agents = live_agents + historical_agents
    all_agents.sort(key=lambda x: x["start"], reverse=True)

    # Apply pagination
    total = len(all_agents)
    paginated = all_agents[offset : offset + limit]

    return jsonify(
        {
            "agents": paginated,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_more": offset + limit < total,
            },
            "live_count": len(live_agents),
            "historical_count": len(historical_agents),
        }
    )


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


@bp.route("/agents/api/update/<agent_id>", methods=["PUT"])
def update_agent(agent_id: str) -> object:
    """Update an agent's title and content or create a new one."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    new_title = data.get("title", "").strip()
    new_content = data.get("content", "").strip()

    if not new_title or not new_content:
        return jsonify({"error": "Title and content are required"}), 400

    # Path to agent files
    agents_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", "agents"
    )
    json_path = os.path.join(agents_path, f"{agent_id}.json")
    txt_path = os.path.join(agents_path, f"{agent_id}.txt")

    # Check if this is update or create
    is_new = not os.path.isfile(json_path)

    try:
        if is_new:
            # Create new agent config
            agent_config = {
                "title": new_title,
                "description": "",
                "model": "gemini-2.0-flash-exp",
            }
        else:
            # Update existing JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                agent_config = json.load(f)
            agent_config["title"] = new_title

        # Update description (first line of content)
        first_line = new_content.split("\n")[0] if new_content else ""
        agent_config["description"] = (
            first_line[:100] + "..." if len(first_line) > 100 else first_line
        )

        # Write JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(agent_config, f, indent=2, ensure_ascii=False)

        # Write TXT file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        action = "created" if is_new else "updated"
        return jsonify({"success": True, "message": f"Agent {action} successfully"})
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
