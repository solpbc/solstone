# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import json
import os

from flask import Blueprint, jsonify, request

from apps.utils import log_app_action
from convey import state

agents_bp = Blueprint(
    "app:agents",
    __name__,
    url_prefix="/app/agents",
)


def _list_items(item_type: str) -> list[dict[str, object]]:
    """Generic function to list items from think/{item_type}/.

    Args:
        item_type: Either 'agents' or 'insights'

    Returns:
        List of items with id, title, and description (and color for insights)
    """
    # Special handling for insights to get colors from get_insights()
    if item_type == "insights":
        from think.utils import get_insights

        insights = get_insights()
        items: list[dict[str, object]] = []

        for name, info in insights.items():
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
    # Agents are in muse/agents/, insights are in think/insights/
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
                        "multi_facet": metadata.get("multi_facet", False),
                        "tools": metadata.get("tools"),
                        "backend": metadata.get("backend"),
                        "model": metadata.get("model"),
                    }
                )
            except Exception:
                continue

    items.sort(key=lambda x: x.get("title", ""))
    return items


@agents_bp.route("/api/available")
def available_agents() -> object:
    """Return list of available agent definitions from think/agents/."""
    return jsonify(_list_items("agents"))


@agents_bp.route("/api/preview/<persona>")
def preview_agent_prompt(persona: str) -> object:
    """Return the complete rendered prompt for an agent.

    Returns:
        {
            "persona": str,
            "title": str,
            "full_prompt": str,           # Combined instruction + extra_context
            "example_invocation": str     # Example prompt for multi-facet agents
        }
    """
    try:
        from datetime import datetime

        from think.utils import get_agent

        config = get_agent(persona)

        instruction = config.get("instruction", "")
        extra_context = config.get("extra_context", "")
        full_prompt = f"{instruction}\n\n---\n\n{extra_context}".strip()

        # Generate example invocation for multi-facet agents
        example_invocation = ""
        if config.get("multi_facet"):
            yesterday = (datetime.now()).strftime("%Y%m%d")  # Simplified for example
            example_invocation = (
                f"You are processing facet 'personal' for yesterday ({yesterday}), "
                f"use get_facet('personal') to load the correct context before starting."
            )

        return jsonify(
            {
                "persona": persona,
                "title": config.get("title", persona),
                "full_prompt": full_prompt,
                "example_invocation": example_invocation,
                "multi_facet": config.get("multi_facet", False),
            }
        )
    except FileNotFoundError:
        return jsonify({"error": f"Agent '{persona}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_item_content(item_type: str, item_id: str) -> tuple[dict, int]:
    """Generic function to get item content from {item_type}/{item_id}.txt.

    Args:
        item_type: Either 'agents' or 'insights'
        item_id: The item identifier

    Returns:
        Tuple of (response dict, status code)
    """
    # Agents are in muse/agents/, insights are in think/insights/
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


@agents_bp.route("/api/content/<agent_id>")
def agent_content(agent_id: str) -> object:
    """Return the .txt content for an agent."""
    response, status = _get_item_content("agents", agent_id)
    return jsonify(response), status


@agents_bp.route("/api/run/<agent_id>")
def agent_run(agent_id: str) -> object:
    """Return formatted markdown for a completed agent run.

    Locates the agent JSONL file and formats it using the formatters framework.

    Returns:
        {
            "header": str,       # Agent metadata header
            "markdown": str,     # Full formatted markdown (header + chunks)
            "error": str | None  # Optional error message
        }
    """
    from pathlib import Path

    from think.formatters import format_file

    # Locate the agent JSONL file
    journal_path = Path(state.journal_root)
    agent_file = journal_path / "agents" / f"{agent_id}.jsonl"

    if not agent_file.exists():
        return jsonify({"error": f"Agent run {agent_id} not found"}), 404

    try:
        chunks, meta = format_file(agent_file)

        # Build full markdown: header + all chunks
        parts = []
        header = meta.get("header", "")
        if header:
            parts.append(header)

        for chunk in chunks:
            parts.append(chunk.get("markdown", ""))

        markdown = "\n".join(parts)

        return jsonify(
            {
                "header": header,
                "markdown": markdown,
                "error": meta.get("error"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@agents_bp.route("/api/list")
def agents_list() -> object:
    """Get list of completed agent runs with pagination."""
    from convey.utils import parse_pagination_params, time_since
    from muse.cortex_client import cortex_agents
    from think.utils import get_agents

    limit, offset = parse_pagination_params(default_limit=20, max_limit=100)

    # Get completed agents from journal
    response = cortex_agents(limit=limit, offset=offset, agent_type="historical")

    # Load agent titles for display
    agents_meta = get_agents()
    persona_titles = {aid: a["title"] for aid, a in agents_meta.items()}

    # Format agents for display
    for agent in response.get("agents", []):
        persona_id = agent.get("persona", "default")
        agent["persona_title"] = persona_titles.get(persona_id, persona_id)
        agent["since"] = (
            time_since(agent["start"] / 1000) if agent.get("start") else "unknown"
        )

    return jsonify(response)


@agents_bp.route("/api/plan", methods=["POST"])
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
        item_type: Either 'agents' or 'insights'
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
    multi_facet = data.get("multi_facet")  # Can be None or boolean
    backend = data.get("backend")  # Can be None or backend name
    model = data.get("model")  # Can be None or model name

    if not new_title or not new_content:
        return {"error": "Title and content are required"}, 400

    # Path to item files
    # Agents are in muse/agents/, insights are in think/insights/
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
                if multi_facet is not None and multi_facet:
                    item_config["multi_facet"] = True
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

                # Multi-facet (boolean field)
                if multi_facet is not None:
                    if multi_facet:
                        item_config["multi_facet"] = True
                    elif "multi_facet" in item_config:
                        del item_config["multi_facet"]
                # Don't delete if multi_facet is None (not provided)

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
        item_name = item_type[
            :-1
        ].title()  # 'agents' -> 'Agent', 'insights' -> 'Insight'
        return {
            "success": True,
            "message": f"{item_name} {action} successfully",
            "is_new": is_new,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


@agents_bp.route("/api/update/<agent_id>", methods=["PUT"])
def update_agent(agent_id: str) -> object:
    """Update an agent's title and content or create a new one."""
    data = request.get_json()

    # Handle insights-specific fields (color, muted)
    if "color" in data or "disabled" in data:
        # This is actually an insight update, not an agent
        return jsonify({"error": "Invalid fields for agent update"}), 400

    response, status = _update_item("agents", agent_id, data)

    # Log successful agent create/update (journal-level since agents are global)
    if status == 200 and response.get("success"):
        action = "agent_create" if response.get("is_new") else "agent_update"
        log_app_action(
            app="agents",
            facet=None,
            action=action,
            params={"agent_id": agent_id, "title": data.get("title", "")},
        )

    return jsonify(response), status


@agents_bp.route("/api/start", methods=["POST"])
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
        from convey.utils import spawn_agent

        # Create the agent request
        agent_id = spawn_agent(
            prompt=prompt_value,
            persona=persona,
            backend=backend,
            config=config,
        )

        return jsonify({"success": True, "agent_id": agent_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Insights API endpoints
@agents_bp.route("/api/insights")
def available_insights() -> object:
    """Return list of available insight definitions from think/insights/."""
    return jsonify(_list_items("insights"))


@agents_bp.route("/api/insights/content/<insight_id>")
def insight_content(insight_id: str) -> object:
    """Return the .txt content for an insight."""
    response, status = _get_item_content("insights", insight_id)
    return jsonify(response), status


@agents_bp.route("/api/insights/update/<insight_id>", methods=["PUT"])
def update_insight(insight_id: str) -> object:
    """Update an insight's title and content or create a new one."""
    data = request.get_json()

    # Handle insight-specific fields
    if "color" in data:
        # Save color in JSON config
        insights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "think", "insights"
        )
        json_path = os.path.join(insights_path, f"{insight_id}.json")

        insight_config = {}
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                insight_config = json.load(f)

        insight_config["color"] = data["color"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(insight_config, f, indent=4)

    if "disabled" in data:
        # Save disabled state in JSON config
        insights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "think", "insights"
        )
        json_path = os.path.join(insights_path, f"{insight_id}.json")

        insight_config = {}
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                insight_config = json.load(f)

        insight_config["disabled"] = data["disabled"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(insight_config, f, indent=4)

    response, status = _update_item("insights", insight_id, data)

    # Log successful insight create/update (journal-level since insights are global)
    if status == 200 and response.get("success"):
        action = "insight_create" if response.get("is_new") else "insight_update"
        log_app_action(
            app="agents",
            facet=None,
            action=action,
            params={"insight_id": insight_id, "title": data.get("title", "")},
        )

    return jsonify(response), status


@agents_bp.route("/api/insights/toggle/<insight_id>", methods=["POST"])
def toggle_insight(insight_id: str) -> object:
    """Toggle the disabled state of an insight."""
    insights_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "think", "insights"
    )
    json_path = os.path.join(insights_path, f"{insight_id}.json")

    if not os.path.isfile(json_path):
        return jsonify({"error": "Insight not found"}), 404

    try:
        # Read existing JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            insight_config = json.load(f)

        # Toggle disabled state
        insight_config["disabled"] = not insight_config.get("disabled", False)

        # Write back to file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(insight_config, f, indent=4)

        # Log the toggle (journal-level since insights are global)
        log_app_action(
            app="agents",
            facet=None,
            action="insight_toggle",
            params={
                "insight_id": insight_id,
                "disabled": insight_config["disabled"],
            },
        )

        return jsonify({"success": True, "disabled": insight_config["disabled"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@agents_bp.route("/api/tools")
def available_tools() -> object:
    """Return list of available MCP tools from muse.mcp."""
    try:
        from muse.mcp import TOOL_PACKS, mcp

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
        "facets": "Facet-specific news and information retrieval",
    }
    return descriptions.get(pack_name, f"Tools for {pack_name} operations")
