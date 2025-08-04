from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from dotenv import load_dotenv

from think.indexer import search_entities
from think.utils import get_domains

bp = Blueprint("domains", __name__, template_folder="../templates")


@bp.route("/domains")
def domains_page() -> str:
    return render_template("domains.html", active="domains")


@bp.route("/api/domains")
def domains_list() -> Any:
    """Return available domains with their metadata."""
    return jsonify(get_domains())


@bp.route("/api/domains", methods=["POST"])
def create_domain() -> Any:
    """Create a new domain with the provided metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    domain_name = data.get("name", "").strip()
    if not domain_name:
        return jsonify({"error": "Domain name is required"}), 400

    # Validate domain name (basic alphanumeric + hyphens/underscores)
    if not domain_name.replace("-", "").replace("_", "").isalnum():
        return (
            jsonify(
                {
                    "error": "Domain name must be alphanumeric with optional hyphens or underscores"
                }
            ),
            400,
        )

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name

    # Check if domain already exists
    if domain_path.exists():
        return jsonify({"error": "Domain already exists"}), 409

    try:
        # Create domain directory
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create domain.json
        domain_data = {
            "title": data.get("title", domain_name),
            "description": data.get("description", ""),
        }

        if data.get("color"):
            domain_data["color"] = data["color"]
        if data.get("emoji"):
            domain_data["emoji"] = data["emoji"]

        domain_json = domain_path / "domain.json"
        with open(domain_json, "w", encoding="utf-8") as f:
            json.dump(domain_data, f, indent=2, ensure_ascii=False)

        # Create empty entities.md
        entities_md = domain_path / "entities.md"
        entities_md.write_text("", encoding="utf-8")

        # Create matters directory
        matters_dir = domain_path / "matters"
        matters_dir.mkdir(exist_ok=True)

        return jsonify({"success": True, "domain": domain_name})

    except Exception as e:
        return jsonify({"error": f"Failed to create domain: {str(e)}"}), 500


@bp.route("/domains/<domain_name>")
def domain_detail(domain_name: str) -> str:
    """Display detailed view for a specific domain."""
    domains = get_domains()
    if domain_name not in domains:
        return render_template("404.html"), 404
    
    domain_data = domains[domain_name]
    return render_template("domain_detail.html", 
                         domain_name=domain_name, 
                         domain_data=domain_data,
                         active="domains")


@bp.route("/api/domains/<domain_name>", methods=["PUT"])
def update_domain(domain_name: str) -> Any:
    """Update an existing domain's metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    domain_json = domain_path / "domain.json"

    if not domain_json.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Read existing domain.json
        with open(domain_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        # Update only provided fields
        if "title" in data:
            existing_data["title"] = data["title"]
        if "description" in data:
            existing_data["description"] = data["description"]
        if "color" in data:
            if data["color"]:
                existing_data["color"] = data["color"]
            else:
                existing_data.pop("color", None)
        if "emoji" in data:
            if data["emoji"]:
                existing_data["emoji"] = data["emoji"]
            else:
                existing_data.pop("emoji", None)

        # Write updated domain.json
        with open(domain_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        return jsonify({"success": True, "domain": domain_name})

    except Exception as e:
        return jsonify({"error": f"Failed to update domain: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities")
def get_domain_entities(domain_name: str) -> Any:
    """Get entities for a specific domain."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    entities_file = domain_path / "entities.md"
    
    if not entities_file.exists():
        return jsonify({"domain_entities": [], "all_entities": []})
    
    # Read domain-specific entities
    domain_entities = []
    try:
        with open(entities_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("* "):
                    parts = line[2:].split(":", 1)
                    if len(parts) == 2:
                        etype, rest = parts
                        etype = etype.strip()
                        name_desc = rest.strip().split(" - ", 1)
                        name = name_desc[0].strip()
                        desc = name_desc[1].strip() if len(name_desc) > 1 else ""
                        domain_entities.append({
                            "type": etype,
                            "name": name,
                            "desc": desc,
                            "starred": True
                        })
    except Exception:
        pass
    
    # Get all entities from global search
    types = ["Person", "Company", "Project", "Tool"]
    all_entities = []
    
    for etype in types:
        _total_top, top_results = search_entities(
            "", limit=500, etype=etype, top=True, order="count"
        )
        _total_other, other_results = search_entities(
            "", limit=500, etype=etype, top=False, order="count"
        )
        
        for result in top_results + other_results:
            meta = result["metadata"]
            entity = {
                "type": etype,
                "name": meta["name"],
                "desc": result["text"],
                "top": meta.get("top", False),
                "count": meta.get("days", 0),
                "starred": False
            }
            
            # Check if this entity is already in domain entities
            for domain_entity in domain_entities:
                if (domain_entity["type"] == entity["type"] and 
                    domain_entity["name"] == entity["name"]):
                    entity["starred"] = True
                    break
            
            all_entities.append(entity)
    
    # Sort: starred entities first, then by count/top status
    all_entities.sort(key=lambda x: (not x["starred"], not x.get("top", False), -x.get("count", 0)))
    
    return jsonify({
        "domain_entities": domain_entities,
        "all_entities": all_entities
    })


@bp.route("/api/domains/<domain_name>/entities", methods=["POST"])
def add_domain_entity(domain_name: str) -> Any:
    """Add an entity to a domain's entities.md file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    entities_file = domain_path / "entities.md"
    
    if not domain_path.exists():
        return jsonify({"error": "Domain not found"}), 404

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()
    desc = data.get("desc", "").strip()
    
    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400
    
    try:
        # Read existing content
        existing_lines = []
        if entities_file.exists():
            with open(entities_file, "r", encoding="utf-8") as f:
                existing_lines = f.readlines()
        
        # Check if entity already exists
        new_line = f"* {etype}: {name}"
        if desc:
            new_line += f" - {desc}"
        new_line += "\n"
        
        for line in existing_lines:
            if line.strip().startswith(f"* {etype}: {name}"):
                return jsonify({"error": "Entity already exists in domain"}), 409
        
        # Add the new entity
        existing_lines.append(new_line)
        
        # Write back to file
        with open(entities_file, "w", encoding="utf-8") as f:
            f.writelines(existing_lines)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"error": f"Failed to add entity: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities", methods=["DELETE"])
def remove_domain_entity(domain_name: str) -> Any:
    """Remove an entity from a domain's entities.md file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    entities_file = domain_path / "entities.md"
    
    if not entities_file.exists():
        return jsonify({"error": "Entities file not found"}), 404

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()
    
    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400
    
    try:
        # Read existing content
        with open(entities_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Filter out the entity to remove
        new_lines = []
        removed = False
        for line in lines:
            if line.strip().startswith(f"* {etype}: {name}"):
                removed = True
                continue
            new_lines.append(line)
        
        if not removed:
            return jsonify({"error": "Entity not found in domain"}), 404
        
        # Write back to file
        with open(entities_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"error": f"Failed to remove entity: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/generate-description", methods=["POST"])
def generate_domain_description(domain_name: str) -> Any:
    """Generate a description for a domain using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    current_description = data.get("current_description", "")
    
    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 500

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    if not domain_path.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Get domain metadata
        domains = get_domains()
        domain_data = domains.get(domain_name, {})
        
        # Build context for the agent
        context_parts = [
            f"Domain Name: {domain_name}",
            f"Domain Title: {domain_data.get('title', domain_name)}",
        ]
        
        if current_description:
            context_parts.append(f"Current Description: {current_description}")
        else:
            context_parts.append("Current Description: (none)")
        
        # Check if domain has entities
        entities_file = domain_path / "entities.md"
        if entities_file.exists():
            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entities_content = f.read().strip()
                if entities_content:
                    context_parts.append(f"Domain Entities: {entities_content}")
            except Exception:
                pass
        
        # Check if domain has matters
        matters_dir = domain_path / "matters"
        if matters_dir.exists():
            matters_files = list(matters_dir.glob("*.md"))
            if matters_files:
                context_parts.append(f"Domain has {len(matters_files)} matter files")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Please generate a compelling, informative description for this domain based on the following context:

{context}

Generate a clear, engaging 1-2 sentence description that captures the essence and purpose of this domain. The description should help users understand what they'll find in this domain and be appropriate for a personal knowledge management system."""

        # Import and run the Google agent
        from think.google import run_agent
        
        # Run the agent synchronously using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            description = loop.run_until_complete(
                run_agent(prompt, persona="domain_describe", max_tokens=2048)
            )
        finally:
            loop.close()
            
        return jsonify({"success": True, "description": description.strip()})
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate description: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/description", methods=["PUT"])
def update_entity_description(domain_name: str) -> Any:
    """Update an entity's description in the domain's entities.md file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_type = data.get("type", "").strip()
    entity_name = data.get("name", "").strip()
    new_description = data.get("description", "").strip()
    
    if not entity_type or not entity_name:
        return jsonify({"error": "Type and name are required"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    entities_file = domain_path / "entities.md"
    
    if not domain_path.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Read existing content
        lines = []
        if entities_file.exists():
            with open(entities_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        # Find and update the entity line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"* {entity_type}: {entity_name}"):
                # Update the line with new description
                if new_description:
                    lines[i] = f"* {entity_type}: {entity_name} - {new_description}\n"
                else:
                    lines[i] = f"* {entity_type}: {entity_name}\n"
                updated = True
                break
        
        if not updated:
            return jsonify({"error": "Entity not found in domain"}), 404
        
        # Write back to file
        with open(entities_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"error": f"Failed to update entity description: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/generate-description", methods=["POST"])
def generate_entity_description(domain_name: str) -> Any:
    """Generate a description for an entity using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_type = data.get("type", "").strip()
    entity_name = data.get("name", "").strip()
    current_description = data.get("current_description", "")
    
    if not entity_type or not entity_name:
        return jsonify({"error": "Type and name are required"}), 400
    
    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 500

    try:
        # Build context for the agent
        context_parts = [
            f"Entity Type: {entity_type}",
            f"Entity Name: {entity_name}",
            f"Domain: {domain_name}",
        ]
        
        if current_description:
            context_parts.append(f"Current Description: {current_description}")
        else:
            context_parts.append("Current Description: (none)")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Please generate a compelling, informative description for this entity based on the following context:

{context}

Generate a clear, concise description (1-2 sentences) that captures what this {entity_type.lower()} is and why it's relevant. The description should be appropriate for a personal knowledge management system and help users understand the entity's significance or role."""

        # Import and run the Google agent
        from think.google import run_agent
        
        # Run the agent synchronously using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            description = loop.run_until_complete(
                run_agent(prompt, persona="domain_describe", max_tokens=1024)
            )
        finally:
            loop.close()
            
        return jsonify({"success": True, "description": description.strip()})
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate entity description: {str(e)}"}), 500
