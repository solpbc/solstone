from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Blueprint, jsonify, render_template, request

from think.domains import get_domain_news, get_domains, get_matter, get_matters
from think.indexer import search_entities

from ..cortex_utils import run_agent_via_cortex

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

        # No need to create matters directory - matters will be created as timestamp directories

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
    return render_template(
        "domain_detail.html",
        domain_name=domain_name,
        domain_data=domain_data,
        active="domains",
    )


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
                        domain_entities.append(
                            {"type": etype, "name": name, "desc": desc, "starred": True}
                        )
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
                "starred": False,
            }

            # Check if this entity is already in domain entities
            for domain_entity in domain_entities:
                if (
                    domain_entity["type"] == entity["type"]
                    and domain_entity["name"] == entity["name"]
                ):
                    entity["starred"] = True
                    break

            all_entities.append(entity)

    # Sort: starred entities first, then by count/top status
    all_entities.sort(
        key=lambda x: (not x["starred"], not x.get("top", False), -x.get("count", 0))
    )

    return jsonify({"domain_entities": domain_entities, "all_entities": all_entities})


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
        matters = get_matters(domain_name)
        if matters:
            context_parts.append(f"Domain has {len(matters)} matters")

        context = "\n".join(context_parts)

        prompt = f"""Please generate a compelling, informative description for this domain based on the following context:

{context}

Generate a clear, engaging 1-2 sentence description that captures the essence and purpose of this domain. The description should help users understand what they'll find in this domain and be appropriate for a personal knowledge management system."""

        # Use Cortex to generate description
        description = run_agent_via_cortex(
            prompt=prompt,
            persona="domain_describe",
            backend="google",
            timeout=60,  # 1 minute for description generation
        )

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

        # Use Cortex to generate description
        description = run_agent_via_cortex(
            prompt=prompt,
            persona="domain_describe",
            backend="google",
            timeout=60,  # 1 minute for description generation
        )

        return jsonify({"success": True, "description": description.strip()})

    except Exception as e:
        return (
            jsonify({"error": f"Failed to generate entity description: {str(e)}"}),
            500,
        )


@bp.route("/domains/<domain_name>/matters/<matter_id>")
def matter_detail(domain_name: str, matter_id: str) -> str:
    """Display detailed view for a specific matter."""
    domains = get_domains()
    if domain_name not in domains:
        return render_template("404.html"), 404

    try:
        # Get comprehensive matter data using the new get_matter function
        matter_data = get_matter(domain_name, matter_id)
    except FileNotFoundError:
        return render_template("404.html"), 404

    domain_data = domains[domain_name]

    return render_template(
        "matter_detail.html",
        domain_name=domain_name,
        domain_data=domain_data,
        matter_id=matter_id,
        matter_data=matter_data,
        active="domains",
    )


@bp.route("/api/domains/<domain_name>/news")
def get_domain_news_feed(domain_name: str) -> Any:
    """Return paginated news entries for a domain."""

    cursor = request.args.get("cursor")
    day = request.args.get("day")
    # Default to 5 newsletters for initial load, 5 more for "load more"
    limit = request.args.get("days", default=5, type=int) or 5
    if limit < 0:
        limit = 5

    try:
        news_payload = get_domain_news(domain_name, cursor=cursor, limit=limit, day=day)
        return jsonify(news_payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to load news: {str(exc)}"}), 500


@bp.route("/api/domains/<domain_name>/matters")
def get_domain_matters(domain_name: str) -> Any:
    """Get matters for a specific domain with pagination support."""

    # Get pagination parameters
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", default=0, type=int)

    try:
        # Get matters from the utility function
        matters_data = get_matters(domain_name, limit=limit, offset=offset)

        # Add activity log count and format data for the frontend
        from pathlib import Path

        matters_list = []
        for matter_id, matter_info in matters_data.items():
            # Count lines in the activity log
            activity_count = 0
            activity_log_path = Path(matter_info["activity_log_path"])
            if activity_log_path.exists():
                try:
                    with open(activity_log_path, "r", encoding="utf-8") as f:
                        activity_count = sum(1 for line in f if line.strip())
                except Exception:
                    pass

            # Get mtime of the activity log file for the timestamp display
            log_mtime = None
            if activity_log_path.exists():
                try:
                    log_mtime = int(activity_log_path.stat().st_mtime)
                except Exception:
                    pass

            matter_display = {
                "matter_id": matter_id,
                "title": matter_info.get("title", ""),
                "description": matter_info.get("description", ""),
                "status": matter_info.get("status", ""),
                "priority": matter_info.get("priority", ""),
                "created": matter_info.get("created", ""),
                "activity_count": activity_count,
                "activity_log_exists": matter_info["activity_log_exists"],
                "log_mtime": log_mtime,
            }
            matters_list.append(matter_display)

        return jsonify({"matters": matters_list})

    except Exception as e:
        return jsonify({"error": f"Failed to get matters: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/matters", methods=["POST"])
def create_matter(domain_name: str) -> Any:
    """Create a new matter in the specified domain using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "Matter description is required"}), 400

    # Title is optional - AI will generate one if not provided
    title = data.get("title", "").strip()

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    if not domain_path.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Import cortex request function
        from pathlib import Path as PathLib

        from think.cortex_client import cortex_request

        # Prepare the prompt for the matter_editor persona
        priority = data.get("priority", "medium")
        status = data.get("status", "active")

        if title:
            prompt = f"""Create a new matter in the '{domain_name}' domain with the following details:

Title: {title}
Description: {description}
Priority: {priority}
Status: {status}

Please create this matter with appropriate structure and initial setup. Parse the description to identify any objectives that should be created."""
        else:
            prompt = f"""Create a new matter in the '{domain_name}' domain based on the following description:

Description: {description}
Priority: {priority}
Status: {status}

Please analyze the description and:
1. Generate an appropriate, concise title for this matter
2. Create the matter with appropriate structure and initial setup
3. Identify and create any objectives that should be created based on the description"""

        # Configure for claude backend with domain access
        config = {
            "domain": domain_name,
        }

        # Spawn the agent using cortex_request and get the agent_id
        active_file = cortex_request(
            prompt=prompt, persona="matter_editor", backend="claude", config=config
        )

        # Extract agent ID from filename
        agent_id = PathLib(active_file).stem.replace("_active", "")

        return jsonify(
            {
                "success": True,
                "agent_id": agent_id,
                "redirect": f"/chat?agent={agent_id}",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to create matter: {str(e)}"}), 500


@bp.route(
    "/api/domains/<domain_name>/matters/<matter_id>/attachments/upload",
    methods=["POST"],
)
def upload_attachment(domain_name: str, matter_id: str) -> Any:
    """Upload a file attachment to a matter."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    matter_path = domain_path / matter_id
    attachments_dir = matter_path / "attachments"

    if not matter_path.exists():
        return jsonify({"error": "Matter not found"}), 404

    try:
        # Create attachments directory if it doesn't exist
        attachments_dir.mkdir(exist_ok=True)

        # Generate safe filename
        from werkzeug.utils import secure_filename

        original_filename = secure_filename(file.filename)

        # Handle duplicate filenames by adding a number
        base_name = Path(original_filename).stem
        extension = Path(original_filename).suffix
        filename = original_filename
        counter = 1

        while (attachments_dir / filename).exists():
            filename = f"{base_name}_{counter}{extension}"
            counter += 1

        # Save the file
        file_path = attachments_dir / filename
        file.save(str(file_path))

        # Get file info
        file_size = file_path.stat().st_size
        mime_type = file.content_type or "application/octet-stream"

        # Create initial metadata
        metadata = {
            "original_name": file.filename,
            "size": file_size,
            "mime_type": mime_type,
            "uploaded": datetime.now().isoformat() + "Z",
        }

        # Save metadata JSON
        metadata_path = attachments_dir / f"{Path(filename).stem}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Log the upload in the matter's activity log
        activity_log_path = matter_path / "activity_log.jsonl"
        log_entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "type": "attachment",
            "message": f"Uploaded attachment: {filename}",
            "user": "system",
        }

        with open(activity_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify(
            {
                "success": True,
                "filename": Path(filename).stem,  # Return stem for metadata endpoint
                "full_filename": filename,
                "size": file_size,
                "mime_type": mime_type,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to upload attachment: {str(e)}"}), 500


@bp.route(
    "/api/domains/<domain_name>/matters/<matter_id>/attachments/metadata",
    methods=["POST"],
)
def update_attachment_metadata(domain_name: str, matter_id: str) -> Any:
    """Update attachment metadata with title and notes."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    filename = data.get("filename", "").strip()
    title = data.get("title", "").strip()
    notes = data.get("notes", "").strip()

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    if not title:
        return jsonify({"error": "Title is required"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    matter_path = domain_path / matter_id
    attachments_dir = matter_path / "attachments"

    if not attachments_dir.exists():
        return jsonify({"error": "Attachments directory not found"}), 404

    try:
        # Find the metadata file
        metadata_path = attachments_dir / f"{filename}.json"

        if not metadata_path.exists():
            return jsonify({"error": "Attachment metadata not found"}), 404

        # Read existing metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Update with title and notes
        metadata["title"] = title
        if notes:
            metadata["description"] = notes
        metadata["modified"] = datetime.now().isoformat() + "Z"

        # Save updated metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Log the metadata update
        activity_log_path = matter_path / "activity_log.jsonl"
        log_entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "type": "update",
            "message": f"Updated attachment metadata: {title}",
            "user": "system",
        }

        with open(activity_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify({"success": True})

    except Exception as e:
        return (
            jsonify({"error": f"Failed to update attachment metadata: {str(e)}"}),
            500,
        )


@bp.route("/api/domains/<domain_name>/matters/<matter_id>", methods=["PUT"])
def update_matter(domain_name: str, matter_id: str) -> Any:
    """Update a matter's metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    matter_path = domain_path / matter_id
    matter_json_path = matter_path / "matter.json"

    if not matter_json_path.exists():
        return jsonify({"error": "Matter not found"}), 404

    try:
        # Read existing matter data
        with open(matter_json_path, "r", encoding="utf-8") as f:
            matter_data = json.load(f)

        # Track what was changed for the activity log
        changes = []

        # Update fields if provided
        if "title" in data and data["title"] != matter_data.get("title"):
            changes.append(f"title to '{data['title']}'")
            matter_data["title"] = data["title"]

        if "description" in data and data["description"] != matter_data.get(
            "description"
        ):
            changes.append("description")
            matter_data["description"] = data["description"]

        if "status" in data and data["status"] != matter_data.get("status"):
            old_status = matter_data.get("status", "unknown")
            changes.append(f"status from '{old_status}' to '{data['status']}'")
            matter_data["status"] = data["status"]

        if "priority" in data and data["priority"] != matter_data.get("priority"):
            old_priority = matter_data.get("priority", "unknown")
            changes.append(f"priority from '{old_priority}' to '{data['priority']}'")
            matter_data["priority"] = data["priority"]

        # Only update if there were changes
        if changes:
            # Add modified timestamp
            matter_data["modified"] = datetime.now().isoformat() + "Z"

            # Save updated matter data
            with open(matter_json_path, "w", encoding="utf-8") as f:
                json.dump(matter_data, f, indent=2, ensure_ascii=False)

            # Log the update in the activity log
            activity_log_path = matter_path / "activity_log.jsonl"
            log_entry = {
                "timestamp": datetime.now().isoformat() + "Z",
                "type": "update",
                "message": f"Updated {', '.join(changes)}",
                "user": "system",
            }

            with open(activity_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            return jsonify({"success": True, "matter_data": matter_data})
        else:
            return jsonify(
                {
                    "success": True,
                    "message": "No changes made",
                    "matter_data": matter_data,
                }
            )

    except Exception as e:
        return jsonify({"error": f"Failed to update matter: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/matters/<matter_id>/objectives", methods=["POST"])
def create_objective(domain_name: str, matter_id: str) -> Any:
    """Create a new objective in the specified matter."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    objective_name = data.get("name", "").strip()
    objective_content = data.get("objective", "").strip()
    if not objective_name:
        return jsonify({"error": "Objective name is required"}), 400
    if not objective_content:
        return jsonify({"error": "Objective content is required"}), 400

    # Validate objective name (alphanumeric with underscores)
    if not objective_name.replace("_", "").replace("-", "").isalnum():
        return (
            jsonify(
                {
                    "error": "Objective name must be alphanumeric with optional underscores or hyphens"
                }
            ),
            400,
        )

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    matter_path = domain_path / matter_id

    if not matter_path.exists():
        return jsonify({"error": "Matter not found"}), 404

    try:
        # Create objective directory
        objective_dir = matter_path / f"objective_{objective_name}"
        if objective_dir.exists():
            return jsonify({"error": "Objective already exists"}), 409

        objective_dir.mkdir(parents=True, exist_ok=True)

        # Create OBJECTIVE.md file
        objective_file = objective_dir / "OBJECTIVE.md"
        objective_file.write_text(objective_content, encoding="utf-8")

        # Log the objective creation in the matter's activity log
        activity_log_path = matter_path / "activity_log.jsonl"
        log_entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "type": "created",
            "message": f"Created objective: {objective_name}",
            "user": "system",
        }

        with open(activity_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify(
            {
                "success": True,
                "objective_name": objective_name,
                "objective_dir": f"objective_{objective_name}",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to create objective: {str(e)}"}), 500
