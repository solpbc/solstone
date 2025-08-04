from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from dotenv import load_dotenv

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
