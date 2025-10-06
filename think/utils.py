import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, NamedTuple, Optional

from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")

# Topic colors are now stored in each topic's JSON metadata file

AGENT_DIR = Path(__file__).parent.parent / "muse" / "agents"


class PromptContent(NamedTuple):
    """Container for prompt text and its resolved path."""

    text: str
    path: Path


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt file cannot be located."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Prompt file not found: {path}")


def _flatten_identity_to_template_vars(identity: dict[str, Any]) -> dict[str, str]:
    """Flatten identity config into template variables with uppercase-first versions.

    Parameters
    ----------
    identity:
        Identity configuration dictionary from get_config()['identity'].

    Returns
    -------
    dict[str, str]
        Template variables including flattened nested objects and uppercase-first versions.
        For example:
        - 'name' → identity['name']
        - 'pronouns_possessive' → identity['pronouns']['possessive']
        - 'Pronouns_possessive' → identity['pronouns']['possessive'].capitalize()
        - 'entity_name' → value of identity['entity'] field
        - 'entity_value' → description from entities.md for the entity
    """
    template_vars: dict[str, str] = {}

    # Flatten top-level and nested values
    for key, value in identity.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries with underscore separator
            for subkey, subvalue in value.items():
                var_name = f"{key}_{subkey}"
                template_vars[var_name] = str(subvalue)
                # Create uppercase-first version
                template_vars[var_name.capitalize()] = str(subvalue).capitalize()
        elif isinstance(value, (str, int, float, bool)):
            # Top-level scalar values
            template_vars[key] = str(value)
            # Create uppercase-first version
            template_vars[key.capitalize()] = str(value).capitalize()

    # Handle entity lookup
    entity_ref = identity.get("entity", "")
    if entity_ref:
        template_vars["entity_name"] = entity_ref
        template_vars["Entity_name"] = entity_ref  # No capitalize for entity name

        # Fetch entity description from entities.md
        try:
            load_dotenv()
            journal = os.getenv("JOURNAL_PATH")
            if journal:
                entities_path = Path(journal) / "entities.md"
                if entities_path.is_file():
                    from think.indexer import parse_entity_line

                    with open(entities_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parsed = parse_entity_line(line)
                            if parsed:
                                _, name, desc = parsed
                                # Match entity by name
                                if name == entity_ref:
                                    template_vars["entity_value"] = desc
                                    template_vars["Entity_value"] = desc.capitalize()
                                    break
        except Exception as exc:
            logging.debug("Failed to load entity description: %s", exc)

    # Add bio if present
    bio = identity.get("bio", "")
    if bio:
        template_vars["bio"] = bio
        template_vars["Bio"] = bio.capitalize()

    return template_vars


def load_prompt(
    name: str, base_dir: str | Path | None = None, *, include_journal: bool = False
) -> PromptContent:
    """Return the text contents and path for a ``.txt`` prompt file.

    Supports Python string.Template variable substitution using identity config
    from get_config()['identity']. Template variables include:
    - Top-level fields: $name, $preferred
    - Nested fields with underscores: $pronouns_possessive, $pronouns_subject
    - Uppercase-first versions: $Pronouns_possessive, $Name
    - Entity fields: $entity_name (entity reference), $entity_value (description)

    Parameters
    ----------
    name:
        Base filename of the prompt without the ``.txt`` suffix. If the suffix is
        included, it will not be duplicated.
    base_dir:
        Optional directory containing the prompt file. Defaults to the directory
        of this module when not provided.
    include_journal:
        If True, prepends the content of ``think/journal.txt`` to the requested
        prompt. Defaults to False.

    Returns
    -------
    PromptContent
        The prompt text (with surrounding whitespace removed and template variables
        substituted) and the resolved path to the ``.txt`` file.
    """

    if not name:
        raise ValueError("Prompt name must be provided")

    filename = name if name.endswith(".txt") else f"{name}.txt"
    prompt_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
    prompt_path = prompt_dir / filename
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - caller handles missing prompt
        raise PromptNotFoundError(prompt_path) from exc

    # Perform template substitution
    try:
        config = get_config()
        identity = config.get("identity", {})
        template_vars = _flatten_identity_to_template_vars(identity)

        # Use safe_substitute to avoid errors for undefined variables
        template = Template(text)
        text = template.safe_substitute(template_vars)
    except Exception as exc:
        # Log but don't fail - return original text if substitution fails
        logging.debug("Template substitution failed for %s: %s", prompt_path, exc)

    # Prepend journal content if requested
    if include_journal and name != "journal":
        journal_content = load_prompt("journal", base_dir=base_dir)
        text = f"{journal_content.text}\n\n{text}"

    return PromptContent(text=text, path=prompt_path)


def day_path(day: Optional[str] = None) -> Path:
    """Return absolute path for a day from ``JOURNAL_PATH`` environment variable.

    Parameters
    ----------
    day : str, optional
        Day in YYYYMMDD format. If None, uses today's date.

    Returns
    -------
    Path
        Absolute path to the day directory. Directory is created if it doesn't exist.

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH is not set.
    ValueError
        If day format is invalid.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    # Handle "today" case
    if day is None:
        day = datetime.now().strftime("%Y%m%d")
    elif not DATE_RE.fullmatch(day):
        raise ValueError("day must be in YYYYMMDD format")

    path = Path(journal) / day
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_dirs() -> dict[str, str]:
    """Return mapping of YYYYMMDD day names to absolute paths.

    Uses JOURNAL_PATH from environment (must be set via load_dotenv() or setup_cli()).

    Returns
    -------
    dict[str, str]
        Mapping of day folder names to their full paths.
        Example: {"20250101": "/path/to/journal/20250101", ...}

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH environment variable is not set.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    if not os.path.isdir(journal):
        return {}

    days: dict[str, str] = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def get_config() -> dict[str, Any]:
    """Return the journal configuration from config/journal.json.

    Returns
    -------
    dict
        Journal configuration with at least an 'identity' key containing
        name, preferred, pronouns, aliases, email_addresses, and timezone fields.
        Returns default empty structure if config file doesn't exist.

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH is not set.
    """
    # Default identity structure - defined once
    default_identity = {
        "name": "",
        "preferred": "",
        "pronouns": {
            "subject": "",
            "object": "",
            "possessive": "",
            "reflexive": "",
        },
        "aliases": [],
        "email_addresses": [],
        "timezone": "",
        "entity": "",
    }

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    config_path = Path(journal) / "config" / "journal.json"

    # Return default structure if file doesn't exist
    if not config_path.exists():
        return {"identity": default_identity.copy()}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Ensure identity section exists with all required fields
        if "identity" not in config:
            config["identity"] = {}

        # Fill in any missing fields with defaults
        for key, default in default_identity.items():
            if key not in config["identity"]:
                config["identity"][key] = default

        return config
    except (json.JSONDecodeError, OSError) as exc:
        # Log error but return default structure to avoid breaking callers
        logging.getLogger(__name__).warning(
            "Failed to load config from %s: %s", config_path, exc
        )
        return {"identity": default_identity.copy()}


def _append_task_log(dir_path: str | Path, message: str) -> None:
    """Append ``message`` to ``task_log.txt`` inside ``dir_path``."""
    path = Path(dir_path) / "task_log.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())}\t{message}\n")
    except Exception:
        pass


def day_log(day: str, message: str) -> None:
    """Convenience wrapper to log message for ``day``."""
    _append_task_log(str(day_path(day)), message)


def journal_log(message: str) -> None:
    """Append ``message`` to the journal's ``task_log.txt``."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        _append_task_log(journal, message)


def touch_health(name: str) -> None:
    """Update the journal's ``name`` heartbeat file.

    The journal path is read from ``JOURNAL_PATH`` in the environment.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return
    path = Path(journal) / "health" / f"{name}.up"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except Exception:
        pass


def setup_cli(parser: argparse.ArgumentParser, *, parse_known: bool = False):
    """Parse command line arguments and configure logging.

    The parser will be extended with ``-v``/``--verbose`` and ``-d``/``--debug`` flags. Environment
    variables from ``.env`` are loaded and ``JOURNAL_PATH`` is validated. The
    parsed arguments are returned. If ``parse_known`` is ``True`` a tuple of
    ``(args, extra)`` is returned using :func:`argparse.ArgumentParser.parse_known_args`.
    """

    load_dotenv()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    if parse_known:
        args, extra = parser.parse_known_args()
    else:
        args = parser.parse_args()
        extra = None

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.isdir(journal):
        parser.error("JOURNAL_PATH not set or invalid")

    return (args, extra) if parse_known else args


def get_topics() -> dict[str, dict[str, object]]:
    """Return available topics with metadata.

    Each key is the topic name. The value contains the ``path`` to the
    ``.txt`` file, the ``color`` from the metadata JSON, the file
    ``mtime`` and any keys loaded from a matching ``.json`` metadata file.
    """

    topics_dir = Path(__file__).parent / "topics"
    topics: dict[str, dict[str, object]] = {}
    for txt_path in sorted(topics_dir.glob("*.txt")):
        name = txt_path.stem
        mtime = int(txt_path.stat().st_mtime)
        info: dict[str, object] = {
            "path": str(txt_path),
            "mtime": mtime,
        }
        json_path = txt_path.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    info.update(data)
                    # Ensure color exists, fallback to a default if missing
                    if "color" not in info:
                        info["color"] = "#6c757d"  # Default gray color
            except Exception as exc:  # pragma: no cover - metadata optional
                logging.debug("Error reading %s: %s", json_path, exc)
                info["color"] = "#6c757d"  # Default gray color
        else:
            info["color"] = "#6c757d"  # Default gray color
        topics[name] = info
    return topics


def get_agent(persona: str = "default") -> dict:
    """Return complete agent configuration for a persona.

    Loads JSON configuration and instruction text, merges with runtime context.

    Parameters
    ----------
    persona:
        Name of the persona to load from agents/ directory.

    Returns
    -------
    dict
        Complete agent configuration including instruction, model, backend, etc.
    """
    config = {}

    # Load JSON config if exists
    json_path = AGENT_DIR / f"{persona}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Load instruction text
    txt_path = AGENT_DIR / f"{persona}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Agent persona not found: {persona}")
    prompt_data = load_prompt(persona, base_dir=AGENT_DIR, include_journal=True)
    config["instruction"] = prompt_data.text

    # Add runtime context (entities and domains)
    extra_parts = []

    # Add entities context
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        ent_path = Path(journal) / "entities.md"
        if ent_path.is_file():
            entities = ent_path.read_text(encoding="utf-8").strip()
            if entities:
                extra_parts.append("## Well-Known Entities\n" + entities)

        # Add domains to agent instructions
        try:
            from think.domains import get_domains

            domains = get_domains()
            if domains:
                domains_list = []
                for domain_name, info in sorted(domains.items()):
                    desc = str(info.get("description", "")).replace("\n", " ").strip()
                    if desc:
                        domains_list.append(f"- `{domain_name}`: {desc}")
                    else:
                        domains_list.append(f"- `{domain_name}`")
                extra_parts.append("## Available Domains\n" + "\n".join(domains_list))
        except Exception:
            pass  # Ignore if domains can't be loaded

    # Add topics to agent instructions
    topics = get_topics()
    if topics:
        topics_list = []
        for topic_name, info in sorted(topics.items()):
            desc = str(info.get("contains", "")).replace("\n", " ").strip()
            if desc:
                topics_list.append(f"- `{topic_name}`: {desc}")
            else:
                topics_list.append(f"- `{topic_name}`")
        extra_parts.append("## Available Topics\n" + "\n".join(topics_list))

    # Add current date/time
    now = datetime.now()
    try:
        import tzlocal

        local_tz = tzlocal.get_localzone()
        now_local = now.astimezone(local_tz)
        time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception:
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
    extra_parts.append(f"## Current Date and Time\n{time_str}")

    if extra_parts:
        config["extra_context"] = "\n\n".join(extra_parts).strip()

    # Set persona name
    config["persona"] = persona

    return config


def create_mcp_client(http_uri: str) -> Any:
    """Return a FastMCP HTTP client for Sunstone tools."""

    http_uri = http_uri.strip()
    if not http_uri:
        raise RuntimeError("MCP server URL not provided")

    from fastmcp import Client

    return Client(http_uri, timeout=15.0)


def parse_time_range(text: str) -> Optional[tuple[str, str, str]]:
    """Return ``(day, start, end)`` from a natural language time range.

    Parameters
    ----------
    text:
        Natural language description of a time range.

    Returns
    -------
    tuple[str, str, str] | None
        ``(day, start, end)`` if a single range within one day was detected.
        ``day`` is ``YYYYMMDD`` and ``start``/``end`` are ``HHMMSS``. ``None``
        if parsing fails.
    """

    try:
        result = timefhuman(text)
    except Exception as exc:  # pragma: no cover - unexpected library failure
        logging.info("timefhuman failed for %s: %s", text, exc)
        return None

    logging.debug("timefhuman(%s) -> %r", text, result)

    if len(result) != 1:
        logging.info("timefhuman did not return a single expression for %s", text)
        return None

    range_item = result[0]
    if not isinstance(range_item, tuple) or len(range_item) != 2:
        logging.info("Expected a range from %s but got %r", text, range_item)
        return None

    start_dt, end_dt = range_item
    if start_dt.date() != end_dt.date():
        logging.info("Range must be within a single day: %s -> %s", start_dt, end_dt)
        return None

    day = start_dt.strftime("%Y%m%d")
    start = start_dt.strftime("%H%M%S")
    end = end_dt.strftime("%H%M%S")
    return day, start, end


def get_raw_file(day: str, name: str) -> tuple[str, str, Any]:
    """Return raw file path, mime type and metadata for a transcript.

    Parameters
    ----------
    day:
        Day folder in ``YYYYMMDD`` format.
    name:
        Transcript filename such as ``HHMMSS_audio.json`` or
        ``HHMMSS_monitor_1_diff.json``.

    Returns
    -------
    tuple[str, str, Any]
        ``(path, mime_type, metadata)`` where ``path`` is relative to the day
        directory, ``mime_type`` is either ``image/png`` or ``audio/flac`` and
        ``metadata`` contains the parsed JSON data (empty on failure).
    """

    day_dir = day_path(day)
    json_path = day_dir / name

    if name.endswith("_audio.json"):
        # Audio files are stored as _raw.flac in the heard directory
        raw_name = name.replace("_audio.json", "_raw.flac")
        rel = f"heard/{raw_name}"
        mime = "audio/flac"
    elif name.endswith("_diff.json"):
        raw_name = name[:-5] + ".png"
        rel = f"seen/{raw_name}"
        mime = "image/png"
    else:
        raise ValueError(f"unsupported transcript name: {name}")

    meta: Any = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # pragma: no cover - optional metadata
        logging.debug("Failed to read %s", json_path)

    return rel, mime, meta


def load_entity_names(
    journal_path: str | Path | None = None,
    required: bool = False,
    domain: str | None = None,
    spoken: bool = False,
) -> str | list[str] | None:
    """Load entity names from entities.md for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    an entities.md file. When spoken=False (default), returns them as a
    comma-delimited string. When spoken=True, returns a list of shortened forms
    optimized for audio transcription.

    When spoken=True:
    - People: First name + nicknames in parens (e.g., "Jeremie Miller (Jer)" → ["Jeremie", "Jer"])
    - Companies: Name in parens if present, otherwise first word
    - Projects: Name in parens if present, otherwise full project name
    - Tools: Excluded entirely

    Args:
        journal_path: Path to journal directory. If None, uses JOURNAL_PATH env var.
        required: If True, raises FileNotFoundError when entities.md is missing.
                 If False, returns None when missing.
        domain: Optional domain name. If provided, loads from domains/{domain}/entities.md
                instead of the top-level entities.md file.
        spoken: If True, returns list of shortened forms for speech recognition.
                If False, returns comma-delimited string of full names.

    Returns:
        When spoken=False: Comma-delimited string of entity names (e.g., "John Smith, Acme Corp"),
                          or None if the file is not found and required=False.
        When spoken=True: List of shortened entity names for speech, or None if file not found.

    Raises:
        FileNotFoundError: If required=True and entities.md doesn't exist.
        ValueError: If journal_path is not provided and JOURNAL_PATH env var is not set.
    """
    if journal_path is None:
        load_dotenv()
        journal_path = os.getenv("JOURNAL_PATH")
        if not journal_path:
            raise ValueError("JOURNAL_PATH not set and no journal_path provided")

    journal_path = Path(journal_path)

    # Choose entities file based on domain parameter
    if domain:
        entities_path = journal_path / "domains" / domain / "entities.md"
    else:
        entities_path = journal_path / "entities.md"

    if not entities_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Required entities file not found: {entities_path}"
            )
        return None

    # Import here to avoid circular dependency
    from think.indexer import parse_entity_line

    # Parse entity names from the file
    entity_names = []
    spoken_names = []

    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_entity_line(line)
            if parsed:
                entity_type, name, _ = parsed

                # Skip tools when in spoken mode
                if spoken and entity_type == "Tool":
                    continue

                # For non-spoken mode, collect full names
                if not spoken:
                    if name and name not in entity_names:
                        entity_names.append(name)
                else:
                    # For spoken mode, extract shortened forms
                    if entity_type == "Person":
                        # Extract first name and nicknames
                        # "Jeremie Miller (Jer)" -> ["Jeremie", "Jer"]

                        # Get base name (without parens)
                        base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
                        first_name = base_name.split()[0] if base_name else None

                        # Add first name
                        if first_name and first_name not in spoken_names:
                            spoken_names.append(first_name)

                        # Extract and add nicknames from parens
                        paren_match = re.search(r"\(([^)]+)\)", name)
                        if paren_match:
                            nicknames = [
                                n.strip() for n in paren_match.group(1).split(",")
                            ]
                            for nick in nicknames:
                                if nick and nick not in spoken_names:
                                    spoken_names.append(nick)

                    elif entity_type == "Company":
                        # If there's a name in parens, use that
                        # Otherwise use first word (or full name if single word)
                        paren_match = re.search(r"\(([^)]+)\)", name)
                        if paren_match:
                            short_name = paren_match.group(1).strip()
                            if short_name and short_name not in spoken_names:
                                spoken_names.append(short_name)
                        else:
                            # Remove parens and get first word (or full if single word)
                            base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
                            words = base_name.split()
                            if len(words) > 1:
                                short_name = words[0]
                            else:
                                short_name = base_name
                            if short_name and short_name not in spoken_names:
                                spoken_names.append(short_name)

                    elif entity_type == "Project":
                        # If there's a name in parens, use just that
                        # Otherwise use full project name
                        paren_match = re.search(r"\(([^)]+)\)", name)
                        if paren_match:
                            short_name = paren_match.group(1).strip()
                        else:
                            short_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
                        if short_name and short_name not in spoken_names:
                            spoken_names.append(short_name)

    if spoken:
        return spoken_names if spoken_names else None
    else:
        if not entity_names:
            return None
        return ", ".join(entity_names)


def get_agents() -> dict[str, dict[str, Any]]:
    """Load agent metadata from think/agents directory.

    Returns:
        Dictionary mapping agent IDs to their metadata including:
        - title: Display title for the agent
        - All configuration fields from get_agent()
    """
    agents = {}
    agents_path = AGENT_DIR

    if not agents_path.exists():
        return agents

    for txt_path in sorted(agents_path.glob("*.txt")):
        agent_id = txt_path.stem
        try:
            # Use get_agent to load full configuration
            config = get_agent(agent_id)
            # Extract title for compatibility
            title = config.get("title", agent_id)
            # Return the config itself as the agent metadata
            agents[agent_id] = config
            agents[agent_id]["title"] = title
        except Exception:
            # Skip agents that can't be loaded
            pass

    return agents
