# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for journal settings management.

Auto-discovered by ``think.call`` and mounted as ``sol call settings ...``.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer

from think.utils import require_solstone

app = typer.Typer(
    help="Journal settings — keys, providers, transcription, identity, and observer."
)


@app.callback()
def _require_up() -> None:
    require_solstone()


keys_app = typer.Typer(help="API key management.")
app.add_typer(keys_app, name="keys")
providers_app = typer.Typer(help="AI provider configuration.")
app.add_typer(providers_app, name="providers")
google_backend_app = typer.Typer(help="Google backend selection.")
app.add_typer(google_backend_app, name="google-backend")
vertex_app = typer.Typer(help="Vertex AI service account credentials.")
app.add_typer(vertex_app, name="vertex-credentials")
transcribe_app = typer.Typer(help="Transcription backend configuration.")
app.add_typer(transcribe_app, name="transcribe")
identity_app = typer.Typer(help="Journal owner identity.")
app.add_typer(identity_app, name="identity")
observer_app = typer.Typer(help="Observer capture settings.")
app.add_typer(observer_app, name="observer")


def _get_config():
    """Read journal config."""
    from think.utils import get_config

    return get_config()


def _write_config(config: dict) -> None:
    """Write journal config with indent=2, trailing newline, 0o600."""
    from think.utils import get_journal

    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)


def _provider_for_env_var(env_var: str) -> str | None:
    """Return the provider mapped to an API env var, if any."""
    from think.providers import PROVIDER_METADATA

    env_to_provider = {
        meta["env_key"]: name
        for name, meta in PROVIDER_METADATA.items()
        if "env_key" in meta
    }
    return env_to_provider.get(env_var)


def _validate_env_var_or_exit(env_var: str) -> None:
    """Exit if env_var is not a supported API key variable."""
    from apps.settings.routes import API_KEY_ENV_VARS

    if env_var not in API_KEY_ENV_VARS:
        typer.echo(
            f"Invalid env var: {env_var}. Must be one of: {', '.join(API_KEY_ENV_VARS)}",
            err=True,
        )
        raise typer.Exit(1)


def _set_provider_type(
    agent_type: str,
    provider: str | None,
    tier: int | None,
    backup: str | None,
) -> dict:
    """Validate and update the provider settings for a single agent type."""
    from think.providers import PROVIDER_REGISTRY

    config = _get_config()
    config.setdefault("providers", {})
    config["providers"].setdefault(agent_type, {})

    if provider is not None:
        if provider not in PROVIDER_REGISTRY:
            typer.echo(
                f"Invalid provider: {provider}. Must be one of: {', '.join(sorted(PROVIDER_REGISTRY.keys()))}",
                err=True,
            )
            raise typer.Exit(1)
        config["providers"][agent_type]["provider"] = provider

    if tier is not None:
        if tier not in {1, 2, 3}:
            typer.echo(f"Invalid tier: {tier}. Must be 1, 2, or 3.", err=True)
            raise typer.Exit(1)
        config["providers"][agent_type]["tier"] = tier

    if backup is not None:
        if backup not in PROVIDER_REGISTRY:
            typer.echo(
                f"Invalid backup provider: {backup}. Must be one of: {', '.join(sorted(PROVIDER_REGISTRY.keys()))}",
                err=True,
            )
            raise typer.Exit(1)
        config["providers"][agent_type]["backup"] = backup

    _write_config(config)
    return config["providers"][agent_type]


@app.command("show")
def show() -> None:
    """Show a summary of journal settings."""
    from apps.settings.routes import API_KEY_ENV_VARS
    from think.models import TYPE_DEFAULTS

    config = _get_config()
    providers_config = config.get("providers", {})
    type_settings = {}
    for agent_type in ("generate", "cogitate"):
        defaults = TYPE_DEFAULTS[agent_type]
        type_config = providers_config.get(agent_type, {})
        type_settings[agent_type] = {
            "provider": type_config.get("provider", defaults["provider"]),
            "tier": type_config.get("tier", defaults["tier"]),
            "backup": type_config.get("backup", defaults["backup"]),
        }

    summary = {
        "identity": config.get("identity", {}),
        "providers": {
            "generate": type_settings["generate"],
            "cogitate": type_settings["cogitate"],
            "google_backend": providers_config.get("google_backend", "auto"),
            "auth": providers_config.get("auth", {}),
            "key_validation": providers_config.get("key_validation", {}),
        },
        "transcribe": config.get("transcribe", {}),
        "observe": config.get("observe", {}),
        "keys": {k: bool(config.get("env", {}).get(k)) for k in API_KEY_ENV_VARS},
    }
    typer.echo(json.dumps(summary, indent=2))


@keys_app.command("show")
def keys_show() -> None:
    """Show configured API key status."""
    from apps.settings.routes import API_KEY_ENV_VARS

    config = _get_config()
    env_config = config.get("env", {})
    status = {k: bool(env_config.get(k)) for k in API_KEY_ENV_VARS}
    typer.echo(json.dumps(status, indent=2))


@keys_app.command("set")
def keys_set(
    env_var: str = typer.Argument(..., help="Environment variable to set."),
    value: str = typer.Argument(..., help="API key value."),
) -> None:
    """Set an API key in journal config."""
    from think.providers import validate_key

    _validate_env_var_or_exit(env_var)
    config = _get_config()
    config.setdefault("env", {})
    config["env"][env_var] = value
    os.environ[env_var] = value

    validation = None
    provider = _provider_for_env_var(env_var)
    if provider:
        config.setdefault("providers", {})
        config["providers"].setdefault("auth", {})
        config["providers"]["auth"][provider] = "api_key"
        validation = validate_key(provider, value)
        validation["timestamp"] = datetime.now(timezone.utc).isoformat()
        config["providers"].setdefault("key_validation", {})
        config["providers"]["key_validation"][provider] = validation

    _write_config(config)
    typer.echo(
        json.dumps(
            {"env_var": env_var, "set": True, "validation": validation},
            indent=2,
        )
    )


@keys_app.command("clear")
def keys_clear(
    env_var: str = typer.Argument(..., help="Environment variable to clear."),
) -> None:
    """Clear an API key from journal config."""
    _validate_env_var_or_exit(env_var)
    config = _get_config()
    env_config = config.setdefault("env", {})
    env_config.pop(env_var, None)
    os.environ.pop(env_var, None)

    provider = _provider_for_env_var(env_var)
    if provider:
        config.setdefault("providers", {})
        config["providers"].setdefault("auth", {})
        config["providers"]["auth"][provider] = "platform"
        config["providers"].setdefault("key_validation", {})
        config["providers"]["key_validation"].pop(provider, None)

    _write_config(config)
    typer.echo(json.dumps({"env_var": env_var, "cleared": True}, indent=2))


@keys_app.command("validate")
def keys_validate() -> None:
    """Re-validate all configured API keys."""
    from think.providers import PROVIDER_METADATA, validate_key
    from think.providers.google import validate_vertex_credentials

    config = _get_config()
    env_config = config.get("env", {})
    env_to_provider = {
        meta["env_key"]: name
        for name, meta in PROVIDER_METADATA.items()
        if "env_key" in meta
    }

    key_validation = {}
    for env_var, provider in env_to_provider.items():
        api_key = env_config.get(env_var, "")
        if api_key:
            result = validate_key(provider, api_key)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            key_validation[provider] = result

    providers_config = config.get("providers", {})
    if providers_config.get("google_backend") == "vertex" and providers_config.get(
        "vertex_credentials"
    ):
        result = validate_vertex_credentials(providers_config["vertex_credentials"])
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        key_validation["google"] = result

    config.setdefault("providers", {})
    config["providers"]["key_validation"] = key_validation
    _write_config(config)
    typer.echo(json.dumps({"key_validation": key_validation}, indent=2))


@providers_app.command("show")
def providers_show() -> None:
    """Show provider configuration."""
    from think.models import TYPE_DEFAULTS
    from think.providers import build_provider_status, get_provider_list

    config = _get_config()
    providers_config = config.get("providers", {})
    type_settings = {}
    for agent_type in ("generate", "cogitate"):
        defaults = TYPE_DEFAULTS[agent_type]
        type_config = providers_config.get(agent_type, {})
        type_settings[agent_type] = {
            "provider": type_config.get("provider", defaults["provider"]),
            "tier": type_config.get("tier", defaults["tier"]),
            "backup": type_config.get("backup", defaults["backup"]),
        }

    providers_list = get_provider_list()
    api_keys = {}
    for provider in providers_list:
        env_key = provider.get("env_key", "")
        api_keys[provider["name"]] = bool(os.getenv(env_key)) if env_key else False

    auth_config = providers_config.get("auth", {})
    auth = {
        provider["name"]: auth_config.get(provider["name"], "platform")
        for provider in providers_list
    }
    vertex_creds_path = providers_config.get("vertex_credentials")
    vertex_creds_configured = bool(
        vertex_creds_path and Path(vertex_creds_path).exists()
    )
    provider_status = build_provider_status(providers_list, vertex_creds_configured)
    result = {
        "providers": providers_list,
        "provider_status": provider_status,
        "generate": type_settings["generate"],
        "cogitate": type_settings["cogitate"],
        "api_keys": api_keys,
        "auth": auth,
        "key_validation": providers_config.get("key_validation", {}),
    }
    typer.echo(json.dumps(result, indent=2))


@providers_app.command("set-generate")
def providers_set_generate(
    provider: str | None = typer.Option(None, "--provider", help="Primary provider."),
    tier: int | None = typer.Option(None, "--tier", help="Tier (1, 2, or 3)."),
    backup: str | None = typer.Option(None, "--backup", help="Backup provider."),
) -> None:
    """Set generate provider defaults."""
    typer.echo(
        json.dumps(_set_provider_type("generate", provider, tier, backup), indent=2)
    )


@providers_app.command("set-cogitate")
def providers_set_cogitate(
    provider: str | None = typer.Option(None, "--provider", help="Primary provider."),
    tier: int | None = typer.Option(None, "--tier", help="Tier (1, 2, or 3)."),
    backup: str | None = typer.Option(None, "--backup", help="Backup provider."),
) -> None:
    """Set cogitate provider defaults."""
    typer.echo(
        json.dumps(_set_provider_type("cogitate", provider, tier, backup), indent=2)
    )


@providers_app.command("set-auth")
def providers_set_auth(
    provider: str = typer.Argument(..., help="Provider name."),
    mode: str = typer.Argument(..., help="Auth mode."),
) -> None:
    """Set provider auth mode."""
    from think.providers import PROVIDER_REGISTRY

    if provider not in PROVIDER_REGISTRY:
        typer.echo(f"Invalid provider in auth: {provider}", err=True)
        raise typer.Exit(1)
    if mode not in ("platform", "api_key"):
        typer.echo(
            f"Invalid auth mode: {mode}. Must be 'platform' or 'api_key'.",
            err=True,
        )
        raise typer.Exit(1)

    config = _get_config()
    config.setdefault("providers", {})
    config["providers"].setdefault("auth", {})
    config["providers"]["auth"][provider] = mode
    _write_config(config)
    typer.echo(json.dumps({provider: mode}, indent=2))


@google_backend_app.command("show")
def google_backend_show() -> None:
    """Show Google backend status."""
    config = _get_config()
    providers_config = config.get("providers", {})
    google_backend = providers_config.get("google_backend", "auto")
    vertex_creds_path = providers_config.get("vertex_credentials")
    vertex_configured = False
    vertex_email = ""
    if vertex_creds_path and Path(vertex_creds_path).exists():
        vertex_configured = True
        try:
            creds_data = json.loads(Path(vertex_creds_path).read_text())
            vertex_email = creds_data.get("client_email", "")
        except Exception:
            pass
    result = {
        "google_backend": google_backend,
        "vertex_credentials_configured": vertex_configured,
        "vertex_credentials_email": vertex_email,
    }
    typer.echo(json.dumps(result, indent=2))


@google_backend_app.command("set")
def google_backend_set(
    backend: str = typer.Argument(..., help="Google backend to use."),
) -> None:
    """Set the Google provider backend."""
    if backend not in ("auto", "aistudio", "vertex"):
        typer.echo(
            f"Invalid google_backend: {backend}. Must be 'auto', 'aistudio', or 'vertex'.",
            err=True,
        )
        raise typer.Exit(1)

    config = _get_config()
    config.setdefault("providers", {})
    config["providers"]["google_backend"] = backend
    _write_config(config)
    typer.echo(json.dumps({"google_backend": backend}, indent=2))


@vertex_app.command("show")
def vertex_credentials_show() -> None:
    """Show Vertex credential status without secrets."""
    config = _get_config()
    providers_config = config.get("providers", {})
    vertex_creds_path = providers_config.get("vertex_credentials")
    configured = False
    email = ""
    if vertex_creds_path and Path(vertex_creds_path).exists():
        configured = True
        try:
            creds_data = json.loads(Path(vertex_creds_path).read_text())
            email = creds_data.get("client_email", "")
        except Exception:
            pass
    validation = providers_config.get("key_validation", {}).get("google_vertex", {})
    result = {
        "configured": configured,
        "email": email,
        "path": vertex_creds_path or "",
        "validation": validation,
    }
    typer.echo(json.dumps(result, indent=2))


@vertex_app.command("import")
def vertex_credentials_import(
    file_path: str = typer.Argument(..., help="Path to service account JSON."),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip API validation of credentials."
    ),
) -> None:
    """Import Vertex service account credentials into the journal config."""
    from think.providers.google import validate_vertex_credentials
    from think.utils import get_journal

    source = Path(file_path)
    if not source.exists():
        typer.echo(f"Credential file not found: {file_path}", err=True)
        raise typer.Exit(1)

    try:
        creds_data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        typer.echo(f"Invalid JSON in credential file: {file_path}", err=True)
        raise typer.Exit(1)

    required_fields = ("type", "project_id", "client_email", "private_key")
    missing = [field for field in required_fields if field not in creds_data]
    if missing:
        typer.echo(f"Missing required fields: {', '.join(missing)}", err=True)
        raise typer.Exit(1)

    journal_root = Path(get_journal())
    creds_dir = journal_root / ".config"
    creds_dir.mkdir(parents=True, exist_ok=True)
    creds_file = creds_dir / "vertex-credentials.json"

    with open(creds_file, "w", encoding="utf-8") as f:
        json.dump(creds_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(creds_file, 0o600)

    config = _get_config()
    config.setdefault("providers", {})
    config["providers"]["vertex_credentials"] = str(creds_file)

    validation = None
    if not skip_validation:
        validation = validate_vertex_credentials(str(creds_file))
        validation["timestamp"] = datetime.now(timezone.utc).isoformat()
        config["providers"].setdefault("key_validation", {})
        config["providers"]["key_validation"]["google_vertex"] = validation

    _write_config(config)
    typer.echo(
        json.dumps(
            {
                "configured": True,
                "email": creds_data.get("client_email", ""),
                "path": str(creds_file),
                "validation": validation,
            },
            indent=2,
        )
    )


@vertex_app.command("clear")
def vertex_credentials_clear() -> None:
    """Clear stored Vertex credentials."""
    from think.utils import get_journal

    config = _get_config()
    config.setdefault("providers", {})
    old_path = config["providers"].get("vertex_credentials")
    if old_path:
        canonical = Path(get_journal()) / ".config" / "vertex-credentials.json"
        if Path(old_path).resolve() == canonical.resolve():
            try:
                canonical.unlink(missing_ok=True)
            except OSError:
                pass
        config["providers"].pop("vertex_credentials", None)
        config["providers"].setdefault("key_validation", {})
        config["providers"]["key_validation"].pop("google_vertex", None)

    _write_config(config)
    typer.echo(json.dumps({"configured": False}, indent=2))


@transcribe_app.command("show")
def transcribe_show() -> None:
    """Show transcription backend configuration."""
    from observe.transcribe import get_backend_list

    config = _get_config()
    transcribe_config = config.get("transcribe", {})
    backends = get_backend_list()
    api_keys = {}
    for backend in backends:
        env_key = backend.get("env_key")
        if env_key:
            api_keys[backend["name"]] = bool(os.getenv(env_key))
        else:
            api_keys[backend["name"]] = True
    result = {"backends": backends, "api_keys": api_keys, "config": transcribe_config}
    typer.echo(json.dumps(result, indent=2))


@transcribe_app.command("set-backend")
def transcribe_set_backend(
    backend: str = typer.Argument(..., help="Transcription backend."),
) -> None:
    """Set the transcription backend."""
    from observe.transcribe import BACKEND_REGISTRY

    if backend not in BACKEND_REGISTRY:
        typer.echo(
            f"Invalid backend: {backend}. Must be one of: {', '.join(sorted(BACKEND_REGISTRY.keys()))}",
            err=True,
        )
        raise typer.Exit(1)

    config = _get_config()
    config.setdefault("transcribe", {})
    config["transcribe"]["backend"] = backend
    _write_config(config)
    typer.echo(json.dumps(config["transcribe"], indent=2))


@transcribe_app.command("set")
def transcribe_set(
    enrich: bool | None = typer.Option(None, "--enrich/--no-enrich"),
    noise_upgrade: bool | None = typer.Option(
        None, "--noise-upgrade/--no-noise-upgrade"
    ),
) -> None:
    """Set transcription options."""
    config = _get_config()
    config.setdefault("transcribe", {})
    if enrich is not None:
        config["transcribe"]["enrich"] = enrich
    if noise_upgrade is not None:
        config["transcribe"]["noise_upgrade"] = noise_upgrade
    _write_config(config)
    typer.echo(json.dumps(config["transcribe"], indent=2))


@identity_app.command("show")
def identity_show() -> None:
    """Show journal identity config."""
    config = _get_config()
    identity = config.get("identity", {})
    typer.echo(json.dumps(identity, indent=2))


@identity_app.command("set")
def identity_set(
    name: str | None = typer.Option(None, "--name"),
    preferred: str | None = typer.Option(None, "--preferred"),
    bio: str | None = typer.Option(None, "--bio"),
    timezone_name: str | None = typer.Option(None, "--timezone"),
    pronouns: str | None = typer.Option(None, "--pronouns"),
    add_email: str | None = typer.Option(None, "--add-email"),
    remove_email: str | None = typer.Option(None, "--remove-email"),
    add_alias: str | None = typer.Option(None, "--add-alias"),
    remove_alias: str | None = typer.Option(None, "--remove-alias"),
) -> None:
    """Update journal owner identity."""
    config = _get_config()
    config.setdefault("identity", {})
    identity = config["identity"]

    if name is not None:
        identity["name"] = name
    if preferred is not None:
        identity["preferred"] = preferred
    if bio is not None:
        identity["bio"] = bio
    if timezone_name is not None:
        identity["timezone"] = timezone_name

    if pronouns is not None:
        try:
            identity["pronouns"] = json.loads(pronouns)
        except json.JSONDecodeError:
            typer.echo("Invalid JSON in pronouns", err=True)
            raise typer.Exit(1)

    if add_email is not None or remove_email is not None:
        emails = list(identity.get("email_addresses", []))
        if add_email is not None and add_email not in emails:
            emails.append(add_email)
        if remove_email is not None:
            emails = [email for email in emails if email != remove_email]
        identity["email_addresses"] = emails

    if add_alias is not None or remove_alias is not None:
        aliases = list(identity.get("aliases", []))
        if add_alias is not None and add_alias not in aliases:
            aliases.append(add_alias)
        if remove_alias is not None:
            aliases = [alias for alias in aliases if alias != remove_alias]
        identity["aliases"] = aliases

    _write_config(config)
    project_root = Path(__file__).resolve().parent.parent.parent
    subprocess.run(
        ["make", "skills"], cwd=project_root, check=False, capture_output=True
    )
    typer.echo(json.dumps(identity, indent=2))


@observer_app.command("show")
def observer_show() -> None:
    """Show observer configuration with defaults."""
    from apps.settings.routes import OBSERVE_TMUX_DEFAULTS

    config = _get_config()
    observe_config = config.get("observe", {})
    tmux_config = observe_config.get("tmux", {})
    result = {
        "tmux": {
            "enabled": tmux_config.get("enabled", OBSERVE_TMUX_DEFAULTS["enabled"]),
            "capture_interval": tmux_config.get(
                "capture_interval", OBSERVE_TMUX_DEFAULTS["capture_interval"]
            ),
        },
        "defaults": {"tmux": OBSERVE_TMUX_DEFAULTS},
    }
    typer.echo(json.dumps(result, indent=2))


@observer_app.command("set")
def observer_set(
    enabled: bool | None = typer.Option(None, "--enabled/--no-enabled"),
    capture_interval: int | None = typer.Option(None, "--capture-interval"),
) -> None:
    """Update observer capture settings."""
    from apps.settings.routes import OBSERVE_TMUX_DEFAULTS

    config = _get_config()
    config.setdefault("observe", {})
    config["observe"].setdefault("tmux", {})

    if capture_interval is not None:
        min_val = OBSERVE_TMUX_DEFAULTS["capture_interval_min"]
        max_val = OBSERVE_TMUX_DEFAULTS["capture_interval_max"]
        if capture_interval < min_val or capture_interval > max_val:
            typer.echo(
                f"tmux.capture_interval must be an integer between {min_val} and {max_val}",
                err=True,
            )
            raise typer.Exit(1)
        config["observe"]["tmux"]["capture_interval"] = capture_interval

    if enabled is not None:
        config["observe"]["tmux"]["enabled"] = enabled

    _write_config(config)
    typer.echo(json.dumps(config["observe"]["tmux"], indent=2))
