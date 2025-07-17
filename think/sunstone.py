import importlib
import os
import sys
from importlib.metadata import entry_points
from io import StringIO
from typing import List, Optional, Tuple


def get_parser_help(module_name: str, func_name: str = "main") -> Tuple[str, str]:
    """Return description and usage for a command without triggering side effects."""
    pytest_env: Optional[str] = os.environ.get("PYTEST_CURRENT_TEST")
    os.environ["PYTEST_CURRENT_TEST"] = "sunstone-discover"
    try:
        module = importlib.import_module(module_name)
        description = module.__doc__.splitlines()[0].strip() if module.__doc__ else ""

        if hasattr(module, "parse_args"):
            try:
                module.parse_args()
            except Exception:
                pass

        help_output = ""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            if hasattr(module, func_name):
                old_argv = sys.argv
                sys.argv = [module_name, "--help"]
                try:
                    getattr(module, func_name)()
                except SystemExit:
                    pass
                finally:
                    sys.argv = old_argv
                help_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        usage = ""
        for line in help_output.splitlines():
            if line.lower().startswith("usage:"):
                parts = line.split()
                if len(parts) > 2:
                    usage = " ".join(parts[2:])
                break

        return description, usage
    except Exception:
        return "", ""
    finally:
        if pytest_env is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = pytest_env


def discover_commands() -> List[Tuple[str, str, str]]:
    """Return available sunstone commands discovered via entry points."""
    commands = []

    eps = entry_points()
    if hasattr(eps, "select"):
        scripts = eps.select(group="console_scripts")
    else:
        scripts = eps.get("console_scripts", [])  # type: ignore[attr-defined]

    sunstone_eps = [
        ep for ep in scripts if ep.value.startswith(("hear.", "see.", "think.", "dream."))
    ]
    sunstone_eps.sort(key=lambda ep: ep.name)

    for ep in sunstone_eps:
        module_path, func_name = ep.value.split(":")
        description, usage = get_parser_help(module_path, func_name)
        commands.append((ep.name, description, usage))

    return commands


def main() -> None:
    """Print available sunstone commands with descriptions."""
    print("Available commands:\n")
    for name, desc, usage in discover_commands():
        if usage:
            print(f"{name} {usage}")
        else:
            print(name)
        if desc:
            print(f"    {desc}")
        print()


if __name__ == "__main__":
    main()
