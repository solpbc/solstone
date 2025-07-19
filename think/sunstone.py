import argparse
import importlib
import logging
import os
import sys
from importlib.metadata import entry_points
from io import StringIO
from typing import List, Optional, Tuple

from .utils import setup_cli


def get_parser_help(module_name: str, func_name: str = "main") -> Tuple[str, str]:
    """Return description and usage for a command without triggering side effects."""
    # Skip self to prevent infinite recursion
    if module_name == "think.sunstone":
        return "Print available sunstone commands with descriptions.", ""

    logging.info(f"Getting parser help for module: {module_name}")

    pytest_env: Optional[str] = os.environ.get("PYTEST_CURRENT_TEST")
    os.environ["PYTEST_CURRENT_TEST"] = "sunstone-discover"
    try:
        logging.debug(f"Importing module: {module_name}")
        module = importlib.import_module(module_name)
        description = module.__doc__.splitlines()[0].strip() if module.__doc__ else ""
        logging.debug(f"Module description: {description}")

        if hasattr(module, "parse_args"):
            logging.debug("Module has parse_args, attempting to call it")
            try:
                module.parse_args()
            except Exception as e:
                logging.warning(f"parse_args failed: {e}")

        help_output = ""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            if hasattr(module, func_name):
                logging.debug(f"Calling {func_name} with --help")
                old_argv = sys.argv
                sys.argv = [module_name, "--help"]
                try:
                    getattr(module, func_name)()
                except SystemExit:
                    pass
                finally:
                    sys.argv = old_argv
                help_output = sys.stdout.getvalue()
                logging.debug(f"Help output length: {len(help_output)} chars")
        finally:
            sys.stdout = old_stdout

        usage = ""
        for line in help_output.splitlines():
            if line.lower().startswith("usage:"):
                parts = line.split()
                if len(parts) > 2:
                    usage = " ".join(parts[2:])
                logging.debug(f"Found usage: {usage}")
                break

        return description, usage
    except (ImportError, RuntimeError) as e:
        logging.warning(f"Exception while processing {module_name}: {e}")
        return f"Error: Could not load {module_name}", ""
    except Exception as e:
        logging.warning(f"Exception while processing {module_name}: {e}")
        return "", ""
    finally:
        if pytest_env is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = pytest_env


def discover_commands() -> List[Tuple[str, str, str]]:
    """Return available sunstone commands discovered via entry points."""
    logging.info("Starting command discovery")
    commands = []

    eps = entry_points()
    logging.debug(f"Entry points type: {type(eps)}")

    if hasattr(eps, "select"):
        scripts = eps.select(group="console_scripts")
    else:
        scripts = eps.get("console_scripts", [])  # type: ignore[attr-defined]

    logging.info(f"Found {len(scripts)} console scripts")

    sunstone_eps = [
        ep
        for ep in scripts
        if ep.value.startswith(("hear.", "see.", "think.", "dream."))
    ]
    logging.info(f"Found {len(sunstone_eps)} sunstone entry points")

    sunstone_eps.sort(key=lambda ep: ep.name)

    for ep in sunstone_eps:
        logging.debug(f"Processing entry point: {ep.name} -> {ep.value}")
        module_path, func_name = ep.value.split(":")
        description, usage = get_parser_help(module_path, func_name)
        commands.append((ep.name, description, usage))
        logging.debug(f"Added command: {ep.name}, desc: {description}, usage: {usage}")

    logging.info(f"Discovery complete, found {len(commands)} commands")
    return commands


def main() -> None:
    """Print available sunstone commands with descriptions."""
    parser = argparse.ArgumentParser(
        description="Print available sunstone commands with descriptions."
    )

    setup_cli(parser)

    logging.info("Starting sunstone command discovery")

    print("Scanning for available commands:\n")
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
