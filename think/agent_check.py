# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import os
import sys
import time

from dotenv import load_dotenv

from think.models import PROVIDER_DEFAULTS, TIER_LITE
from think.providers import PROVIDER_METADATA, PROVIDER_REGISTRY, get_provider_module
from think.providers.cli import check_cli_binary

COGITATE_BINARIES = {
    "anthropic": "claude",
    "openai": "codex",
    "google": "gemini",
}


def check_generate(provider_name: str, timeout: int) -> tuple[bool, str]:
    env_key = PROVIDER_METADATA[provider_name]["env_key"]
    if not os.getenv(env_key):
        return False, f"FAIL: {env_key} not set"

    try:
        module = get_provider_module(provider_name)
        model = PROVIDER_DEFAULTS[provider_name][TIER_LITE]
        result = module.run_generate(
            contents="Say OK",
            model=model,
            temperature=0,
            max_output_tokens=16,
            system_instruction=None,
            json_output=False,
            thinking_budget=0,
            timeout_s=timeout,
        )
        text = result.get("text", "") if isinstance(result, dict) else ""
        if text:
            return True, "OK"
        return False, "FAIL: empty response text"
    except Exception as exc:
        return False, f"FAIL: {exc}"


def check_cogitate(provider_name: str) -> tuple[bool, str]:
    try:
        binary_name = COGITATE_BINARIES[provider_name]
        check_cli_binary(binary_name)
        return True, f"OK ({binary_name} found)"
    except Exception:
        binary_name = COGITATE_BINARIES.get(provider_name, provider_name)
        return False, f"FAIL: {binary_name} not found"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AI provider connectivity")
    parser.add_argument(
        "--provider",
        action="append",
        help=f"Provider to check (repeatable). Available: {', '.join(PROVIDER_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--interface",
        choices=["generate", "cogitate"],
        default=None,
        help="Interface to check (default: both)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for generate checks (default: 30)",
    )
    args = parser.parse_args()
    load_dotenv()

    if args.provider:
        providers = args.provider
        for provider_name in providers:
            if provider_name not in PROVIDER_REGISTRY:
                available = ", ".join(PROVIDER_REGISTRY.keys())
                print(
                    f"Unknown provider: {provider_name}. Available providers: {available}",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        providers = list(PROVIDER_REGISTRY.keys())

    interfaces = [args.interface] if args.interface else ["generate", "cogitate"]

    provider_width = max(len(name) for name in providers) if providers else 0
    interface_width = max(len(name) for name in interfaces) if interfaces else 0

    total = 0
    passed = 0
    failed = 0

    for provider_name in providers:
        for interface_name in interfaces:
            start = time.perf_counter()
            if interface_name == "generate":
                ok, message = check_generate(provider_name, args.timeout)
            else:
                ok, message = check_cogitate(provider_name)
            elapsed_s = time.perf_counter() - start

            mark = "✓" if ok else "✗"
            print(
                f"{mark} "
                f"{provider_name:<{provider_width}}  "
                f"{interface_name:<{interface_width}}  "
                f"{message} ({elapsed_s:.1f}s)"
            )

            total += 1
            if ok:
                passed += 1
            else:
                failed += 1

    print(f"{total} checks: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
