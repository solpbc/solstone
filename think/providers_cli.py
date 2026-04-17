# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import think.models as models
import think.providers as providers
import think.utils as think_utils
from think.models import TIER_FLASH, TIER_LITE, TIER_PRO
from think.utils import get_journal, require_solstone, setup_cli


def _check_generate(provider_name: str, tier: int, timeout: int) -> tuple[str, str]:
    env_key = providers.PROVIDER_METADATA[provider_name]["env_key"]
    if env_key and not os.getenv(env_key):
        label = providers.PROVIDER_METADATA[provider_name]["label"]
        return "skip", f"{label} not configured (no {env_key})"
    if not env_key:
        result = providers.validate_key(provider_name, "")
        if not result.get("valid"):
            return (
                "skip",
                f"Ollama not reachable ({result.get('error', 'unreachable')})",
            )
    try:
        module = providers.get_provider_module(provider_name)
        model = models.PROVIDER_DEFAULTS[provider_name][tier]
        result = module.run_generate(
            contents="Say OK",
            model=model,
            temperature=0,
            max_output_tokens=16,
            system_instruction=None,
            json_output=False,
            thinking_budget=None,
            timeout_s=timeout,
        )
        text = result.get("text", "") if isinstance(result, dict) else ""
        if text:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage:
                from think.models import log_token_usage

                log_token_usage(
                    model=models.PROVIDER_DEFAULTS[provider_name][tier],
                    usage=usage,
                    context="health.check.generate",
                    type="generate",
                )
            return "ok", "OK"
        return "fail", "FAIL: empty response text"
    except Exception as exc:
        return "fail", f"FAIL: {exc}"


async def _check_cogitate(
    provider_name: str, tier: int, timeout: int
) -> tuple[str, str]:
    env_key = providers.PROVIDER_METADATA[provider_name]["env_key"]
    if env_key and not os.getenv(env_key):
        label = providers.PROVIDER_METADATA[provider_name]["label"]
        return "skip", f"{label} not configured (no {env_key})"
    if not env_key:
        result = providers.validate_key(provider_name, "")
        if not result.get("valid"):
            return (
                "skip",
                f"Ollama not reachable ({result.get('error', 'unreachable')})",
            )
    binary = providers.PROVIDER_METADATA[provider_name].get("cogitate_cli", "")
    if binary and not shutil.which(binary):
        return "skip", f"{binary} CLI not installed"
    try:
        module = providers.get_provider_module(provider_name)
        model = models.PROVIDER_DEFAULTS[provider_name][tier]
        result = await asyncio.wait_for(
            module.run_cogitate(
                config={"prompt": "Say OK", "model": model}, on_event=None
            ),
            timeout=timeout,
        )
        return ("ok", "OK") if result else ("fail", "FAIL: empty response")
    except asyncio.TimeoutError:
        return "fail", f"FAIL: timed out after {timeout}s"
    except Exception as exc:
        return "fail", f"FAIL: {exc}"


async def _run_check(args: argparse.Namespace) -> None:
    targeted_pairs = None
    lock_fd = None
    if args.targeted and not args.provider and not args.tier:
        import fcntl

        from think.models import TYPE_DEFAULTS, get_backup_provider

        targeted_pairs = set()
        providers_config = think_utils.get_config().get("providers", {})
        for talent_type, defaults in TYPE_DEFAULTS.items():
            type_config = providers_config.get(talent_type, {})
            provider = type_config.get("provider", defaults["provider"])
            tier = type_config.get("tier", defaults["tier"])
            targeted_pairs.add((provider, tier))
            backup = get_backup_provider(talent_type)
            if backup:
                targeted_pairs.add((backup, tier))
        health_dir = Path(get_journal()) / "health"
        health_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = open(health_dir / "recheck.lock", "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_fd.close()
            return

    provider_names = args.provider or list(providers.PROVIDER_REGISTRY.keys())
    for provider_name in provider_names:
        if provider_name not in providers.PROVIDER_REGISTRY:
            available = ", ".join(providers.PROVIDER_REGISTRY)
            print(
                f"Unknown provider: {provider_name}. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)

    interfaces = [args.interface] if args.interface else ["generate", "cogitate"]
    tier_names = {TIER_PRO: "pro", TIER_FLASH: "flash", TIER_LITE: "lite"}
    tiers = [args.tier] if args.tier else [TIER_PRO, TIER_FLASH, TIER_LITE]
    provider_width = max(len(name) for name in provider_names)
    model_width = max(
        len(models.PROVIDER_DEFAULTS[p][t]) for p in provider_names for t in tiers
    )
    interface_width = max(len(name) for name in interfaces)
    results: list[dict[str, object]] = []
    counts = {"total": 0, "passed": 0, "skipped": 0, "failed": 0}
    cache: dict[tuple[str, str, str], tuple[str, str, str]] = {}

    for provider_name in provider_names:
        for tier in tiers:
            if (
                targeted_pairs is not None
                and (provider_name, tier) not in targeted_pairs
            ):
                continue
            model = models.PROVIDER_DEFAULTS[provider_name][tier]
            for interface_name in interfaces:
                start = time.perf_counter()
                cache_key = (provider_name, model, interface_name)
                reused_from = None
                if cache_key in cache:
                    status, message, reused_from = cache[cache_key]
                    elapsed_s = 0.0
                elif interface_name == "generate":
                    status, message = _check_generate(provider_name, tier, args.timeout)
                    elapsed_s = time.perf_counter() - start
                    cache[cache_key] = (status, message, tier_names[tier])
                else:
                    status, message = await _check_cogitate(
                        provider_name, tier, args.timeout
                    )
                    elapsed_s = time.perf_counter() - start
                    cache[cache_key] = (status, message, tier_names[tier])

                result = {
                    "provider": provider_name,
                    "tier": tier_names[tier],
                    "model": model,
                    "interface": interface_name,
                    "ok": status != "fail",
                    "status": status,
                    "message": str(message),
                    "elapsed_s": round(elapsed_s, 1),
                }
                if reused_from:
                    result["reused_from"] = reused_from
                results.append(result)
                counts["total"] += 1
                counts[
                    "passed"
                    if status == "ok"
                    else "skipped"
                    if status == "skip"
                    else "failed"
                ] += 1
                if not args.json:
                    mark = (
                        "="
                        if reused_from
                        else "✓"
                        if status == "ok"
                        else "-"
                        if status == "skip"
                        else "✗"
                    )
                    detail = (
                        f"{message} (={reused_from})" if reused_from else str(message)
                    )
                    print(
                        f"{mark} {provider_name:<{provider_width}}  {tier_names[tier]:<5}  "
                        f"{model:<{model_width}}  {interface_name:<{interface_width}}  "
                        f"{detail} ({elapsed_s:.1f}s)"
                    )

    payload = {
        "results": results,
        "summary": counts,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    health_dir = Path(get_journal()) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "talents.json").write_text(json.dumps(payload, indent=2))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"{counts['total']} checks: {counts['passed']} passed, "
            f"{counts['skipped']} skipped, {counts['failed']} failed"
        )
    if lock_fd is not None:
        lock_fd.close()
    sys.exit(1 if counts["failed"] else 0)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Provider health commands")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    check_parser = subparsers.add_parser("check", help="Check AI provider connectivity")
    check_parser.add_argument(
        "--provider",
        action="append",
        help=f"Provider to check (repeatable). Available: {', '.join(providers.PROVIDER_REGISTRY.keys())}",
    )
    check_parser.add_argument(
        "--interface", choices=["generate", "cogitate"], default=None
    )
    check_parser.add_argument("--timeout", type=int, default=30)
    check_parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
    check_parser.add_argument("--json", action="store_true")
    check_parser.add_argument("--targeted", action="store_true")
    args = setup_cli(parser)
    require_solstone()
    await _run_check(args)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
