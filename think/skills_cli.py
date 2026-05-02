# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""sol skills — install, uninstall, and inspect coding-agent skill bundles.

Two install modes:

- User mode (default): copies <repo>/skills/<name>/ into per-agent user
  config directories (~/.claude/skills/, ~/.codex/skills/, optionally
  ~/.gemini/skills/).
- Project mode (--project [DIR]): symlinks talent/ and apps/*/talent/
  SKILL.md sources into <DIR>/.claude/skills/ and <DIR>/.agents/skills/.

Subcommands: install, uninstall, list.

Note: this is a different namespace from `sol call skills`, which manages
owner-wide journal skill patterns under apps/skills/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from think.utils import get_project_root

ALL_AGENTS = "all"
PROJECT_MULTI_AGENT = "agents"
PROJECT_CLAUDE_SKILLS_REL = ".claude/skills"
PROJECT_AGENTS_SKILLS_REL = ".agents/skills"
GLOBAL_SKIP_MESSAGE = (
    "no AI coding agent config directories found — skipping skill registration"
)
SUBCOMMAND_DESCRIPTION = """User mode: copies/removes <repo>/skills/* in per-agent user config dirs.
Project mode: symlinks/removes talent and apps/*/talent skills under DIR.
User-mode install refuses symlink bundle targets; regular files inside bundle
dirs are replaced atomically.
Gemini is skipped silently in default --agent all when ~/.gemini/skills/ is
absent; explicit --agent gemini prints a skip line.
This is separate from `sol call skills`, which manages owner-wide journal
skill patterns."""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    display_name: str
    parent_dir: str
    skills_dir: str
    silent_when_default_all: bool


@dataclass(frozen=True)
class ActionRow:
    agent: str
    skill: str
    action: str
    path: Path
    reason: str | None = None


@dataclass
class InstallReport:
    rows: list[ActionRow]

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.action == "error")

    @property
    def all_skipped(self) -> bool:
        return bool(self.rows) and all(
            row.action == "skipped"
            and row.reason is not None
            and row.reason.startswith("config dir absent at ")
            for row in self.rows
        )


@dataclass(frozen=True)
class StatusRow:
    agent: str
    skill: str
    state: str
    path: Path


AGENTS: dict[str, AgentSpec] = {
    "claude": AgentSpec(
        name="claude",
        display_name="Claude Code",
        parent_dir=".claude",
        skills_dir=PROJECT_CLAUDE_SKILLS_REL,
        silent_when_default_all=False,
    ),
    "codex": AgentSpec(
        name="codex",
        display_name="Codex",
        parent_dir=".codex",
        skills_dir=".codex/skills",
        silent_when_default_all=False,
    ),
    "gemini": AgentSpec(
        name="gemini",
        display_name="Gemini",
        parent_dir=".gemini",
        skills_dir=".gemini/skills",
        silent_when_default_all=True,
    ),
}


def discover_user_bundles(repo_root: Path) -> list[Path]:
    """Return public user skill bundle directories."""
    root = repo_root / "skills"
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "SKILL.md").is_file()
    )


def discover_project_sources(repo_root: Path) -> list[Path]:
    """Return project skill source directories, rejecting duplicate names."""
    sources = sorted(
        [path.parent for path in (repo_root / "talent").glob("*/SKILL.md")]
        + [path.parent for path in (repo_root / "apps").glob("*/talent/*/SKILL.md")]
    )
    seen: dict[str, Path] = {}
    for source in sources:
        previous = seen.get(source.name)
        if previous is not None:
            raise ValueError(
                f"duplicate skill name {source.name!r}: {previous} and {source}"
            )
        seen[source.name] = source
    return sources


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=dst.parent, prefix=".tmp_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out_file:
            with src.open("rb") as in_file:
                shutil.copyfileobj(in_file, out_file)
        os.replace(temp_path, dst)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def _copy_tree_atomically(src_dir: Path, dst_dir: Path) -> str:
    existed = dst_dir.exists()
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(path for path in src_dir.rglob("*") if path.is_file()):
        _atomic_copy_file(src, dst_dir / src.relative_to(src_dir))
    return "replaced" if existed else "installed"


def _expand_user_agents(agents: list[str]) -> tuple[list[AgentSpec], bool]:
    default_all = ALL_AGENTS in agents
    names = list(AGENTS) if default_all else agents
    return [AGENTS[name] for name in names], default_all


def _project_targets(target: Path, agents: list[str]) -> list[tuple[str, Path]]:
    if ALL_AGENTS in agents:
        return [
            ("claude", target / PROJECT_CLAUDE_SKILLS_REL),
            (PROJECT_MULTI_AGENT, target / PROJECT_AGENTS_SKILLS_REL),
        ]
    if agents == ["claude"]:
        return [("claude", target / PROJECT_CLAUDE_SKILLS_REL)]
    agent = agents[0] if agents else ""
    raise ValueError(
        f"--agent {agent} is not supported with --project; use --agent all or --agent claude"
    )


def _missing_config_row(
    spec: AgentSpec, home: Path, default_all: bool
) -> ActionRow | bool | None:
    parent = home / spec.parent_dir
    if parent.exists():
        return None
    if default_all and spec.silent_when_default_all:
        return True
    return ActionRow(
        spec.name,
        "",
        "skipped",
        parent,
        reason=f"config dir absent at {parent}",
    )


def _append_write_error(
    rows: list[ActionRow],
    agent: str,
    skill: str,
    path: Path,
    exc: OSError,
) -> None:
    rows.append(ActionRow(agent, skill, "error", path, reason=str(exc)))


def install_user(repo_root: Path, home: Path, agents: list[str]) -> InstallReport:
    bundles = discover_user_bundles(repo_root)
    selected, default_all = _expand_user_agents(agents)
    rows: list[ActionRow] = []

    for spec in selected:
        skip = _missing_config_row(spec, home, default_all)
        if skip is True:
            continue
        if isinstance(skip, ActionRow):
            rows.append(skip)
            continue

        skills_root = home / spec.skills_dir
        try:
            skills_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _append_write_error(rows, spec.name, "", skills_root, exc)
            continue

        for bundle in bundles:
            target = skills_root / bundle.name
            if target.is_symlink():
                rows.append(
                    ActionRow(
                        spec.name,
                        bundle.name,
                        "error",
                        target,
                        reason="refusing to overwrite symlink",
                    )
                )
                continue
            if target.exists() and not target.is_dir():
                rows.append(
                    ActionRow(
                        spec.name,
                        bundle.name,
                        "error",
                        target,
                        reason="refusing to overwrite non-directory",
                    )
                )
                continue
            try:
                action = _copy_tree_atomically(bundle, target)
            except OSError as exc:
                _append_write_error(rows, spec.name, bundle.name, target, exc)
                continue
            rows.append(ActionRow(spec.name, bundle.name, action, target))

    return InstallReport(rows)


def uninstall_user(repo_root: Path, home: Path, agents: list[str]) -> InstallReport:
    bundles = discover_user_bundles(repo_root)
    selected, default_all = _expand_user_agents(agents)
    rows: list[ActionRow] = []

    for spec in selected:
        skip = _missing_config_row(spec, home, default_all)
        if skip is True:
            continue
        if isinstance(skip, ActionRow):
            rows.append(skip)
            continue

        skills_root = home / spec.skills_dir
        for bundle in bundles:
            target = skills_root / bundle.name
            if not target.exists() and not target.is_symlink():
                rows.append(
                    ActionRow(
                        spec.name,
                        bundle.name,
                        "skipped",
                        target,
                        reason="nothing to remove",
                    )
                )
                continue
            if target.is_symlink() or not target.is_dir():
                rows.append(
                    ActionRow(
                        spec.name,
                        bundle.name,
                        "error",
                        target,
                        reason="refusing to remove non-directory",
                    )
                )
                continue
            try:
                shutil.rmtree(target)
            except OSError as exc:
                _append_write_error(rows, spec.name, bundle.name, target, exc)
                continue
            rows.append(ActionRow(spec.name, bundle.name, "removed", target))

    return InstallReport(rows)


def _install_project_source(
    agent: str,
    source: Path,
    link_parent: Path,
    rows: list[ActionRow],
) -> None:
    link = link_parent / source.name
    target = os.path.relpath(source, link_parent)

    if link.is_symlink():
        if os.readlink(link) == target:
            rows.append(ActionRow(agent, source.name, "noop", link))
            return
        try:
            link.unlink()
            link.symlink_to(target)
        except OSError as exc:
            _append_write_error(rows, agent, source.name, link, exc)
            return
        rows.append(ActionRow(agent, source.name, "replaced", link))
        return

    if link.exists():
        rows.append(
            ActionRow(
                agent,
                source.name,
                "error",
                link,
                reason="refusing to overwrite non-symlink",
            )
        )
        return

    try:
        link.symlink_to(target)
    except OSError as exc:
        _append_write_error(rows, agent, source.name, link, exc)
        return
    rows.append(ActionRow(agent, source.name, "installed", link))


def _remove_stale_project_links(
    agent: str,
    link_parent: Path,
    source_names: set[str],
    rows: list[ActionRow],
) -> None:
    if not link_parent.is_dir():
        return
    for link in sorted(link_parent.iterdir()):
        if not link.is_symlink() or link.name in source_names:
            continue
        try:
            link.unlink()
        except OSError as exc:
            _append_write_error(rows, agent, link.name, link, exc)
            continue
        rows.append(ActionRow(agent, link.name, "removed", link, reason="stale"))


def install_project(repo_root: Path, target: Path, agents: list[str]) -> InstallReport:
    sources = discover_project_sources(repo_root)
    source_names = {source.name for source in sources}
    rows: list[ActionRow] = []

    for agent, link_parent in _project_targets(target, agents):
        try:
            link_parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _append_write_error(rows, agent, "", link_parent, exc)
            continue
        for source in sources:
            _install_project_source(agent, source, link_parent, rows)
        _remove_stale_project_links(agent, link_parent, source_names, rows)

    return InstallReport(rows)


def uninstall_project(
    repo_root: Path, target: Path, agents: list[str]
) -> InstallReport:
    sources = discover_project_sources(repo_root)
    rows: list[ActionRow] = []

    for agent, link_parent in _project_targets(target, agents):
        for source in sources:
            link = link_parent / source.name
            if not link.exists() and not link.is_symlink():
                rows.append(
                    ActionRow(
                        agent,
                        source.name,
                        "skipped",
                        link,
                        reason="nothing to remove",
                    )
                )
                continue
            if not link.is_symlink():
                rows.append(
                    ActionRow(
                        agent,
                        source.name,
                        "error",
                        link,
                        reason="refusing to remove non-symlink",
                    )
                )
                continue
            try:
                link.unlink()
            except OSError as exc:
                _append_write_error(rows, agent, source.name, link, exc)
                continue
            rows.append(ActionRow(agent, source.name, "removed", link))

    return InstallReport(rows)


def list_user_status(repo_root: Path, home: Path, agents: list[str]) -> list[StatusRow]:
    bundles = discover_user_bundles(repo_root)
    selected, _default_all = _expand_user_agents(agents)
    rows: list[StatusRow] = []

    for spec in selected:
        for bundle in bundles:
            target = home / spec.skills_dir / bundle.name
            state = "installed" if (target / "SKILL.md").is_file() else "not installed"
            rows.append(StatusRow(spec.name, bundle.name, state, target))

    return rows


def list_project_status(
    repo_root: Path, target: Path, agents: list[str]
) -> list[StatusRow]:
    sources = discover_project_sources(repo_root)
    rows: list[StatusRow] = []

    for agent, link_parent in _project_targets(target, agents):
        for source in sources:
            link = link_parent / source.name
            expected = os.path.relpath(source, link_parent)
            state = (
                "installed"
                if link.is_symlink() and os.readlink(link) == expected
                else "not installed"
            )
            rows.append(StatusRow(agent, source.name, state, link))

    return rows


def _print_report(report: InstallReport, operation: str) -> None:
    for row in report.rows:
        if row.action == "noop":
            continue
        if row.action == "error":
            print(f"error: {operation} {row.path}: {row.reason}", file=sys.stderr)
        elif row.action == "skipped":
            skill = f" {row.skill}" if row.skill else ""
            print(f"skipped {row.agent}{skill} ({row.reason})")
        elif row.action == "removed" and row.reason:
            print(f"removed {row.agent} {row.skill} ({row.reason}) -> {row.path}")
        else:
            print(f"{row.action} {row.agent} {row.skill} -> {row.path}")

    if report.all_skipped:
        print(GLOBAL_SKIP_MESSAGE)


def _print_status(rows: list[StatusRow]) -> None:
    print(f"{'agent':<10} {'skill':<20} state")
    for row in rows:
        print(f"{row.agent:<10} {row.skill:<20} {row.state}")


def _add_agent_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        choices=["claude", "codex", "gemini", ALL_AGENTS],
        default=ALL_AGENTS,
        help="agent registry to update",
    )


def _add_project_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        nargs="?",
        const=os.getcwd(),
        default=None,
        help="install project symlinks into DIR, or cwd when DIR is omitted",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sol skills",
        description=(
            "Install, uninstall, and inspect coding-agent skill bundles. "
            "This is separate from `sol call skills`, which manages owner-wide "
            "journal skill patterns. User mode refuses symlink bundle targets; "
            "gemini is skipped silently in --agent all when ~/.gemini is absent."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    install_parser = subparsers.add_parser(
        "install",
        help="install skill bundles",
        description=SUBCOMMAND_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_agent_option(install_parser)
    _add_project_option(install_parser)

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="uninstall skill bundles",
        description=SUBCOMMAND_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_agent_option(uninstall_parser)
    _add_project_option(uninstall_parser)

    list_parser = subparsers.add_parser("list", help="list skill install status")
    _add_project_option(list_parser)

    return parser


def _resolve_project_target(project_value: str | None) -> Path | None:
    if project_value is None:
        return None
    return Path(project_value).expanduser().resolve()


def _run_report(
    operation: str,
    action: Callable[[Path, Path, list[str]], InstallReport],
    repo_root: Path,
    location: Path,
    agents: list[str],
) -> int:
    report = action(repo_root, location, agents)
    _print_report(report, operation)
    return 1 if report.error_count else 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = Path(get_project_root())
    target = _resolve_project_target(args.project)

    try:
        if args.cmd == "install":
            if target is None:
                return _run_report(
                    "install", install_user, repo_root, Path.home(), [args.agent]
                )
            return _run_report(
                "install", install_project, repo_root, target, [args.agent]
            )

        if args.cmd == "uninstall":
            if target is None:
                return _run_report(
                    "uninstall", uninstall_user, repo_root, Path.home(), [args.agent]
                )
            return _run_report(
                "uninstall", uninstall_project, repo_root, target, [args.agent]
            )

        if args.cmd == "list":
            if target is None:
                _print_status(list_user_status(repo_root, Path.home(), [ALL_AGENTS]))
            else:
                _print_status(list_project_status(repo_root, target, [ALL_AGENTS]))
            return 0
    except (OSError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
