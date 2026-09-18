# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Static AST analysis proving only CLI and tests invoke publish_release."""

import ast
from pathlib import Path
import unittest


class TestInventory(unittest.TestCase):
    def test_publish_release_callers_are_strictly_isolated(self):
        repo_root = Path(__file__).parent.parent
        src_dir = repo_root / "src"
        tools_dir = repo_root / "tools"

        disallowed_callers = []

        for p in list(src_dir.rglob("*.py")) + list(tools_dir.rglob("*.py")):
            rel = p.relative_to(repo_root)
            # Allowed sites in src: cli.py, publish.py (definition)
            if str(rel) in ("src/solstone_platform/cli.py", "src/solstone_platform/publish.py"):
                continue

            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(rel))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "publish_release":
                        disallowed_callers.append(f"{rel}:{node.lineno}")
                    elif isinstance(func, ast.Attribute) and func.attr == "publish_release":
                        disallowed_callers.append(f"{rel}:{node.lineno}")

        self.assertEqual(disallowed_callers, [], f"Disallowed callers of publish_release found: {disallowed_callers}")


if __name__ == "__main__":
    unittest.main()
