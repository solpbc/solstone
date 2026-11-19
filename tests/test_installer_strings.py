# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Exercise build-time formatting where catalog text enters shell or JSON syntax."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.installer_strings import parse_pattern, render_installer_strings
from solstone_platform.refusals import Refusal
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallerStrings(unittest.TestCase):
    def render(self, source, pattern):
        key = "installer-example"
        return render_installer_strings(
            source, {key: parse_pattern(pattern, key)}, pseudo_locale=False, require_all=True,
        )

    def test_shell_and_printf_escaping_preserve_literal_text_and_binding(self):
        pattern = r'''{ "l'été \"$HOME\" at 100% \\ " }{ $arg1 }'''
        script = self.render(
            """printf "@{installer-example:p ["%s"]}@" 'user value'""", pattern,
        )
        proc = subprocess.run(["sh", "-c", script], check=True, capture_output=True, text=True)
        self.assertEqual(proc.stdout, """l'été "$HOME" at 100% \\ user value""")

    def test_json_list_message_escapes_quotes_and_format_characters(self):
        script = self.render(
            """printf '{"message":"@{installer-example:j}@"}'""", r'''{ "100% \"ready\" \\ test" }''',
        )
        proc = subprocess.run(["sh", "-c", script], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(proc.stdout), {"message": '100% "ready" \\ test'})

    def test_direct_json_notice_refuses_characters_that_require_runtime_escaping(self):
        with self.assertRaisesRegex(Refusal, "unsupported character in a JSON notice"):
            self.render('@{installer-example:n}@', 'a "quoted" notice')

    def test_production_refuses_the_scratch_locale(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            path = Path(tmp) / "install.sh"
            with self.assertRaisesRegex(Refusal, "production installer must use en-US"):
                build_installer(REPO_ROOT, path, is_production=True, pseudo_locale=True)
            self.assertFalse(path.exists())
