# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Exercise build-time formatting where catalog text enters shell or JSON syntax."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.installer_strings import (
    ENGLISH_CATALOG,
    TRANSLATIONS_DIR,
    locale_status,
    parse_pattern,
    read_locale_catalog,
    render_installer_strings,
    update_source_lock,
)
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

    def test_production_refuses_draft_locale(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            path = Path(tmp) / "install.sh"
            for loc in ("es-419", "de", "it", "te"):
                with self.assertRaisesRegex(Refusal, "production installer must use en-US"):
                    build_installer(REPO_ROOT, path, is_production=True, locale=loc)
                self.assertFalse(path.exists())

    def test_multiple_draft_locales_coexist_in_lock_and_track_staleness_independently(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            en_path = root / ENGLISH_CATALOG
            tr_dir = root / TRANSLATIONS_DIR
            en_path.parent.mkdir(parents=True, exist_ok=True)
            tr_dir.mkdir(parents=True, exist_ok=True)
            en_path.write_text(
                "installer-hello = hello { $arg1 }\ninstaller-keep = keep English\n",
                encoding="utf-8",
            )
            (tr_dir / "es-419.ftl").write_text("installer-hello = hola { $arg1 }\n", encoding="utf-8")
            (tr_dir / "de.ftl").write_text("installer-hello = hallo { $arg1 }\n", encoding="utf-8")
            (tr_dir / "it.ftl").write_text("installer-hello = ciao { $arg1 }\n", encoding="utf-8")

            for loc in ("es-419", "de", "it"):
                update_source_lock(root, acknowledge_locale=loc)

            for loc, word in (("es-419", "hola "), ("de", "hallo "), ("it", "ciao ")):
                merged = read_locale_catalog(root, locale=loc)
                self.assertEqual(merged["installer-hello"], [word, 1])
                self.assertEqual(merged["installer-keep"], ["keep English"])
                st = locale_status(root, loc)
                self.assertEqual(st["fresh"], ["installer-hello"])
                self.assertEqual(st["stale"], [])
                self.assertEqual(st["missing"], ["installer-keep"])

            # Absent draft locale (`te`) falls back cleanly to English.
            merged_te = read_locale_catalog(root, locale="te")
            self.assertEqual(merged_te["installer-hello"], ["hello ", 1])

            # Editing English and running lock without --acknowledge marks every draft locale stale.
            en_path.write_text(
                "installer-hello = hello again { $arg1 }\ninstaller-keep = keep English\n",
                encoding="utf-8",
            )
            update_source_lock(root)
            for loc in ("es-419", "de", "it"):
                st_after = locale_status(root, loc)
                self.assertEqual(st_after["fresh"], [])
                self.assertEqual(st_after["stale"], ["installer-hello"])

            # Re-acknowledging only `de` leaves `es-419` and `it` stale.
            update_source_lock(root, acknowledge_locale="de")
            self.assertEqual(locale_status(root, "de")["fresh"], ["installer-hello"])
            self.assertEqual(locale_status(root, "es-419")["stale"], ["installer-hello"])
            self.assertEqual(locale_status(root, "it")["stale"], ["installer-hello"])

    def test_draft_locale_overlays_translations_falls_back_and_tracks_staleness(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            en_path = root / ENGLISH_CATALOG
            tr_dir = root / TRANSLATIONS_DIR
            en_path.parent.mkdir(parents=True, exist_ok=True)
            tr_dir.mkdir(parents=True, exist_ok=True)
            en_path.write_text(
                "installer-hello = hello { $arg1 }\ninstaller-keep = keep English\n",
                encoding="utf-8",
            )
            (tr_dir / "es-419.ftl").write_text(
                "installer-hello = hola { $arg1 }\n",
                encoding="utf-8",
            )

            merged = read_locale_catalog(root, locale="es-419")
            self.assertEqual(merged["installer-hello"], ["hola ", 1])
            self.assertEqual(merged["installer-keep"], ["keep English"])

            update_source_lock(root, acknowledge_locale="es-419")
            st = locale_status(root, "es-419")
            self.assertEqual(st["fresh"], ["installer-hello"])
            self.assertEqual(st["stale"], [])
            self.assertEqual(st["missing"], ["installer-keep"])

            # Another editor updates English prose and runs lock without --acknowledge es-419.
            en_path.write_text(
                "installer-hello = hello again { $arg1 }\ninstaller-keep = keep English\n",
                encoding="utf-8",
            )
            update_source_lock(root)
            st_after = locale_status(root, "es-419")
            self.assertEqual(st_after["fresh"], [])
            self.assertEqual(st_after["stale"], ["installer-hello"])
            self.assertEqual(st_after["missing"], ["installer-keep"])

    def test_draft_locale_refuses_unknown_id_and_argument_mismatch(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            en_path = root / ENGLISH_CATALOG
            tr_dir = root / TRANSLATIONS_DIR
            en_path.parent.mkdir(parents=True, exist_ok=True)
            tr_dir.mkdir(parents=True, exist_ok=True)
            en_path.write_text("installer-hello = hello { $arg1 }\n", encoding="utf-8")

            (tr_dir / "es-419.ftl").write_text("installer-unknown = hola\n", encoding="utf-8")
            with self.assertRaisesRegex(Refusal, "unknown message IDs"):
                read_locale_catalog(root, locale="es-419")

            (tr_dir / "es-419.ftl").write_text("installer-hello = hola sin argumento\n", encoding="utf-8")
            with self.assertRaisesRegex(Refusal, "argument mismatch"):
                read_locale_catalog(root, locale="es-419")
