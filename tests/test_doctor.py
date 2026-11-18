"""dk doctor: missing tools fail, missing caches warn, each with its fix."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.dk.platforms.gc_mwcc import doctorcmd, objcache


class DoctorTest(unittest.TestCase):
    def test_empty_project_fails_with_fixes(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        cfg = SimpleNamespace(root=root, platform_options={})
        with mock.patch.object(objcache, "main_root", lambda cfg: root), \
                mock.patch("tools.dk.platforms.gc_mwcc.route.sweep_path", lambda cfg: root / "build/dk/sweep.json"), \
                mock.patch.dict("os.environ", {"DK_ORACLE": ""}):
            rows = doctorcmd.checks(cfg, None, "http://127.0.0.1:1")
        status = {what: s for s, what, _, _ in rows}
        self.assertEqual(status["tool objdiff-cli"], "FAIL")
        self.assertEqual(status["original game files"], "FAIL")
        self.assertEqual(status["compiler sweep"], "warn")
        self.assertEqual(status["compile cache"], "ok")
        self.assertTrue(all(fix for s, _, _, fix in rows if s != "ok"))

    def test_cache_mismatch_log_fails(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        (root / "build/dk/objcache").mkdir(parents=True)
        (root / "build/dk/objcache/MISMATCH.log").write_text("x\n")
        cfg = SimpleNamespace(root=root, platform_options={})
        with mock.patch.object(objcache, "main_root", lambda cfg: root), \
                mock.patch("tools.dk.platforms.gc_mwcc.route.sweep_path", lambda cfg: root / "nope"):
            rows = doctorcmd.checks(cfg, None, "")
        self.assertIn(("FAIL", "compile cache"), [(s, w) for s, w, _, _ in rows])


if __name__ == "__main__":
    unittest.main()
