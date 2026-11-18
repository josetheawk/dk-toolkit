"""dk test: the project's own test folders (dk.toml [test] paths) run after the toolkit's tests/."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.dk.core import testcmd


class ProjectPathsTest(unittest.TestCase):
    def test_existing_paths_run_after_tests(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "dk.toml").write_text('[test]\npaths = ["mine", "missing"]\n')
            (root / "mine").mkdir()
            self.assertEqual(testcmd.project_paths(root), ["mine"])          # a missing folder is left out
            py = testcmd.commands(root, ["-x"], pytest=True)
            self.assertEqual(len(py), 1)
            self.assertEqual(py[0][-3:], ["tests", "mine", "-x"])
            ut = testcmd.commands(root, [], pytest=False)
            self.assertEqual([c[c.index("-s") + 1] for c in ut], ["tests", "mine"])

    def test_no_key_or_no_config_is_tests_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(testcmd.project_paths(root), [])
            (root / "dk.toml").write_text("[project]\nname = 'x'\n")
            self.assertEqual(testcmd.commands(root, [], pytest=True)[0][-1], "tests")


if __name__ == "__main__":
    unittest.main()
