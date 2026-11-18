import ast
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import List

from tests.helpers import git, make_git_project
from tools.dk.core import progress

REPO = Path(__file__).resolve().parents[1]


def linked_tool_args():
    """configure.py's linked_tool_args, without running configure.py itself."""
    tree = ast.parse((REPO / "configure.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "linked_tool_args")
    ns = {"Path": Path, "List": List, "is_windows": lambda: False}
    exec(compile(ast.Module([fn], []), "configure.py", "exec"), ns)
    return ns["linked_tool_args"]


@unittest.skipUnless((REPO / "configure.py").is_file(), "needs the project's configure.py (game projects only)")
class ConfigureInWorktreeTest(unittest.TestCase):
    def test_linked_tools_become_configure_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            main, tree = Path(tmp, "main"), Path(tmp, "tree")
            (main / "build" / "tools").mkdir(parents=True)
            (main / "build" / "compilers").mkdir()
            for name in ("dtk", "objdiff-cli", "wibo", "sjiswrap.exe"):
                (main / "build" / "tools" / name).write_text("")
            (tree / "build").mkdir(parents=True)
            fn = linked_tool_args()
            self.assertEqual(fn(main), [])                  # the main checkout downloads as usual
            (tree / "build" / "tools").symlink_to(main / "build" / "tools")
            (tree / "build" / "compilers").symlink_to(main / "build" / "compilers")
            got = fn(tree)
            self.assertEqual(got[got.index("--dtk") + 1], str((main / "build" / "tools" / "dtk").resolve()))
            self.assertEqual(got[got.index("--compilers") + 1], str((main / "build" / "compilers").resolve()))
            self.assertEqual(sorted(got[::2]), ["--compilers", "--dtk", "--objdiff", "--sjiswrap", "--wrapper"])


class StaleReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = make_git_project(Path(self.tmp.name) / "p")
        self.cfg.report.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.report.write_text('{"measures": {"total_code": "10", "matched_code": "5"}}')

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_report_older_than_head_is_flagged(self):
        self.assertIsNone(progress.stale(self.cfg))
        old = time.time() - 600
        os.utime(self.cfg.report, (old, old))
        why = progress.stale(self.cfg)
        self.assertIn("is older than HEAD", why)
        out = io.StringIO()
        with redirect_stdout(out):
            progress.run(self.cfg, SimpleNamespace(build=False))
        self.assertIn("warning: build/report.json is older than HEAD", out.getvalue())

    def test_a_newer_commit_after_the_report(self):
        (self.cfg.root / "src" / "b.txt").write_text("b\n")
        git(self.cfg.root, "add", "-A")
        later = int(self.cfg.report.stat().st_mtime) + 120
        env_date = f"{later} +0000"
        import subprocess
        subprocess.run(["git", "commit", "-q", "-m", "later"], cwd=self.cfg.root, check=True,
                       env={**os.environ, "GIT_COMMITTER_DATE": env_date, "GIT_AUTHOR_DATE": env_date})
        self.assertIsNotNone(progress.stale(self.cfg))


if __name__ == "__main__":
    unittest.main()
