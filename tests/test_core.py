import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_project
from tools.dk.core import build, cli, progress
from tools.dk.core.config import ConfigError, load
from tools.dk.core.doclint import GENERATED_MARK, lint


class ConfigTest(unittest.TestCase):
    def test_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp))
            self.assertEqual(cfg.game_id, "DEMO01")
            self.assertEqual(cfg.original, (Path(tmp) / "orig/main.dol").resolve())
            self.assertEqual(cfg.platform_options["symbols"], "config/symbols.txt")
            self.assertEqual(cfg.docs.playbook_file_lines, 10)

    def test_missing_file_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load(Path(tmp))
            (Path(tmp) / "dk.toml").write_text("[project]\nname='x'\n")
            with self.assertRaises(ConfigError):
                load(Path(tmp))


class DocLintTest(unittest.TestCase):
    def _problems(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp), files=files)
            return [p.message for p in lint(cfg)]

    def test_clean(self):
        files = {
            "AGENTS.md": "# Agents\nSee [playbook](playbook/README.md).\n",
            "playbook/README.md": "# Playbook\n[process](process.md)\n",
            "playbook/process.md": "# Process\n",
            "docs/generated/classes.md": f"{GENERATED_MARK} analyze rtti -->\n# Classes\n",
            "docs/upstream/x.md": "[dead](nowhere.md)\n",  # excluded
        }
        self.assertEqual(self._problems(files), [])

    def test_violations(self):
        files = {
            "AGENTS.md": "1\n2\n3\n4\n5\n6\n",
            "playbook/README.md": "Notes from demogame work\n",
            "playbook/long.md": "x\n" * 11,
            "docs/demo/a.md": "[gone](missing.md) [ok](https://example.com) [anchor](#top)\n",
            "docs/generated/raw.md": "# no header\n",
        }
        problems = self._problems(files)
        self.assertTrue(any("budget 5" in p for p in problems))         # AGENTS.md
        self.assertTrue(any("budget 10" in p for p in problems))        # one playbook file
        self.assertTrue(any("'DemoGame'" in p for p in problems))       # case-insensitive term
        self.assertTrue(any("broken link missing.md" in p for p in problems))
        self.assertTrue(any("generated doc" in p for p in problems))
        self.assertEqual(len([p for p in problems if "broken link" in p]), 1)


class BuildVerdictTest(unittest.TestCase):
    def test_hash_only_failure(self):
        self.assertTrue(build._hash_check_failed("[7/15] CHECK x\nFAILED: [code=1] build/GH2E69/ok\n"))
        self.assertTrue(build._hash_check_failed("FAILED: build/X/ok\n"))
        self.assertFalse(build._hash_check_failed("FAILED: build/X/src/a.o\nFAILED: build/X/ok\n"))
        self.assertFalse(build._hash_check_failed("all good\n"))


class ProgressTest(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp))
            cfg.report.parent.mkdir(parents=True)
            cfg.report.write_text(json.dumps({
                "measures": {"total_code": "200", "matched_code": "50", "complete_code": "20",
                             "total_data": "100", "total_functions": 4, "matched_functions": 1,
                             "total_units": 2},
                "categories": [{"id": "game", "measures": {}}],
            }))
            lines = progress.summary_lines(cfg, categories=True)
            self.assertEqual(len(lines), 1)  # empty category hidden
            self.assertIn(" 25.00% matched", lines[0])
            self.assertIn(" 10.00% linked", lines[0])
            self.assertIn("functions 1/4", lines[0])
            self.assertIn("units linked 0/2", lines[0])


class CliTest(unittest.TestCase):
    def test_overview_lists_every_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main([]), 0)
        from tools.dk.core.config import ConfigError, load
        try:
            platform = load().platform
        except ConfigError:
            platform = None                         # the toolkit repo itself: core commands only
        for c in cli._commands(platform):
            self.assertIn(c.name, out.getvalue())

    def test_unknown_command_is_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
            cli.main(["no-such-command"])
        self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()


class KillSessionTest(unittest.TestCase):
    def test_kills_children_that_changed_process_group(self):
        import os
        import subprocess
        import time
        from tools.dk.core.util import kill_session
        # The child leaves the session leader's process group (like wibo) but stays in its session.
        proc = subprocess.Popen(["python3", "-c", "import os,time,subprocess;"
                                 "c=subprocess.Popen(['python3','-c','import os,time;os.setpgid(0,0);time.sleep(60)']);"
                                 "print(c.pid,flush=True);time.sleep(60)"],
                                stdout=subprocess.PIPE, text=True, start_new_session=True)
        child = int(proc.stdout.readline())
        kill_session(proc.pid)
        proc.wait(timeout=5)
        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            try:
                os.waitpid(child, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(0.05)
        else:
            self.fail("grandchild in its own process group survived")


class IncludersTest(unittest.TestCase):
    def test_finds_includes_by_basename(self):
        from tools.dk.core.rmcmd import includers
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp), files={
                "include/game/a/Old.h": "struct Old;\n",
                "src/game/x.cpp": '#include "game/a/Old.h"\n',
                "src/game/y.cpp": "#include <game/a/OldNot.h>\n",
            })
            self.assertEqual(includers(cfg, "include/game/a/Old.h"), ["src/game/x.cpp"])


class ToolkitSyncTest(unittest.TestCase):
    def test_export_copies_owned_paths_only(self):
        import tempfile
        from pathlib import Path
        from tools.dk.core import toolkitcmd
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            src, dst = Path(a), Path(b)
            (src / "tools" / "dk" / "core").mkdir(parents=True)
            (src / "tools" / "dk" / "core" / "x.py").write_text("x = 1\n")
            (src / "playbook").mkdir()
            (src / "playbook" / "README.md").write_text("# rules\n")
            (src / "dk").write_text("#!/bin/sh\n")
            (src / "src").mkdir()
            (src / "src" / "game.cpp").write_text("// project code\n")       # not owned
            (dst / "playbook").mkdir()
            (dst / "playbook" / "stale.md").write_text("old\n")
            added, changed, removed = toolkitcmd.apply(src, dst)
            self.assertEqual(sorted(added), ["dk", "playbook/README.md", "tools/dk/core/x.py"])
            self.assertEqual(removed, ["playbook/stale.md"])
            self.assertFalse((dst / "src").exists())
            self.assertEqual(toolkitcmd.plan(src, dst), ([], [], []))
