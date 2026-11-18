import io
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from tests.helpers import DK_TOML, git
from tools.dk.core import toolkitcmd
from tools.dk.core.command import OK, USAGE
from tools.dk.core.config import load


def repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "dk test")
    git(root, "config", "user.email", "dk@test.invalid")
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start")
    return root


TOOLKIT = {"dk": "#!/bin/sh\n", "tools/dk/core/a.py": "a = 1\n", "tools/dk/platforms/other/p.py": "other = 1\n",
           "playbook/README.md": "# rules\n"}


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        top = Path(self.tmp.name)
        self.toolkit = repo(top / "toolkit", TOOLKIT)
        self.project = repo(top / "project", {**TOOLKIT, "src/game.cpp": "// game\n",
                                              "dk.toml": DK_TOML.format(sha1="0" * 40)
                                              + '\n[toolkit]\n# the shared repo\nrepo = "../toolkit"\n'})
        self.cfg = load(self.project)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cmd(self, action, force=False, set_=False):
        out = io.StringIO()
        with redirect_stdout(out):
            code = toolkitcmd.run(self.cfg, SimpleNamespace(action=action, force=force, set=set_))
        return code, out.getvalue()

    def commit(self, root: Path, msg: str):
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", msg)

    def test_no_base_refuses_until_one_is_recorded(self):
        code, out = self.run_cmd("export")
        self.assertEqual(code, USAGE)
        self.assertIn("no sync base recorded", out)
        self.assertEqual(self.run_cmd("base", set_=True)[0], OK)
        base = tomllib.loads((self.project / "dk.toml").read_text())["toolkit"]
        self.assertEqual(base["base"], toolkitcmd.head(self.toolkit))
        self.assertEqual(base["repo"], "../toolkit")                   # other keys kept
        self.assertIn("# the shared repo", (self.project / "dk.toml").read_text())

    def test_base_set_refuses_while_the_copies_differ(self):
        (self.project / "tools/dk/core/a.py").write_text("a = 2\n")
        self.assertEqual(self.run_cmd("base", set_=True)[0], USAGE)

    def test_export_refuses_when_another_project_exported_since(self):
        self.run_cmd("base", set_=True)
        self.commit(self.project, "base")
        (self.toolkit / "tools/dk/platforms/third/t.py").parent.mkdir(parents=True)
        (self.toolkit / "tools/dk/platforms/third/t.py").write_text("third = 1\n")   # another project's export
        self.commit(self.toolkit, "third platform")
        (self.project / "tools/dk/core/a.py").write_text("a = 2\n")
        self.commit(self.project, "ours")
        code, out = self.run_cmd("export")
        self.assertEqual(code, USAGE)
        self.assertIn("another project exported", out)
        self.assertIn("git switch -c sync-demo", out)
        self.assertTrue((self.toolkit / "tools/dk/platforms/third/t.py").is_file())       # nothing removed
        self.assertEqual((self.toolkit / "tools/dk/core/a.py").read_text(), "a = 1\n")     # nothing copied

    def test_the_recipe_merges_both_sides(self):
        self.run_cmd("base", set_=True)
        self.commit(self.project, "base")
        base = toolkitcmd.head(self.toolkit)
        (self.toolkit / "tools/dk/platforms/third").mkdir(parents=True)
        (self.toolkit / "tools/dk/platforms/third/t.py").write_text("third = 1\n")
        self.commit(self.toolkit, "third platform")
        (self.project / "tools/dk/core/a.py").write_text("a = 2\n")
        self.commit(self.project, "ours")
        git(self.toolkit, "switch", "-q", "-c", "sync-demo", base)
        self.assertEqual(self.run_cmd("export", force=True)[0], OK)
        self.commit(self.toolkit, "demo: toolkit changes")
        git(self.toolkit, "switch", "-q", "main")
        git(self.toolkit, "merge", "-q", "--no-edit", "sync-demo")
        self.assertTrue((self.toolkit / "tools/dk/platforms/third/t.py").is_file())
        self.assertEqual((self.toolkit / "tools/dk/core/a.py").read_text(), "a = 2\n")
        self.assertEqual(self.run_cmd("import", force=True)[0], OK)
        self.assertTrue((self.project / "tools/dk/platforms/third/t.py").is_file())
        self.commit(self.project, "import")
        (self.project / "tools/dk/core/a.py").write_text("a = 3\n")          # the base now allows a plain export
        self.commit(self.project, "more")
        self.assertEqual(self.run_cmd("export")[0], OK)

    def test_import_refuses_when_this_project_changed_since(self):
        self.run_cmd("base", set_=True)
        (self.project / "tools/dk/core/a.py").write_text("a = 2\n")
        self.commit(self.project, "ours")
        (self.toolkit / "playbook/README.md").write_text("# rules, newer\n")
        self.commit(self.toolkit, "theirs")
        code, out = self.run_cmd("import")
        self.assertEqual(code, USAGE)
        self.assertIn("this project's owned files changed since its last sync", out)
        self.assertEqual((self.project / "tools/dk/core/a.py").read_text(), "a = 2\n")

    def test_an_unchanged_project_imports_and_a_committed_export_still_matches(self):
        self.run_cmd("base", set_=True)
        self.commit(self.project, "base")
        (self.toolkit / "playbook/README.md").write_text("# rules, newer\n")
        self.commit(self.toolkit, "theirs")
        self.assertEqual(self.run_cmd("import")[0], OK)
        self.assertEqual((self.project / "playbook/README.md").read_text(), "# rules, newer\n")
        self.commit(self.project, "import")
        (self.project / "tools/dk/core/a.py").write_text("a = 2\n")
        self.commit(self.project, "ours")
        self.assertEqual(self.run_cmd("export")[0], OK)
        self.commit(self.toolkit, "ours exported")                        # HEAD moves, the content is ours
        (self.project / "tools/dk/core/a.py").write_text("a = 3\n")
        self.commit(self.project, "more")
        self.assertEqual(self.run_cmd("export")[0], OK)


if __name__ == "__main__":
    unittest.main()
