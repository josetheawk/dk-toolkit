"""Per-agent worktrees and landing on the shared branch, with throwaway git repos."""

import dataclasses
import tempfile
import unittest
from pathlib import Path

from tests.helpers import git, make_git_project
from tools.dk.core import isolated
from tools.dk.core.config import load


class SymlinkedOriginalTest(unittest.TestCase):
    def test_a_verify_never_touches_the_file_a_symlinked_original_points_at(self):
        with tempfile.TemporaryDirectory() as d:
            elsewhere = Path(d) / "elsewhere" / "main.dol"
            elsewhere.parent.mkdir()
            elsewhere.write_bytes(b"ORIGINAL")
            make_git_project(Path(d) / "repo")
            link = Path(d) / "repo" / "orig" / "main.dol"
            link.parent.mkdir()
            link.symlink_to(elsewhere)
            cfg = load(Path(d) / "repo")
            self.assertEqual(cfg.rel(cfg.original), "orig/main.dol")    # in-repo path, not the link target
            isolated.pin_head(cfg)
            ok, detail = isolated.verify_overlay(cfg, {})
            self.assertTrue(ok, detail)
            self.assertFalse(elsewhere.is_symlink())
            self.assertEqual(elsewhere.read_bytes(), b"ORIGINAL")

    def test_an_original_outside_the_project_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_git_project(Path(d) / "repo")
            outside = Path(d) / "main.dol"
            outside.write_bytes(b"ORIGINAL")
            cfg = dataclasses.replace(cfg, original=outside)
            isolated.pin_head(cfg)
            ok, detail = isolated.verify_overlay(cfg, {})
            self.assertFalse(ok)
            self.assertIn("not inside the project root", detail)
            self.assertEqual(outside.read_bytes(), b"ORIGINAL")

    def test_verify_builds_and_fails(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_git_project(Path(d) / "repo")
            isolated.pin_head(cfg)
            self.assertTrue(isolated.verify_overlay(cfg, {"src/a.txt": b"fine\n"})[0])
            self.assertFalse(isolated.verify_overlay(cfg, {"src/a.txt": b"BROKEN\n"})[0])


def _run(cfg, action, **kw):
    import argparse
    import contextlib
    import io
    from tools.dk.core import worktreecmd
    defaults = {"rev": None, "no_build": True, "force": False, "keep_mine": []}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = worktreecmd.run(cfg, argparse.Namespace(action=action, **{**defaults, **kw}))
    return code, out.getvalue()


class WorktreeCommandTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.cfg = make_git_project(self.repo)
        (self.repo / "orig").mkdir()
        (self.repo / "orig" / "main.dol").write_bytes(b"ORIGINAL")
        (self.repo / "build" / "tools").mkdir(parents=True)
        (self.repo / "build" / "tools" / "dtk").write_bytes(b"tool")
        (self.repo / "build" / "compilers").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_makes_a_provisioned_worktree_on_its_own_branch(self):
        from tools.dk.core import landing, worktreecmd
        code, out = _run(self.cfg, "add", name="a1")
        self.assertEqual(code, 0, out)
        tree = self.repo / "worktrees" / "a1"
        self.assertEqual(git(tree, "symbolic-ref", "HEAD"), "refs/heads/dk/a1")
        self.assertEqual(git(tree, "rev-parse", "HEAD"), git(self.repo, "rev-parse", "shared"))
        self.assertEqual(landing.meta(tree), {"name": "a1", "shared": "refs/heads/shared"})
        self.assertIsNone(landing.meta(self.repo))
        dol = tree / "orig" / "main.dol"
        self.assertFalse(dol.is_symlink())                 # a copy: a build tree can't reach main's file
        self.assertEqual(dol.read_bytes(), b"ORIGINAL")
        self.assertTrue((tree / "build" / "tools").is_symlink())
        self.assertEqual(worktreecmd.changes(tree), [])    # everything it added is ignored
        self.assertIn("--dtk", worktreecmd.setup_steps(self.cfg, tree)[0])

    def test_tools_fall_back_to_the_main_checkout(self):
        _run(self.cfg, "add", name="a1")
        tree = self.repo / "worktrees" / "a1"
        (tree / "build" / "tools").unlink()
        (tree / "build" / "compilers").unlink()
        args = isolated.tool_args(tree)
        self.assertEqual(args[args.index("--dtk") + 1], str((self.repo / "build" / "tools" / "dtk").resolve()))
        self.assertEqual(args[args.index("--compilers") + 1], str((self.repo / "build" / "compilers").resolve()))

    def test_bad_or_taken_names_are_refused(self):
        self.assertEqual(_run(self.cfg, "add", name="../x")[0], 2)
        self.assertEqual(_run(self.cfg, "add", name="a1")[0], 0)
        self.assertEqual(_run(self.cfg, "add", name="a1")[0], 2)

    def test_list_shows_how_far_behind_and_what_changed(self):
        _run(self.cfg, "add", name="b2")
        _run(self.cfg, "add", name="a1")
        tree = self.repo / "worktrees" / "a1"
        (tree / "src" / "a.txt").write_text("mine\n")
        (self.repo / "src" / "b.txt").write_text("b\n")
        git(self.repo, "add", "src/b.txt")
        git(self.repo, "commit", "-q", "-m", "moved on")
        code, out = _run(self.cfg, "list")
        rows = [l.split()[:5] for l in out.splitlines()[2:4]]
        self.assertEqual([r[0] for r in rows], ["a1", "b2"])
        self.assertEqual([(r[1], r[3], r[4]) for r in rows], [("dk/a1", "1", "1"), ("dk/b2", "1", "0")])

    def test_remove_refuses_uncommitted_work_and_keeps_unlanded_branches(self):
        _run(self.cfg, "add", name="a1")
        _run(self.cfg, "add", name="a2")
        tree = self.repo / "worktrees" / "a1"
        (tree / "src" / "a.txt").write_text("mine\n")
        code, out = _run(self.cfg, "remove", name="a1")
        self.assertEqual(code, 1, out)
        self.assertTrue(tree.is_dir())
        code, out = _run(self.cfg, "remove", name="a1", force=True)
        self.assertEqual(code, 0, out)
        self.assertFalse(tree.exists())
        self.assertIn("deleted branch dk/a1", out)
        self.assertEqual(self.repo.joinpath("orig", "main.dol").read_bytes(), b"ORIGINAL")
        self.assertTrue((self.repo / "build" / "tools" / "dtk").is_file())   # links removed, not their targets
        t2 = self.repo / "worktrees" / "a2"
        (t2 / "src" / "a.txt").write_text("unlanded\n")
        git(t2, "commit", "-q", "-am", "only on dk/a2")
        code, out = _run(self.cfg, "remove", name="a2")
        self.assertEqual(code, 0, out)
        self.assertIn("kept branch dk/a2 (1 commit(s) not on shared)", out)
        self.assertEqual(_run(self.cfg, "add", name="a2")[0], 0)           # resumes the kept branch
        self.assertEqual((t2 / "src" / "a.txt").read_text(), "unlanded\n")


LANDER = """
import sys
from pathlib import Path
from tools.dk.core import isolated
from tools.dk.core.config import load
cfg = load(Path(sys.argv[1]))
for attempt in range(isolated.UNLOCKED_TRIES + 1):
    isolated.pin_head(cfg, attempt)
    overlay = isolated.working_overlay(cfg, sys.argv[3:])
    ok, detail = isolated.verify_overlay(cfg, overlay)
    if not ok:
        break
    ok, detail = isolated.commit_overlay(cfg, overlay, sys.argv[2])
    if detail != isolated.MOVED:
        break
print(attempt, ok, detail)
sys.exit(0 if ok else 1)
"""
LINES = "".join(f"line {i}\n" for i in range(1, 10))


def land(tree, paths, message="land"):
    """One lander, in-process: what save / commit-tools do."""
    cfg = load(tree)
    for attempt in range(isolated.UNLOCKED_TRIES + 1):
        isolated.pin_head(cfg, attempt)
        overlay = isolated.working_overlay(cfg, paths)
        ok, detail = isolated.verify_overlay(cfg, overlay)
        if not ok:
            return ok, detail
        ok, detail = isolated.commit_overlay(cfg, overlay, message)
        if detail != isolated.MOVED:
            return ok, detail
    return ok, detail


class LandingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.cfg = make_git_project(self.repo, files={"src/m.txt": LINES})
        for name in ("a1", "a2"):
            self.assertEqual(_run(self.cfg, "add", name=name)[0], 0)
        self.a1, self.a2 = (self.repo / "worktrees" / n for n in ("a1", "a2"))
        self.start = git(self.repo, "rev-parse", "shared")

    def tearDown(self):
        from tools.dk.core import landing
        isolated._pinned, isolated._holding = None, False
        while landing._lock["depth"]:          # a failed test must not leak the lock into the next
            landing.release()
        self._tmp.cleanup()

    def tip(self, rel):
        return git(self.repo, "show", f"shared:{rel}") + "\n"

    def test_a_landing_goes_on_the_shared_branch_and_into_the_main_checkout(self):
        (self.a1 / "src" / "a.txt").write_text("from a1\n")
        ok, detail = land(self.a1, ["src/a.txt"])
        self.assertTrue(ok, detail)
        self.assertIn("on shared", detail)
        tip = git(self.repo, "rev-parse", "shared")
        self.assertEqual(git(self.repo, "rev-parse", "shared^"), self.start)
        self.assertEqual(self.tip("src/a.txt"), "from a1\n")
        self.assertEqual((self.repo / "src" / "a.txt").read_text(), "from a1\n")   # main checkout carried along
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        self.assertEqual(git(self.a1, "rev-parse", "HEAD"), tip)                    # dk/a1 moved forward
        self.assertEqual(git(self.a1, "status", "--porcelain"), "")
        self.assertEqual(git(self.a2, "rev-parse", "HEAD"), self.start)             # others sync when they land

    def test_a_failing_build_lands_nothing(self):
        (self.a1 / "src" / "a.txt").write_text("BROKEN\n")
        ok, _ = land(self.a1, ["src/a.txt"])
        self.assertFalse(ok)
        self.assertEqual(git(self.repo, "rev-parse", "shared"), self.start)

    def test_edits_in_the_main_checkout_merge_or_refuse_the_landing(self):
        (self.repo / "src" / "m.txt").write_text(LINES.replace("line 9", "main 9"))   # main's own edit
        (self.a1 / "src" / "m.txt").write_text(LINES.replace("line 1", "a1 1"))
        ok, detail = land(self.a1, ["src/m.txt"])
        self.assertTrue(ok, detail)
        self.assertEqual(self.tip("src/m.txt"), LINES.replace("line 1", "a1 1"))
        self.assertEqual((self.repo / "src" / "m.txt").read_text(),
                         LINES.replace("line 1", "a1 1").replace("line 9", "main 9"))
        self.assertEqual(git(self.repo, "status", "--porcelain"), "M src/m.txt")   # main's edit, still uncommitted
        landed = git(self.repo, "rev-parse", "shared")
        (self.a2 / "src" / "m.txt").write_text(LINES.replace("line 1", "a1 1").replace("line 9", "a2 9"))
        ok, detail = land(self.a2, ["src/m.txt"])
        self.assertFalse(ok)
        self.assertIn("do not merge", detail)
        self.assertEqual(git(self.repo, "rev-parse", "shared"), landed)

    def test_a_path_staged_in_the_main_checkout_is_refused(self):
        (self.repo / "src" / "a.txt").write_text("staged in main\n")
        git(self.repo, "add", "src/a.txt")
        (self.a1 / "src" / "a.txt").write_text("from a1\n")
        ok, detail = land(self.a1, ["src/a.txt"])
        self.assertFalse(ok)
        self.assertIn("staged there differently", detail)
        self.assertEqual(git(self.repo, "rev-parse", "shared"), self.start)

    def test_two_landers_at_once_both_land(self):
        import os
        import subprocess
        import sys
        from tools.dk.core.config import REPO
        barrier = Path(self._tmp.name) / "barrier"
        barrier.mkdir()
        (self.a1 / "src" / "m.txt").write_text(LINES.replace("line 1", "a1 1"))
        (self.a1 / "src" / "one.txt").write_text("a1\n")
        (self.a2 / "src" / "m.txt").write_text(LINES.replace("line 9", "a2 9"))
        (self.a2 / "src" / "two.txt").write_text("a2\n")
        env = {**os.environ, "DK_TEST_BARRIER": str(barrier), "PYTHONPATH": str(REPO)}
        procs = [subprocess.Popen([sys.executable, "-c", LANDER, str(tree), f"from {name}", "src/m.txt", f"src/{new}.txt"],
                                  cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for tree, name, new in ((self.a1, "a1", "one"), (self.a2, "a2", "two"))]
        outs = [p.communicate(timeout=120)[0] for p in procs]
        self.assertEqual([p.returncode for p in procs], [0, 0], outs)
        attempts = sorted(int(o.split()[0]) for o in outs)
        self.assertEqual(attempts, [0, 1], outs)              # they raced: one re-synced and retried
        log = git(self.repo, "log", "--format=%s", f"{self.start}..shared").splitlines()
        self.assertEqual(sorted(log), ["from a1", "from a2"])
        both = LINES.replace("line 1", "a1 1").replace("line 9", "a2 9")
        self.assertEqual(self.tip("src/m.txt"), both)
        for rel, text in (("src/m.txt", both), ("src/one.txt", "a1\n"), ("src/two.txt", "a2\n")):
            self.assertEqual((self.repo / rel).read_text(), text)
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        winner = self.a1 if log[0] == "from a1" else self.a2      # log is newest first: the retrier
        self.assertEqual(git(winner, "rev-parse", "HEAD"), git(self.repo, "rev-parse", "shared"))
        self.assertEqual(git(winner, "status", "--porcelain"), "")

    def test_the_last_attempt_holds_the_land_lock_until_its_commit(self):
        import fcntl
        from tools.dk.core import landing

        def free():
            with open(landing.common_dir(self.repo) / landing.LOCK, "w") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return False
                fcntl.flock(f, fcntl.LOCK_UN)
                return True

        cfg = load(self.a1)
        isolated.pin_head(cfg, 0)
        self.assertTrue(free())
        isolated.pin_head(cfg, isolated.UNLOCKED_TRIES)
        self.assertFalse(free())                     # (flock is per open file: a second open contends)
        (self.a1 / "src" / "a.txt").write_text("x\n")
        ok, detail = isolated.commit_overlay(cfg, isolated.working_overlay(cfg, ["src/a.txt"]), "x")
        self.assertTrue(ok, detail)
        self.assertTrue(free())

    def test_the_main_checkout_still_commits_on_its_own_branch(self):
        (self.repo / "src" / "a.txt").write_text("in main\n")
        ok, detail = land(self.repo, ["src/a.txt"])
        self.assertTrue(ok, detail)
        self.assertEqual(self.tip("src/a.txt"), "in main\n")
        self.assertEqual((self.a1 / "src" / "a.txt").read_text(), "a\n")      # worktrees pick it up on sync


class SyncTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.cfg = make_git_project(self.repo, files={"src/m.txt": LINES, "src/b.txt": "b\n"})
        _run(self.cfg, "add", name="a1")
        self.a1 = self.repo / "worktrees" / "a1"
        # the shared branch moves on: a.txt and m.txt line 9 change, c.txt is new, b.txt is deleted
        (self.repo / "src" / "a.txt").write_text("theirs\n")
        (self.repo / "src" / "m.txt").write_text(LINES.replace("line 9", "theirs 9"))
        (self.repo / "src" / "c.txt").write_text("new\n")
        git(self.repo, "rm", "-q", "src/b.txt")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "theirs")

    def tearDown(self):
        self._tmp.cleanup()

    def sync(self, keep=()):
        from tools.dk.core import landing
        return landing.sync(self.a1, tuple(keep))

    def test_untouched_paths_take_theirs_and_edits_merge(self):
        (self.a1 / "src" / "m.txt").write_text(LINES.replace("line 1", "mine 1"))
        ok, detail = self.sync()
        self.assertTrue(ok, detail)
        self.assertEqual(git(self.a1, "rev-parse", "HEAD"), git(self.repo, "rev-parse", "shared"))
        self.assertEqual((self.a1 / "src" / "a.txt").read_text(), "theirs\n")
        self.assertEqual((self.a1 / "src" / "c.txt").read_text(), "new\n")
        self.assertFalse((self.a1 / "src" / "b.txt").exists())
        self.assertEqual((self.a1 / "src" / "m.txt").read_text(), LINES.replace("line 1", "mine 1").replace("line 9", "theirs 9"))
        self.assertEqual(git(self.a1, "status", "--porcelain"), "M src/m.txt")    # only my own change is left
        self.assertTrue(self.sync()[0])                                            # idempotent

    def test_a_conflict_refuses_the_whole_sync(self):
        head = git(self.a1, "rev-parse", "HEAD")
        (self.a1 / "src" / "m.txt").write_text(LINES.replace("line 9", "mine 9"))
        (self.a1 / "src" / "b.txt").write_text("edited here, deleted there\n")
        ok, detail = self.sync()
        self.assertFalse(ok)
        self.assertIn("src/b.txt, src/m.txt", detail)
        self.assertEqual(git(self.a1, "rev-parse", "HEAD"), head)
        self.assertEqual((self.a1 / "src" / "a.txt").read_text(), "a\n")          # nothing was written
        with self.assertRaises(isolated.LandError):
            isolated.pin_head(load(self.a1))
        ok, detail = self.sync(keep=["src/m.txt", "src/b.txt"])
        self.assertTrue(ok, detail)
        self.assertEqual((self.a1 / "src" / "m.txt").read_text(), LINES.replace("line 9", "mine 9"))
        self.assertEqual((self.a1 / "src" / "a.txt").read_text(), "theirs\n")

    def test_the_sync_command(self):
        (self.a1 / "src" / "m.txt").write_text(LINES.replace("line 9", "mine 9"))
        code, out = _run(load(self.a1), "sync")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED", out)
        self.assertIn("src/m.txt", out)
        code, out = _run(load(self.a1), "sync", keep_mine=["./src/m.txt"])
        self.assertEqual(code, 0, out)
        self.assertIn("1 kept as yours", out)
        self.assertEqual(_run(self.cfg, "sync")[0], 2)          # the main checkout is not a dk worktree

    def test_a_sync_that_brings_config_changes_refreshes_the_build_inputs(self):
        from unittest import mock
        from tools.dk import platforms
        (self.repo / "config").mkdir(exist_ok=True)
        (self.repo / "config" / "symbols.txt").write_text("fn_a = .text:0x80000000;\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "symbols")
        calls = []
        fake = type("P", (), {"after_config_sync": staticmethod(lambda cfg: calls.append(cfg.root) or True)})
        with mock.patch.object(platforms, "load", return_value=fake):
            code, out = _run(load(self.a1), "sync")
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, [self.a1])
        self.assertIn("config changed (config/symbols.txt): the build's split outputs were regenerated", out)
        (self.repo / "config" / "symbols.txt").write_text("fn_b = .text:0x80000000;\n")
        git(self.repo, "commit", "-q", "-am", "symbols again")
        with mock.patch.object(platforms, "load", return_value=type("P", (), {})):
            code, out = _run(load(self.a1), "sync")
        self.assertIn("build outputs made from it are stale", out)
        self.assertIn("./dk build", out)

    def test_a_sync_that_brings_toolkit_changes_stops_the_running_command(self):
        # the running process loaded the old toolkit: it must not compose files in a newer format
        (self.repo / "tools" / "dk").mkdir(parents=True)
        (self.repo / "tools" / "dk" / "units.py").write_text("# newer\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "toolkit")
        with self.assertRaisesRegex(isolated.LandError, "run the same command again"):
            isolated.pin_head(load(self.a1))
        self.assertEqual(git(self.a1, "rev-parse", "HEAD"), git(self.repo, "rev-parse", "shared"))   # synced anyway
        self.assertEqual(isolated.pin_head(load(self.a1)), git(self.repo, "rev-parse", "shared"))    # the rerun goes on

    def test_raw_commits_in_the_worktree_are_refused(self):
        (self.a1 / "src" / "a.txt").write_text("raw\n")
        git(self.a1, "commit", "-q", "-am", "raw")
        ok, detail = self.sync()
        self.assertFalse(ok)
        self.assertIn("not on shared", detail)


class NestedCheckoutDocsTest(unittest.TestCase):
    def test_doc_lint_skips_other_checkouts_inside_the_repo(self):
        from tools.dk.core import doclint
        with tempfile.TemporaryDirectory() as d:
            cfg = make_git_project(Path(d) / "repo", files={"README.md": "ok\n"})
            nested = cfg.root / "worktrees" / "x"
            nested.mkdir(parents=True)
            (nested / ".git").write_text("gitdir: elsewhere\n")
            (nested / "README.md").write_text("[gone](missing.md)\n")
            self.assertEqual([cfg.rel(p) for p in doclint.markdown_files(cfg)], ["README.md"])


if __name__ == "__main__":
    unittest.main()
