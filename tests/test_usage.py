"""The usage log (core/usage.py) and its report (core/usagecmd.py), on synthetic logs."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_project
from tools.dk.core import cli, usage, usagecmd
from tools.dk.platforms.gc_mwcc.usagehooks import USAGE_LANDING, USAGE_SEARCH


def row(time, cmd, exit=0, wt="w1", secs=1.0, **extra):
    r = {"time": f"2026-11-{time}:00-06:00", "cmd": cmd, "sub": "", "flags": extra.pop("flags", []), "args": [],
         "worktree": wt, "exit": exit, "secs": secs}
    r.update(extra)
    return r


# Two worktrees. w1: close on f1 then a save landing f1 (hit); try on f2 three times, never landed;
# regs --reach on f3, then a save refused by the floor; a second session (gap > 3 h) lands f2, which
# must not count for the first session's try. w2: close on f1 with no landing in w2 (miss), a crash.
SYNTHETIC = [
    row("10T09:00", "close", 1, function="f1", unit="u1", score_before=91.0, score_after=100.0, secs=40),
    row("10T09:05", "save", 0, unit="u1", committed=True, landed=["f0", "f1"], secs=120),
    row("10T09:10", "try", 1, function="f2", unit="u1", secs=2),
    row("10T09:11", "try", 1, function="f2", unit="u1", secs=3),
    row("10T09:12", "try", 2, function="f2", unit="u1", msg="error: no variants in a.cpp", secs=0.1),
    row("10T09:20", "regs", 1, function="f3", flags=["--reach"], secs=5),
    row("10T09:30", "save", 1, unit="u1", committed=False, refused_by="floor", msg="REJECTED: below the progress floor: 12 < 13"),
    row("10T14:00", "save", 0, unit="u1", committed=True, landed=["f2"], secs=100),
    row("10T09:00", "close", 1, wt="w2", function="f1", secs=60),
    row("10T09:01", "check", None, wt="w2", error="KeyError", secs=0.2),
    row("10T09:02", "inlines", 0, wt="w2", flags=["--scan"], functions=["f9", "f8"], secs=9),
    row("11T09:00", "save", 0, wt="w2", committed=True, landed=["f9"], secs=90),
]


class ArgsTest(unittest.TestCase):
    def test_shorten(self):
        root = Path("/repo/wt")
        got = usage.shorten_args(["/repo/wt/src/a.cpp", str(Path.home() / "x.txt"), "-m", "long message",
                                  "--note=some words", "a\nb", "fn", "--private", "/x/" + "d" * 90 + "/y/z.cpp"], root)
        self.assertEqual(got, ["src/a.cpp", "~/x.txt", "-m", "<text 12>", "--note=<text 10>", "<text 3>", "fn",
                               "--private", ".../y/z.cpp"])

    def test_main_checkout_of_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            main, wt = Path(tmp) / "repo", Path(tmp) / "repo/worktrees/a"
            (main / ".git/worktrees/a").mkdir(parents=True)
            wt.mkdir(parents=True)
            (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/a\n")
            self.assertEqual(usage.main_checkout(wt), main)
            self.assertEqual(usage.worktree_name(wt), "a")
            self.assertEqual(usage.worktree_name(main), "main")

    def test_log_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DK_USAGE_LOG", None)
                self.assertEqual(usage.log_path(cfg), Path(tmp).parent / "dk-usage.jsonl")
                os.environ["DK_USAGE_LOG"] = "off"
                self.assertIsNone(usage.log_path(cfg))
                os.environ["DK_USAGE_LOG"] = "/x/log.jsonl"
                self.assertEqual(usage.log_path(cfg), Path("/x/log.jsonl"))

    def test_refusal_kind(self):
        self.assertEqual(usage.refusal_kind("below the progress floor: x"), "floor")
        self.assertEqual(usage.refusal_kind("frozen function edited:\n  f"), "frozen")
        self.assertEqual(usage.refusal_kind("forbidden: pointer-offset"), "quality")
        self.assertEqual(usage.refusal_kind("hash mismatch"), "verify")


class InvokeTest(unittest.TestCase):
    def _run(self, main, argv, log):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"DK_USAGE_LOG": str(log)}), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                return usage.invoke(main, argv), out.getvalue()
            except BaseException as e:      # noqa: BLE001 - the test inspects what invoke lets through
                return e, out.getvalue()

    def test_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "sub/log.jsonl"

            def ok(argv):
                usage.note(unit="u", score_after=100.0)
                print("all fine")
                return 0

            def mismatch(argv):
                print("REJECTED: unit is not an exact match")
                print("a later ordinary line")
                return 1

            def crash(argv):
                raise KeyError("boom")

            self.assertEqual(self._run(ok, ["check", "u", "--private"], log), (0, "all fine\n"))
            self.assertEqual(self._run(mismatch, ["accept", "u"], log)[0], 1)
            self.assertIsInstance(self._run(crash, ["close", "u", "f"], log)[0], KeyError)
            got = self._run(lambda a: (_ for _ in ()).throw(SystemExit(2)), ["check", "--bogus"], log)[0]
            self.assertIsInstance(got, SystemExit)
            rows = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual([r["exit"] for r in rows], [0, 1, None, 2])
            self.assertEqual(rows[0]["cmd"], "check")
            self.assertEqual(rows[0]["flags"], ["--private"])
            self.assertEqual(rows[0]["unit"], "u")
            self.assertNotIn("msg", rows[0])
            self.assertEqual(rows[1]["msg"], "REJECTED: unit is not an exact match")
            self.assertEqual(rows[2]["error"], "KeyError")
            self.assertGreaterEqual(rows[0]["secs"], 0)

    def test_logging_errors_are_swallowed_and_off_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "file"
            blocker.write_text("")
            self.assertEqual(self._run(lambda a: 0, ["status"], blocker / "log.jsonl")[0], 0)   # parent is a file
            self.assertEqual(self._run(lambda a: 3, ["build"], "off")[0], 3)

    def test_cli_records_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_project(Path(tmp))
            log = Path(tmp) / "log.jsonl"
            log.write_text("\n".join(json.dumps(r) for r in SYNTHETIC) + "\n")
            other = Path(tmp) / "own.jsonl"
            with mock.patch("tools.dk.core.cli.load", lambda: __import__("tools.dk.core.config", fromlist=["load"]).load(Path(tmp))):
                code, out = self._run(cli.main, ["usage", "report", "--log", str(log)], other)
            self.assertEqual(code, 0)
            self.assertIn("Search tools", out)
            mine = json.loads(other.read_text())
            self.assertEqual((mine["cmd"], mine["sub"]), ("usage", "report"))


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.a = usagecmd.analyze(sorted(SYNTHETIC, key=usagecmd._when), USAGE_SEARCH, USAGE_LANDING)

    def test_commands(self):
        by = {c["command"]: c for c in self.a["commands"]}
        self.assertEqual(by["save"]["runs"], 4)
        self.assertEqual(by["save"]["failures"], 1)
        self.assertEqual(by["try"]["usage"], 1)
        self.assertEqual(by["try"]["top_messages"][0][0], "error: no variants in a.cpp")
        self.assertEqual(by["check"]["crash"], 1)
        self.assertEqual(by["check"]["crash_types"], {"KeyError": 1})
        self.assertEqual(by["close"]["median_s"], 40.0)
        self.assertEqual(by["close"]["p95_s"], 60.0)

    def test_hit_rates(self):
        tools = {t["tool"]: t for t in self.a["search_tools"]}
        self.assertEqual((tools["close"]["functions"], tools["close"]["landed"]), (2, 1))   # w1 hit, w2 miss
        self.assertEqual((tools["try"]["runs"], tools["try"]["landed"]), (3, 0))           # landed in a later session
        self.assertEqual(tools["regs --reach"]["landed"], 0)
        self.assertEqual(tools["regs --why-temp"]["hit_rate"], None)
        self.assertEqual((tools["inlines --scan"]["functions"], tools["inlines --scan"]["landed"]), (2, 0))  # next day

    def test_unlanded_and_guards(self):
        top = self.a["unlanded"][0]
        self.assertEqual((top["function"], top["runs"]), ("f2", 3))
        self.assertNotIn("f1", [u["function"] for u in self.a["unlanded"] if u["sessions"] > 1])
        self.assertEqual(self.a["guards"], [{"guard": "floor", "command": "save", "refusals": 1}])
        self.assertEqual(self.a["sessions"], 4)

    def test_render_and_since(self):
        text = "\n".join(usagecmd.render(self.a))
        self.assertTrue(text.startswith("12 dk runs from 2026-11-10 to 2026-11-11"))
        self.assertIn("close led to a landing for 1 of 2 function(s) (50%)", text)
        self.assertIn("Most work without a landing: f2, 3 runs", text)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            log.write_text("\n".join(json.dumps(r) for r in SYNTHETIC) + "\nnot json\n")
            self.assertEqual(len(usagecmd.read_log(log)), 12)
            self.assertEqual(len(usagecmd.read_log(log, "2026-11-11")), 1)
        self.assertEqual(usagecmd.render(usagecmd.analyze([], {}, ())), ["The usage log has no runs in this period."])


if __name__ == "__main__":
    unittest.main()
