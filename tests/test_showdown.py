import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.helpers import make_project
from tools.dk.platforms.gc_mwcc import llmtry, showdown

SET = {"easy": [{"unit": "main/a", "source": "src/a.cpp", "function": "fn_E1", "address": 1, "size": 0x40, "fuzzy": 94.4},
                {"unit": "main/a", "source": "src/a.cpp", "function": "fn_E2", "address": 2, "size": 0x60, "fuzzy": 0.0}],
       "medium": [{"unit": "main/b", "source": "src/b.cpp", "function": "fn_M1", "address": 3, "size": 0x200, "fuzzy": 98.5}],
       "hard": [{"unit": "main/c", "source": "src/c.cpp", "function": "fn_H1", "address": 4, "size": 0x800, "fuzzy": 99.0}]}

ALPHA = """tier easy start 10:13
10:13 easy fn_E1 94.4 -> 100 exact  (switch with extra empty cases)
10:16 easy fn_E2(__sinit_x_cpp) 0 -> 100 code, link-exact only with aligned(8) => device, counts MISS
tier easy done 10:23: 1 exact
10:23 MEDIUM tier start (60 min budget)
10:28 medium fn_M1 98.5 -> 98.5 miss (callee-saved registers swapped; close no gain)
10:31 medium misses: fn_M1 98.6 (all register-rank swaps)
10:35 hard fn_H1 99.31 -> 99.31 miss  (12-byte struct copy picks other temps)
"""
BETA = """10:14 easy fn_E1 94.4 -> 94.4 miss seed1
10:20 easy fn_E1 94.4 -> 100.0 exact seed2
10:21 easy fn_E2 0.0 -> 0.0 miss seed1
10:22 easy fn_NOT_IN_SET 1.0 -> 2.0 miss seed1
"""


class ParseTest(unittest.TestCase):
    def test_log_formats(self):
        rows, starts = showdown.parse_log(ALPHA)
        self.assertEqual(starts, {"easy": 10 * 60 + 13, "medium": 10 * 60 + 23})
        got = [(r["function"], r["verdict"], r["end"], r["caveat"]) for r in rows]
        self.assertEqual(got, [("fn_E1", "exact", 100.0, False), ("fn_E2", "miss", 100.0, False),
                               ("fn_M1", "miss", 98.5, False), ("fn_H1", "miss", 99.31, False)])
        self.assertEqual(rows[0]["reason"], "switch with extra empty cases")
        self.assertEqual(rows[1]["tier"], "easy")                       # `fn(__sinit..)` and `counts MISS`

    def test_exact_with_a_caveat_and_seed_lines(self):
        rows, _ = showdown.parse_log("10:24 easy fn_A 0 -> 100 exact*  (the TU's __sinit; check cannot pair it)\n"
                                     "10:25 easy fn_B 90.3 -> 90.3 miss seed2\n10:26 easy fn_C 1 -> 2 progress\n")
        self.assertEqual([(r["function"], r["verdict"], r["caveat"]) for r in rows], [("fn_A", "exact", True), ("fn_B", "miss", False)])
        self.assertEqual(rows[1]["reason"], "seed2")

    def test_blocker_classes_from_the_reason_then_the_facts(self):
        fn = {"function": "fn_X", "size": 0x40, "fuzzy": 95.0}
        b = showdown.blocker
        self.assertEqual(b(fn, "register numbering in the shared table idiom; unit on GC/1.2.5"), "near-register")
        self.assertEqual(b(fn, "string-pool offsets depend on 10 unwritten functions of the unit"), "unit-data")
        self.assertEqual(b(fn, "prologue scheduling of four conversions"), "near-scheduling")
        self.assertEqual(b(fn, "compiler-search best GC/1.3.2 99.58"), "compiler-version")
        self.assertEqual(b(fn, "it is the TU static initializer"), "sinit")
        self.assertEqual(b({**fn, "function": "__sinit_q_cpp"}, ""), "sinit")
        self.assertEqual(b({**fn, "size": 0x800}, "nothing said"), "giant")
        self.assertEqual(b({**fn, "fuzzy": 0.0}, ""), "fresh")
        self.assertEqual(b({**fn, "fuzzy": 40.0}, ""), "far")
        self.assertEqual(b(fn, ""), "unclassified")

    def test_notes_cut_leaves_code_and_inner_comments(self):
        self.assertEqual(showdown.notes_cut("// a note\n// more\n\nint f() {\n    // inside\n    return 1;\n}\n"),
                         "int f() {\n    // inside\n    return 1;\n}\n")


class FakeScorer:
    """Stands in for llmtry.Scorer: `evaluate` returns what the compile + judge would."""

    def __init__(self, src, evals):
        self.src, self.evals, self.unit, self.work, self.forbidden = src, evals, {"name": "main/a"}, Path("/nonexistent"), []

    def evaluate(self, chunk, slot):
        ev = dict(self.evals[chunk.split()[0]])
        if ev.get("compiled", True):
            ev.setdefault("text", self.src.replace("return a;", chunk))
        return ev


class JudgeChunkTest(unittest.TestCase):
    SRC = "int g;\nint f(int a) {\n    return a;\n}\n"

    def verdict(self, ev, chunk="x return a + 1;"):
        evals = {"x": ev}
        cfg = SimpleNamespace(root=Path("/nonexistent"))
        fn = {"function": "f", "unit": "main/a"}
        return showdown.judge_chunk(cfg, fn, chunk, {"f": FakeScorer(self.SRC, evals)})

    def test_verdicts(self):
        self.assertEqual(self.verdict({"compiled": False, "error": "syntax error"}),
                         {"verdict": "miss", "percent": None, "flags": ["syntax error"]})
        self.assertEqual(self.verdict({"compiled": True, "percent": 99.5})["verdict"], "miss")
        self.assertEqual(self.verdict({"compiled": True, "percent": 100.0})["verdict"], "certain")
        self.assertEqual(self.verdict({"compiled": True, "percent": 100.0, "judge": ["worse: h"]})["verdict"], "judgment")
        dev = self.verdict({"compiled": True, "percent": 100.0, "text": self.SRC.replace("return a;", "volatile int t = a; return t;")})
        self.assertEqual(dev["percent"], 100.0)
        self.assertIn(dev["verdict"], ("device", "judgment"))               # volatile alone asks for judgment
        pad = self.verdict({"compiled": True, "percent": 100.0, "text": self.SRC.replace("return a;", "int pad[2]; return a;")})
        self.assertEqual(pad["verdict"], "device")
        self.assertTrue(any("padding" in f for f in pad["flags"]), pad["flags"])


class RaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.ddir = Path(self.tmp.name)
        (d / "set.json").write_text(json.dumps(SET))
        (d / "alpha.log").write_text(ALPHA)
        (d / "alpha").mkdir()
        (d / "alpha" / "fn_E1.cpp").write_text("// note\nCLEAN\n")
        (d / "alpha" / "fn_E2.cpp").write_text("DEVICE\n")
        (d / "beta.log").write_text(BETA)                  # logs only: no .cpp to judge
        (d / "gamma").mkdir()                             # files only: no log
        (d / "gamma" / "fn_M1.cpp").write_text("JUDGE\n")
        (d / "alpha.err").write_text("ignored")

    @staticmethod
    def judge(fn, chunk):
        word = chunk.strip().splitlines()[-1]
        return {"verdict": {"CLEAN": "certain", "JUDGE": "judgment", "DEVICE": "device"}.get(word, "miss"),
                "percent": 100.0 if word in ("CLEAN", "JUDGE", "DEVICE") else 90.0,
                "flags": [] if word == "CLEAN" else [f"flag {word}"]}

    def test_tallies_per_contestant_and_tier(self):
        board = showdown.score_race(self.ddir, self.judge)
        self.assertEqual(sorted(board["contestants"]), ["alpha", "beta", "gamma"])
        a = board["contestants"]["alpha"]
        self.assertEqual({k: a["cells"]["easy"][k] for k in showdown.KINDS},
                         {"certain": 1, "judgment": 0, "device": 1, "unjudged": 0, "miss": 0})
        self.assertEqual(a["cells"]["medium"]["miss"], 1)
        self.assertEqual(a["cells"]["hard"]["miss"], 1)
        self.assertEqual((a["total"]["functions"], a["total"]["certain"], a["total"]["miss"]), (4, 1, 2))
        self.assertEqual(a["cells"]["easy"]["first_exact_min"], 0)          # explicit start 10:13, exact line 10:13
        self.assertEqual(a["cells"]["easy"]["attempts"], 2)
        b = board["contestants"]["beta"]
        self.assertEqual({k: b["cells"]["easy"][k] for k in showdown.KINDS},
                         {"certain": 0, "judgment": 0, "device": 0, "unjudged": 1, "miss": 1})
        self.assertEqual(b["cells"]["easy"]["first_exact_min"], 6)          # no start line: first logged line 10:14
        self.assertEqual(b["cells"]["easy"]["attempts"], 3)                  # the function outside the set is not counted
        g = board["contestants"]["gamma"]
        self.assertEqual(g["cells"]["medium"]["judgment"], 1)
        self.assertEqual(g["total"]["attempts"], 0)
        self.assertEqual(sum(c["total"]["functions"] for c in board["contestants"].values()), 12)

    def test_misses_carry_their_blocker_and_logged_claims_are_flagged(self):
        board = showdown.score_race(self.ddir, lambda fn, chunk: {"verdict": "miss", "percent": 90.0, "flags": ["no"]})
        rows = {r["function"]: r for r in board["contestants"]["alpha"]["rows"]}
        self.assertEqual(rows["fn_M1"]["blocker"], "near-register")
        self.assertEqual(rows["fn_H1"]["blocker"], "giant")           # its reason names no class; 0x800 bytes
        self.assertIn("log says exact", rows["fn_E1"]["note"])
        self.assertEqual(rows["fn_E1"]["log_best"], 100.0)
        gamma = {r["function"]: r for r in board["contestants"]["gamma"]["rows"]}
        self.assertEqual(gamma["fn_E1"]["blocker"], "unclassified")   # nothing logged, and 94.4% says no class
        self.assertEqual(gamma["fn_E2"]["blocker"], "fresh")          # nothing logged: from the facts (0%)

    def test_render_and_files(self):
        board = showdown.score_race(self.ddir, self.judge)
        md = showdown.render(board)
        self.assertIn("| **alpha** | **all** | 4 | **1** | 0 | 1 | 0 | 2 |", md)
        self.assertIn("| easy | fn_E1 | 0x40 | 94.4 | EXACT |", md)
        self.assertIn("exact (unjudged)", md)
        self.assertIn("## Misses by blocker class", md)
        self.assertIn("alpha fn_E2 (easy): device: flag DEVICE", md)
        self.assertLess(md.index("| **alpha**"), md.index("| **beta**"))        # most certain exact first

    def test_command_writes_scoreboard_and_json(self):
        (self.ddir / "proj").mkdir()
        cfg = make_project(self.ddir / "proj")
        args = argparse.Namespace(showdown=str(self.ddir), logs=[], paired=False, json="")
        with mock.patch.object(showdown, "judge_chunk", lambda cfg, fn, chunk, scorers: self.judge(fn, chunk)), \
                mock.patch("builtins.print"):
            self.assertEqual(llmtry.bench_report(cfg, args), 0)
        doc = json.loads((self.ddir / "scoreboard.json").read_text())
        self.assertEqual(doc["contestants"]["alpha"]["total"]["certain"], 1)
        self.assertEqual(doc["set"], {"easy": 2, "medium": 1, "hard": 1})
        self.assertTrue((self.ddir / "scoreboard.md").read_text().startswith("# Showdown scoreboard"))
        with mock.patch("builtins.print"):
            missing = argparse.Namespace(showdown=str(self.ddir / "nowhere"))
            self.assertEqual(showdown.run(cfg, missing), 2)


if __name__ == "__main__":
    unittest.main()
