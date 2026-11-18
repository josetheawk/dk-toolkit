import json
import tempfile
import unittest
from pathlib import Path

from tools.dk.core import stuck
from tools.dk.platforms.gc_mwcc import gate, lorafix, region


class LoraFix(unittest.TestCase):
    def test_hints(self):
        for h in ("rule:cast-int", "xform:for-init-list@42", "region:stmt:L1-82"):
            self.assertTrue(lorafix.valid_hint(h), h)
        for h in ("free text", "rule:", "xform:a@x", "region:a b", "rule:A;rm"):
            self.assertFalse(lorafix.valid_hint(h), h)

    def test_tool_block_structured(self):
        t = lorafix.tool_block(97.5, [1, 2], ("rule:x",))
        self.assertEqual(t, "TOOLS: pct=97.50 diff_rows=2 hint=rule:x")

    def test_summary_reports_the_best_attempts_diff_rows(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "run.jsonl"
            rows = [{"kind": "run"},
                    {"kind": "attempt", "round": 0, "compiled": True, "percent": 90.0, "rows": ["a", "b", "c"]},
                    {"kind": "attempt", "round": 0, "compiled": True, "percent": 95.0, "rows": ["a"], "variant": "v"},
                    {"kind": "attempt", "round": 1, "compiled": False}]
            log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            s = lorafix.summarize(log)
        self.assertEqual((s["best_pct"], s["candidate"], s["diff_rows"]), (95.0, "v", 1))

    def test_stuck_rule_and_replay(self):
        flat = [(0, 90.0, 4), (1, 90.0, 4), (2, 90.0, 4), (3, 90.0, 4)]
        self.assertEqual(stuck.stuck_at(flat), 2)
        self.assertIsNone(stuck.stuck_at([(0, 90.0, 4), (1, 91.0, 4), (2, 92.0, 4)]))
        self.assertIsNone(stuck.stuck_at([(0, 100.0, 8), (1, 100.0, 8), (2, 100.0, 8)]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            rows = [{"kind": "attempt", "function": "f", "round": r, "k": k, "compiled": True, "percent": 90.0}
                    for r in range(4) for k in range(4)]
            p.write_text("\n".join(json.dumps(x) for x in rows))
            r = stuck.replay([p])
            self.assertEqual((r["flagged"], r["precision"], r["saved_attempts"]), (1, 1.0, 4))

    def test_region_outcomes_and_splice(self):
        self.assertEqual(region.outcome(90, None, False, False), "malformed")
        self.assertEqual(region.outcome(90, None, False, True), "unappliable")
        self.assertEqual(region.outcome(90, 95, False, True), "applied-better")
        self.assertEqual(region.outcome(90, 80, False, True), "applied-worse")
        self.assertEqual(region.outcome(90, 100, True, True), "region-exact")
        self.assertEqual(region.splice(["a", "b", "c"], 2, 2, "X\nY"), "a\nX\nY\nc\n")
        r = region.pick([{"rid": "x", "class": "reg", "differing": 9}, {"rid": "y", "class": "shape", "differing": 2}], "auto")
        self.assertEqual(r["rid"], "y")

    def test_eval_sets_in_gate_files(self):
        import inspect
        self.assertIn('"eval"', inspect.getsource(gate.set_files))


class ExportDrops(unittest.TestCase):
    def test_leaks_flags_never_train_text_and_eval_function(self):
        import re

        from tools.dk.platforms.gc_mwcc import llmdata
        rec = {"tags": {"unit": "u", "function": "fn_1"},
               "messages": [{"content": "sys"}, {"content": "```cpp\nvoid fn_8054F858(int a) { a++; }\n```"}]}
        pats = [re.compile(r"\bfn_8054F858\s*\(")]
        self.assertEqual(llmdata.leaks(rec, set(), set(), set(), pats), "never-train definition in text")
        self.assertEqual(llmdata.leaks(rec, set(), {"fn_1"}, set(), []), "function")
        self.assertIsNone(llmdata.leaks(rec, set(), set(), set(), []))
        # dataset export filters its rows and DPO pairs through exactly this call
        import inspect
        self.assertGreaterEqual(inspect.getsource(llmdata).count("leaks("), 4)


class Census(unittest.TestCase):
    def test_hunks_monotone_band(self):
        from tools.dk.platforms.gc_mwcc import census
        self.assertEqual(census.hunks("a\nb\nc\nd", "a\nX\nc\nY"), 2)
        self.assertEqual(census.hunks("a", "a"), 0)
        self.assertTrue(census.monotone([50, 50, 70, 100]))
        self.assertFalse(census.monotone([50, 40, 100]))
        self.assertEqual(census.band(0x100), "<=0x200")
