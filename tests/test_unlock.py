import json
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import closequeue, unlock


def unit(name, code, fns=(), sections=(), complete=False, auto=False):
    return {"name": name, "measures": {"total_code": str(code)},
            "metadata": {"complete": complete, "auto_generated": auto, "source_path": f"src/{name}.cpp"},
            "functions": [{"name": n, "size": str(s), "fuzzy_match_percent": p} for n, s, p in fns],
            "sections": [{"name": n, "size": str(s), "fuzzy_match_percent": p} for n, s, p in sections]}


def row(u, fn, size, pct, cls, worker, frozen="", idle=0, address=0):
    return {"unit": u, "function": fn, "address": address, "size": size, "fuzzy": pct, "class": cls, "worker": worker,
            "frozen": frozen, "attempts": idle, "idle": idle, "data_open": []}


REPORT = {"measures": {"total_code": "100000"}, "units": [
    unit("done", 20000, [("a", 100, 100.0)], [(".text", 100, 100.0)], complete=True),
    unit("zero", 4000, [("z", 400, 100.0)], [(".text", 400, 100.0), (".sdata2", 8, 50.0), (".sbss", 8, 0.0)]),
    unit("one", 8000, [("o1", 200, 99.0), ("o2", 64, 100.0)], [(".text", 264, 99.0), (".data", 16, 100.0)]),
    unit("two", 9000, [("t1", 100, 98.0), ("t2", 120, 95.0)], [(".text", 220, 97.0)]),
    unit("big", 30000, [("g", 0x1000, 99.0)], [(".text", 0x1000, 99.0), (".data", 64, 80.0)]),
    unit("held", 6000, [("h", 80, 99.9)], [(".text", 80, 99.9)]),
    unit("thawed", 3000, [("s", 60, 99.5)], [(".text", 60, 99.5)]),
    unit("auto_1", 500, [("fn_1", 500, 0.0)], [(".text", 500, 0.0)], auto=True),
    unit("layout", 700, [("l", 700, 100.0)], [(".text", 700, 100.0)]),
]}
ROWS = [row("one", "o1", 200, 99.0, "near-register", "cpu", address=1),
        row("two", "t1", 100, 98.0, "near-shape", "lora", address=2),
        row("two", "t2", 120, 95.0, "near-shape", "lora", idle=2, address=3),
        row("big", "g", 0x1000, 99.0, "giant", "claude", address=4),
        row("held", "h", 80, 99.9, "near-register", "frozen", frozen="gate-lora2", address=5),
        row("thawed", "s", 60, 99.5, "near-scheduling", "frozen", frozen="near-sample-60", address=6)]


class UnlockTest(unittest.TestCase):
    def setUp(self):
        self.r = unlock.rank(REPORT, ROWS, tried={"one::o1": 2}, thaw=("near-sample-60", "near-sample-fresh-30"))
        self.by = {u["unit"]: u for u in self.r["units"]}

    def test_order_and_values(self):
        self.assertEqual([u["unit"] for u in self.r["units"]],
                         ["thawed", "one", "big", "zero", "two", "layout", "auto_1"])
        self.assertEqual(self.by["zero"]["cost"], 3)                   # .sdata2 2 + .sbss 1 (order only)
        self.assertEqual(self.by["two"]["cost"], 3 + 3 + 2)            # two near-shape, one with 2 idle nights
        self.assertEqual(self.by["big"]["cost"], 12 + 4 + 2)           # giant + 4 KB + its open .data
        self.assertEqual(self.by["one"]["cost"], 1 + 2)                # cpu_tried from the memo/log counts
        self.assertEqual(self.by["layout"]["blockers"][0]["kind"], "link")
        self.assertEqual(self.by["auto_1"]["blockers"][0]["class"], "no-source")

    def test_last_blocker_and_data_sections(self):
        self.assertEqual(self.by["one"]["last_blocker"], "o1")
        self.assertIsNone(self.by["two"]["last_blocker"])
        self.assertIsNone(self.by["big"]["last_blocker"])              # the data section is a second blocker
        self.assertEqual(self.by["big"]["data_open"], [".data"])
        self.assertEqual(self.by["zero"]["data_open"], [".sdata2", ".sbss"])
        self.assertEqual(self.by["one"]["data_open"], [])

    def test_frozen_listed_never_queued_and_thaw(self):
        self.assertEqual([u["unit"] for u in self.r["held"]], ["held"])
        self.assertIsNone(self.r["held"][0]["value"])
        b = self.by["thawed"]["blockers"][0]
        self.assertEqual((b["worker"], b["thawed"], b["held"]), ("cpu", True, "near-sample-60"))
        none = unlock.rank(REPORT, ROWS, thaw=())
        self.assertEqual(sorted(u["unit"] for u in none["held"]), ["held", "thawed"])
        rows = unlock.annotate(ROWS, self.r)
        q = closequeue.queue_rows(rows, {("held", "h"): "gate-lora2"}, set(), set(), 80.0, 2048, 8192)
        self.assertNotIn("h", [r["function"] for r in q])
        self.assertIn("s", [r["function"] for r in q])
        self.assertTrue(next(r for r in rows if r["function"] == "o1")["last_blocker"])
        self.assertEqual(ROWS[5]["worker"], "frozen")                  # annotate copies, never mutates route's rows
        plain = closequeue.queue_rows(ROWS, {}, set(), set(), 80.0, 2048, 8192)
        self.assertNotIn("s", [r["function"] for r in plain])

    def test_explicit_thaw_releases_gate_lora2(self):
        # 2026-11-18: nothing is thawed by default (every eval set frozen); --thaw opts a set in
        self.assertEqual(unlock.THAW, ())
        self.assertTrue(unlock.rank(REPORT, ROWS, tried={"one::o1": 2})["held"])          # held units listed apart
        r = unlock.rank(REPORT, ROWS, tried={"one::o1": 2}, thaw=("gate-lora2", "near-sample-60", "near-sample-fresh-30"))
        self.assertEqual(r["held"], [])
        b = {u["unit"]: u for u in r["units"]}["held"]["blockers"][0]
        self.assertEqual((b["worker"], b["thawed"], b["held"]), ("cpu", True, "gate-lora2"))
        rows = unlock.annotate(ROWS, r)
        q = closequeue.queue_rows(rows, {}, set(), set(), 80.0, 2048, 8192)
        self.assertIn("h", [r["function"] for r in q])

    def test_linked_curve_and_leverage(self):
        self.assertEqual(self.r["linked_pct"], 20.0)
        self.assertAlmostEqual(self.r["linked_if_done"]["cpu"], 20.0 + 3.0 + 8.0 + 0.7)   # thawed, one, layout
        self.assertEqual(self.r["linked_alone"]["data"], 4.0)
        self.assertAlmostEqual(self.r["linked_if_done"]["blocked"], 20.0 + 4 + 3 + 9 + 30 + 8 + 0.7 + 0.5)
        part = self.r["units"][:2]
        self.assertEqual(unlock.leverage(self.r["units"], 2), round(sum(u["total_code"] for u in part) / 260, 2))

    def test_deterministic_json(self):
        a = json.dumps(unlock.rank(REPORT, ROWS, tried={"one::o1": 2}), indent=1)
        b = json.dumps(unlock.rank(REPORT, list(reversed(ROWS)), tried={"one::o1": 2}), indent=1)
        self.assertEqual(a, b)

    def test_tried_counts_from_logs_and_memos(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "q.jsonl").write_text("\n".join(json.dumps(x) for x in [
                {"unit": "u", "function": "f", "exact": False}, {"unit": "u", "function": "f", "exact": False},
                {"unit": "u", "function": "g", "exact": True}, ]) + "\nnot json\n")
            (d / "memo").mkdir()
            (d / "memo" / "f.json").write_text(json.dumps({"unit": "u", "function": "f", "runs": 1, "best": 99.0}))
            (d / "memo" / "h.json").write_text(json.dumps({"unit": "u", "function": "h", "runs": 3, "best": 98.0}))
            self.assertEqual(unlock.tried_counts([d / "q.jsonl"], [d / "memo", d / "missing"]), {"u::f": 2, "u::h": 3})


if __name__ == "__main__":
    unittest.main()
