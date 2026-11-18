import json
import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import regions

SRC = """void f(int a) {
    int x = a;
    if (a) {
        x = 1;
        x = 2;
    }
    g(x);
}
"""


def body():
    return cbody.parse_body(SRC, SRC.index("{"))


def pair(rows):
    """rows: (target text or None, our text or None, our line or None, flagged)."""
    lt, rt = [], []
    for t, o, line, flag in rows:
        a = {"instruction": {"formatted": t}} if t is not None else {}
        b = {"instruction": {"formatted": o, **({"line_number": line} if line else {})}} if o is not None else {}
        if flag:
            a["diff_kind"] = b["diff_kind"] = "DIFF_ARG_MISMATCH"
        lt.append(a)
        rt.append(b)
    return lt, rt


ROWS = [("stwu r1, -0x10(r1)", "stwu r1, -0x10(r1)", 1, False),       # 0 frame ({ line)
        ("mr r31, r3", "mr r30, r3", 2, True),                         # 1 int x = a: register only
        ("cmpwi r3, 0", "cmpwi r3, 0", 3, False),                      # 2 if head
        ("beq 0x20", "beq 0x24", 3, False),                            # 3 branch offset moved, not flagged
        ("li r31, 1", "li r31, 1", 4, False),                          # 4
        ("li r31, 2", "li r31, 3", 5, True),                           # 5 shape
        ("mr r3, r31", "mr r3, r31", 7, False),                        # 6 g(x)
        ("lwz r0, 0(r4)", "lwz r0, 0(r4)", 40, False),                 # 7 inlined header line: the call site
        ("bl g", None, None, True),                                    # 8 target only: joins the previous row
        ("blr", "blr", 8, False)]                                      # 9 frame (} line)


class RegionsTest(unittest.TestCase):
    def test_every_row_in_exactly_one_region(self):
        lt, rt = pair(ROWS)
        for r_max in (48, 2, 1):
            regs = regions.segment(lt, rt, SRC, body(), r_max)
            idx = sorted(k for r in regs for k in r["indices"])
            self.assertEqual(idx, list(range(len(ROWS))), r_max)

    def test_frame_call_site_and_target_only_rows(self):
        lt, rt = pair(ROWS)
        regs = regions.segment(lt, rt, SRC, body(), 2)
        where = {k: r["rid"] for r in regs for k in r["indices"]}
        self.assertEqual(where[0], where[9])
        self.assertTrue(where[0].startswith("frame"))
        self.assertEqual(where[7], where[6])                           # inlined helper line -> g(x)
        self.assertEqual(where[8], where[6])                           # target-only row -> previous region

    def test_cut_pack_and_atomic(self):
        lt, rt = pair(ROWS)
        whole = regions.segment(lt, rt, SRC, body(), 48)
        self.assertEqual([r["kind"] for r in whole], ["stmt", "frame"])   # everything packs under one cap
        cut = regions.segment(lt, rt, SRC, body(), 2)
        kinds = {r["rid"]: r for r in cut}
        self.assertTrue(any(r["kind"] == "atomic" for r in cut))      # g(x) holds 3 rows > 2
        self.assertEqual(len([r for r in cut if r["kind"] != "frame"]), 4)
        self.assertEqual(len(kinds), len(cut))                         # unique ids

    def test_classes(self):
        lt, rt = pair(ROWS)
        cls = {tuple(r["indices"]): r["class"] for r in regions.segment(lt, rt, SRC, body(), 2)}
        self.assertEqual(cls[(1,)], "reg")
        self.assertEqual(cls[(2, 3)], "exact")                         # unflagged branch offsets are equal
        self.assertEqual(cls[(4, 5)], "shape")
        lt2, rt2 = pair([("lwz r3, 0(r4)", "lwz r5, 4(r4)", 2, True), ("lwz r5, 4(r4)", "lwz r3, 0(r4)", 2, True)])
        self.assertEqual(regions.classify(lt2, rt2, [0, 1]), ("sched", 0))
        self.assertEqual(regions.classify(lt, rt, [4, 5]), ("shape", 1))

    def test_deterministic_and_stats(self):
        lt, rt = pair(ROWS)
        a = json.dumps(regions.segment(lt, rt, SRC, body(), 2), sort_keys=True)
        self.assertEqual(a, json.dumps(regions.segment(lt, rt, SRC, body(), 2), sort_keys=True))
        st = regions.stats(regions.segment(lt, rt, SRC, body(), 2), len(ROWS), 1)
        self.assertEqual(st["shape_regions"], 2)                      # x = 2 and g(x) (a row ours lacks)
        self.assertEqual(st["rows"], 10)
        self.assertEqual(st["unmapped_share"], 0.1)


if __name__ == "__main__":
    unittest.main()
