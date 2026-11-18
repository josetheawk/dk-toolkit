import unittest

from tools.dk.platforms.gc_mwcc.closecmd import tally
from tools.dk.platforms.gc_mwcc.closequeue import queue_rows


def row(fn, size=200, fuzzy=98.0, klass="near-shape", worker="claude", unit="u/a", frozen="", address=0):
    return {"unit": unit, "function": fn, "size": size, "fuzzy": fuzzy, "class": klass, "worker": worker,
            "frozen": frozen, "address": address}


class QueueTest(unittest.TestCase):
    def test_skips_frozen_blocked_benchmark_and_far(self):
        rows = [row("a"), row("b", worker="frozen"), row("c", frozen="gate-lora2"), row("d", worker="blocked", klass="sinit"),
                row("e", unit="u/bench"), row("f", fuzzy=50.0), row("g", fuzzy=0.0, klass="fresh"), row("h"), row("i"),
                row("j", size=9000)]
        got = queue_rows(rows, {("u/a", "h")}, {"u/bench"}, {("u/a", "i")}, 80.0, 2048, 4096)
        self.assertEqual([r["function"] for r in got], ["a"])

    def test_small_first_then_closest(self):
        rows = [row("big", size=3000, fuzzy=99.9, klass="giant"), row("near", fuzzy=99.5), row("far", fuzzy=91.0, size=40),
                row("same", fuzzy=99.5, size=100)]
        got = queue_rows(rows, set(), set(), set(), 90.0, 2048, 4096)
        self.assertEqual([r["function"] for r in got], ["same", "near", "far", "big"])


class TallyTest(unittest.TestCase):
    def test_per_kind_counts(self):
        rows = [{"fix_kind": ["a", "b"], "exact": True}, {"fix_kind": ["a"], "exact": False}, {"fix_kind": [], "exact": False}]
        self.assertEqual(tally(rows, "fix_kind"), {"a": [1, 2], "-": [0, 1], "b": [1, 1]})


if __name__ == "__main__":
    unittest.main()
