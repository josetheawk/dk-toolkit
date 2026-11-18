"""Lining the oracle's record up with our instructions when the counts differ (oracle.emitted_rows).

Synthetic: many dropped branches, extra instructions of ours, source lines breaking a tie in repeated code."""

import unittest

from tools.dk.platforms.gc_mwcc import oracle


OPS = {"lwz": 34, "stw": 49, "addi": 63, "mr": 139, "li": 137, "bl": 1, "b": 5, "cmpwi": 83}


def program(n: int, seed: int = 7) -> list[str]:
    import random
    rnd = random.Random(seed)
    return [rnd.choice(["lwz", "stw", "addi", "mr", "li", "bl", "cmpwi", "b"]) for _ in range(n)]


class AlignTest(unittest.TestCase):
    def test_many_dropped_branches_line_up(self):
        ours = ["lwz" if j % 50 == 49 else mn for j, mn in enumerate(program(400))]  # no b right before
        listed, want = [], []
        for j, mn in enumerate(ours):                 # the record lists a branch-to-next after every 50th
            listed.append(mn)
            want.append(j)
            if j % 50 == 49:
                listed.append("b")
                want.append(None)
        rec = {"insns": len(listed), "ops": [OPS[x] for x in listed]}
        self.assertEqual(len(listed) - len(ours), 8)                  # more than the old limit of 3
        self.assertEqual(oracle.emitted_rows(rec, ours), want)

    def test_extra_instructions_of_ours(self):
        ours = program(120, seed=3)
        drop = {30, 31, 90}                            # ours has three instructions the record does not list
        listed = [mn for j, mn in enumerate(ours) if j not in drop]
        rec = {"insns": len(listed), "ops": [OPS[x] for x in listed]}
        got = oracle.emitted_rows(rec, ours)
        self.assertNotIn(None, got)                                   # every listed one is ours
        self.assertEqual([ours[j] for j in got], listed)               # (a run of equal ones may shift)
        self.assertEqual(got, sorted(got))

    def test_source_lines_pick_the_dropped_copy(self):
        block = ["lwz", "cmpwi", "b", "addi", "stw"]
        listed, lines = [], []
        for copy in range(6):                          # six identical unrolled copies, one line each
            listed += block
            lines += [10 + copy] * len(block)
        ours = [mn for k, mn in enumerate(listed) if not (lines[k] in (12, 14) and mn == "b")]
        our_lines = [ln for k, ln in enumerate(lines) if not (ln in (12, 14) and listed[k] == "b")]
        rec = {"insns": len(listed), "ops": [OPS[x] for x in listed], "lines": lines}
        got = oracle.emitted_rows(rec, ours, lines=our_lines)
        dropped = [k for k, j in enumerate(got) if j is None]
        self.assertEqual([lines[k] for k in dropped], [12, 14])
        self.assertTrue(all(lines[k] == our_lines[j] for k, j in enumerate(got) if j is not None))

    def test_refuses_what_cannot_be_the_same_function(self):
        self.assertIsNone(oracle.emitted_rows({"insns": 9, "ops": [0] * 9}, ["lwz"]))
        ours = program(200, seed=5)
        other = program(200, seed=11)                  # same length +2, unrelated opcodes
        rec = {"insns": 202, "ops": [OPS[x] for x in other] + [5, 5]}
        self.assertIsNone(oracle.emitted_rows(rec, ours))



if __name__ == "__main__":
    unittest.main()
