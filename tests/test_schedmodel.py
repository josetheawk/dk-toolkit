"""dk sched's scheduler model (schedmodel.py) on synthetic scheduler records."""

import unittest

from tools.dk.platforms.gc_mwcc import schedmodel as M

OPS = {"1": {"name": "LWZ", "rank": 3, "stage": 3, "lat": 2, "cycles": [1, 1, 0], "ser": 0},
       "2": {"name": "ADD", "rank": 2, "stage": 2, "lat": 1, "cycles": [1, 0, 0], "ser": 0},
       "3": {"name": "MULLW", "rank": 2, "stage": 1, "lat": 5, "cycles": [5, 0, 0], "ser": 0},
       "4": {"name": "ADDI", "rank": 2, "stage": 2, "lat": 1, "cycles": [1, 0, 0], "ser": 0},
       "5": {"name": "MR", "rank": 1, "stage": 2, "lat": 1, "cycles": [1, 0, 0], "ser": 0},
       "6": {"name": "BL", "rank": 0, "stage": 0, "lat": 0, "cycles": [0, 0, 0], "ser": 0}}


def rec(ops, edges, pre=True, srcs=None):
    """ops: opcode per node in codegen order; edges: (from, to, latency). Heights/deadlines computed here."""
    n = len(ops)
    lat = [OPS[str(o)]["lat"] for o in ops]
    succ = {i: [(j, l) for a, j, l in edges if a == i] for i in range(n)}
    height = [0] * n
    for i in reversed(range(n)):
        height[i] = max([lat[i]] + [height[j] + l for j, l in succ[i]])
    crit = max(height)
    preds = [sum(1 for _, j, _ in edges if j == i) for i in range(n)]
    dag = [{"i": i, "lat": lat[i], "latest": crit - height[i], "height": height[i], "preds": preds[i],
            "succ": [[j, l] for j, l in succ[i]]} for i in range(n)]
    insns = [{"op": o, "flags": 0, "src": (srcs or [10] * n)[i], "args": []} for i, o in enumerate(ops)]
    r = {"fn": "f", "pre": pre, "block": 0, "insns": insns, "dag": dag}
    r["out"] = M.schedule(M.nodes_of(dict(r, _ops=OPS)), OPS, pre)[0]
    return r


def doc(*recs):
    return {"schema": "sched.v1", "ops": OPS, "blocks": list(recs)}


class ScheduleTest(unittest.TestCase):
    def test_loads_are_hoisted_over_their_users(self):
        # lwz p; lwz [0]; add; lwz [4]; add; mullw: the second load overtakes the first add (height)
        r = rec([1, 1, 2, 1, 2, 3], [(0, 1, 2), (0, 3, 2), (1, 2, 2), (3, 4, 2), (2, 5, 1), (4, 5, 1)])
        self.assertEqual(r["out"], [0, 1, 3, 2, 4, 5])

    def test_codegen_order_breaks_a_tie_and_the_lever_flips_it(self):
        r = dict(rec([4, 4, 6], [(0, 2, 0), (1, 2, 0)], srcs=[10, 11, 11]), _ops=OPS)
        self.assertEqual(r["out"], [0, 1, 2])
        self.assertEqual(M.decide(r, 1, 0)["rule"], "codegen order")
        self.assertTrue(M.levers(r, 1, 0))

    def test_the_critical_path_beats_codegen_order(self):
        # addi a (feeds bl only); addi b -> mr: b heads the longer chain, its deadline has come
        r = dict(rec([4, 4, 5, 6], [(0, 3, 0), (1, 2, 1), (2, 3, 0)]), _ops=OPS)
        self.assertEqual(r["out"][0], 1)
        self.assertEqual(M.decide(r, 0, 1)["rule"], "deadline")
        self.assertEqual(M.levers(r, 0, 1), [])

    def test_uncovering_beats_codegen_order(self):
        # li c, addi a -> add (two preds); addi b -> mr (its only pred): b uncovers the copy, so it goes first
        r = dict(rec([4, 4, 2, 4, 5, 6], [(0, 2, 1), (1, 2, 1), (2, 5, 0), (3, 4, 1), (4, 5, 0)]), _ops=OPS)
        self.assertEqual(r["out"][0], 3)
        self.assertEqual(M.decide(r, 0, 3)["rule"], "uncovering")
        self.assertEqual(M.levers(r, 0, 3), [])

    def test_check_reproduces_records(self):
        r = rec([1, 1, 2, 1, 2, 3], [(0, 1, 2), (0, 3, 2), (1, 2, 2), (3, 4, 2), (2, 5, 1), (4, 5, 1)])
        self.assertEqual(M.check(doc(r)), {("pre", True): 1})


def row(text, diff=None):
    e = {"instruction": {"formatted": text}}
    if diff:
        e["diff_kind"] = diff
    return e


class ExplainTest(unittest.TestCase):
    def test_a_swapped_pair_is_explained_with_a_lever(self):
        r = rec([4, 4, 6], [(0, 2, 0), (1, 2, 0)], srcs=[10, 11, 11])
        ours = [row("addi r5, r1, 0x8", "x"), row("addi r6, r1, 0xc", "x"), row("bl f")]
        target = [row("addi r6, r1, 0xc", "x"), row("addi r5, r1, 0x8", "x"), row("bl f")]
        classes = [{"row": 0, "class": "SCHEDULE"}, {"row": 1, "class": "SCHEDULE"}]
        rep = M.explain(doc(r), "f__Fv", target, ours, classes)
        self.assertEqual(len(rep), 1)
        self.assertEqual(rep[0]["status"], "codegen order")
        self.assertEqual(rep[0]["decide"]["rule"], "codegen order")
        self.assertEqual(rep[0]["x"]["line"], 11)

    def test_other_instructions_make_a_mixed_region(self):
        r = rec([4, 4, 6], [(0, 2, 0), (1, 2, 0)])
        ours = [row("addi r5, r1, 0x8", "x"), row("addi r6, r1, 0xc", "x"), row("bl f")]
        target = [row("li r6, 0x1", "x"), row("addi r5, r1, 0x8", "x"), row("bl g", "x")]
        rep = M.explain(doc(r), "f", target, ours, [{"row": 0, "class": "SCHEDULE"}])
        self.assertEqual(rep[0]["status"], "mixed")

    def test_families_fold_extended_mnemonics(self):
        self.assertEqual(M.family_asm("clrlwi r0, r3, 24"), "rlwinm")
        self.assertEqual(M.family_asm("cmplwi r0, 0x0"), "cmpli")
        self.assertEqual(M.family_asm("bne 0x40"), "bcond")
        self.assertEqual(M.family_op("BF"), "bcond")
        self.assertEqual(M.short_name("fn_1__3FooFv"), "fn_1")


if __name__ == "__main__":
    unittest.main()
