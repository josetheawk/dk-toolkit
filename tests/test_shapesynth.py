"""dataset shape-synth (shapesynth.py): the pure parts (invented cases, local names, the T35 check). Synthetic
fixtures only; the compile path is exercised by the command itself."""
import unittest

from tools.dk.platforms.gc_mwcc import shapesynth as ss


class CaseTest(unittest.TestCase):
    def test_case_from_a_compare_is_not_invented(self):
        before = "void f(int x) {\n    if (x == 3) {\n        g();\n    }\n}"
        after = "void f(int x) {\n    switch (x) {\n    case 3:\n        g();\n    }\n}"
        self.assertEqual(ss.invented_cases(before, after), [])

    def test_new_case_value_is_invented(self):
        before = "void f(int x) {\n    switch (x) {\n    case 3:\n        g();\n    }\n}"
        after = "void f(int x) {\n    switch (x) {\n    case 3:\n    case 9:\n        g();\n    }\n}"
        self.assertEqual(ss.invented_cases(before, after), ["9"])

    def test_shared_labels_keep_their_values(self):
        before = "switch (x) {\ncase 1:\n    g();\n    break;\ncase 2:\n    g();\n    break;\n}"
        after = "switch (x) {\ncase 1:\ncase 2:\n    g();\n    break;\n}"
        self.assertEqual(ss.invented_cases(before, after), [])


class NameTest(unittest.TestCase):
    def test_from_member_and_getter(self):
        self.assertEqual(ss.natural_name("car->mTires[i]", "Tire&", set()), "tires")
        self.assertEqual(ss.natural_name("obj->GetSpeed()", "float", set()), "speed")

    def test_hex_members_fall_back_to_type(self):
        self.assertEqual(ss.natural_name("p->m1C", "float", set()), "value")
        self.assertEqual(ss.natural_name("p->m1C", "Foo*", {"ptr"}), "obj")

    def test_never_a_placeholder_or_taken(self):
        for expr, t in (("a->m0", "int"), ("fn_80001000()", "int"), ("", "")):
            n = ss.natural_name(expr, t, {"value", "count", "idx"})
            self.assertFalse(ss._PLACEHOLDER.fullmatch(n), n)
            self.assertNotIn(n, {"value", "count", "idx"})


class CheckTest(unittest.TestCase):
    def rec(self, kinds, cls="shape"):
        return {"fix_kind": kinds, "diff_class": cls}

    def test_tv(self):
        self.assertEqual(ss.tv({"a": 1}, {"a": 5}), 0.0)
        self.assertEqual(ss.tv({"a": 1}, {"b": 1}), 1.0)
        self.assertAlmostEqual(ss.tv({"a": 1, "b": 1}, {"a": 1}), 0.5)

    def test_pass_and_cover(self):
        real = [self.rec(["operator"]), self.rec(["operator", "literal"]), self.rec(["identifier"], "register")]
        synth = [self.rec(["operator"])] * 300 + [self.rec(["literal"])] * 250 + [self.rec(["identifier"], "register")] * 50
        c = ss.check(real, synth, min_cover=200)
        self.assertEqual(c["covered"], 2)                 # identifier has only 50 synthetic examples
        self.assertFalse(c["pass"])                       # 2/3 < 70 %
        c = ss.check(real[:2], synth, min_cover=200)
        self.assertEqual(c["cover_share"], 1.0)
        self.assertLessEqual(c["tv_class"], 0.1)
        self.assertTrue(c["pass"])

    def test_canon_named_value_up_to_name_and_type(self):
        cur = "void f(Car* a) {\n    g(a->mB.m0);\n}"
        prev = "void f(Car* a) {\n    float tireX = a->mB.m0;\n    g(tireX);\n}"
        cand = "void f(Car* a) {\n    __typeof__(a->mB.m0) t0 = a->mB.m0;\n    g(t0);\n}"
        self.assertEqual(ss.canon(cand, cur), ss.canon(prev, cur))
        ref = "void f(Car* a) {\n    Tire& tire = a->mT[0];\n    g(tire.m0);\n}"
        ref_c = "void f(Car* a) {\n    __typeof__(a->mT[0])& e0 = a->mT[0];\n    g(e0.m0);\n}"
        self.assertEqual(ss.canon(ref, cur), ss.canon(ref_c, cur))
        ptr = "void f(Car* a) {\n    Tire* q = a->mP;\n    g(q->m0);\n}"
        ptr_c = "void f(Car* a) {\n    __typeof__(a->mP) t0 = a->mP;\n    g(t0->m0);\n}"
        self.assertEqual(ss.canon(ptr, cur), ss.canon(ptr_c, cur))      # T37: pointer locals compare too
        self.assertNotEqual(ss.canon(ptr_c, cur), ss.canon(ref_c.replace("e0.", "e0->"), cur))
        other = "void f(Car* a) {\n    __typeof__(a->mB) t0 = a->mB;\n    g(t0.m0);\n}"
        self.assertNotEqual(ss.canon(other, cur), ss.canon(prev, cur))

    def test_within(self):
        text = "int a;\nvoid f() { x(); }\nint b;\n"
        sp = (7, text.index("}") + 1)
        self.assertTrue(ss.within(text, text.replace("x();", "y();"), sp))
        self.assertFalse(ss.within(text, text.replace("int b;", "int c;"), sp))


class TableTest(unittest.TestCase):
    def test_cache_key_follows_the_table(self):
        v = ss.table_version()
        self.assertRegex(v, r"^[0-9a-f]{12}$")
        self.assertEqual(v, ss.table_version())

    def test_probe_counts_the_folds(self):
        self.assertEqual({r for r, _ in ss.FOLDS}, {"inline-local", "name-value"})
        self.assertTrue(set(r for r, _ in ss.FOLDS) <= set(ss.SHAPE_TABLE))

    def test_fold_and_name_drawn_evenly(self):
        import random
        moves = [("k written out at its 2 uses", "a")] + [(f"g(x) named once as t0 ({i})", str(i)) for i in range(20)]
        rng = random.Random(1)
        first = [ss.sides("inline-local", moves, rng)[0][0].startswith("k written") for _ in range(2000)]
        self.assertTrue(0.45 < sum(first) / len(first) < 0.55, sum(first))        # 1/2, not 1/21
        self.assertEqual(sorted(ss.sides("inline-local", moves, rng)), sorted(moves))
        other = [("x", "1"), ("y", "2")]
        self.assertEqual(sorted(ss.sides("else-return", other, rng)), other)
        self.assertEqual(ss.FOLD_MARK["name-value"], "folded into its use")



class _FakeFn:
    """build_path's view of a function: states are strings, a fold appends "F<k>", other moves "o<k>"."""

    def __init__(self, reversible=True):
        self.reversible = reversible

    def defn(self, text):
        return text

    def moves(self, text, rules=ss.SHAPE_TABLE, per_rule=ss.PER_RULE):
        return {"else-return": [(f"o{k}", text + f" o{k}") for k in range(3)],
                "inline-local": [(f"q{k} written out at its 2 uses", text + f" F{k}") for k in range(2)] +
                                [(f"g() named once as t0 ({k})", text + f" n{k}") for k in range(2)]}

    def fold_moves(self, text):
        return [("inline-local", label, new) for label, new in self.moves(text)["inline-local"]
                if ss.FOLD_MARK["inline-local"] in label]

    def typed(self, text):
        return text

    def rename_new(self, prev, text):
        return text

    def reverse(self, cur, prev):
        return "x" if self.reversible else None


class TargetedDrawTest(unittest.TestCase):
    def test_schedule_share_order_and_exhaustion(self):
        pool = list(range(0, 1000, 4))                          # 250 functions with a foldable local
        jobs = ss.schedule(1000, pool, 0.3, 7)
        anys = [i for i, m in jobs if m == "any"]
        folds = [i for i, m in jobs if m == "fold"]
        self.assertEqual(anys, list(range(1000)))              # every input once, in order
        self.assertEqual(folds, pool)                          # the pool in its order, each once
        head = jobs[:600]                                      # while the pool lasts: ~30 % fold draws
        share = sum(m == "fold" for _i, m in head) / len(head)
        self.assertTrue(0.25 < share < 0.35, share)
        self.assertEqual(ss.schedule(5, [], 0.3, 7), [(i, "any") for i in range(5)])
        self.assertEqual(ss.schedule(5, [1, 2], 0.0, 7), [(i, "any") for i in range(5)])
        self.assertEqual(jobs, ss.schedule(1000, pool, 0.3, 7))

    def test_draw_kinds_compile_apart(self):
        from pathlib import Path
        w = Path("/w")
        self.assertNotEqual(ss.work_dir(w, "f", "any"), ss.work_dir(w, "f", "fold"))
        self.assertEqual(ss.work_dir(w, "f"), ss.work_dir(w, "f", "any"))
        self.assertEqual(ss.work_dir(w, "f").parent, w)

    def test_fold_draw_starts_with_a_fold(self):
        import random
        for seed in range(30):
            steps = ss.build_path(_FakeFn(), "S", 3, random.Random(seed), {}, fold_first=True)
            self.assertEqual(len(steps), 3)
            self.assertIn(ss.FOLD_MARK["inline-local"], steps[0]["label"])
            self.assertEqual(steps[0]["rule"], "inline-local")
        stats: dict = {}
        self.assertEqual(ss.build_path(_FakeFn(False), "S", 2, random.Random(1), stats, fold_first=True), [])
        self.assertEqual(stats["fold draw: no fold made"], 1)   # no fallback to another move
        firsts = {ss.build_path(_FakeFn(), "S", 1, random.Random(s), {})[0]["label"][:1] for s in range(40)}
        self.assertIn("o", firsts)                             # an ordinary draw is not forced


if __name__ == "__main__":
    unittest.main()
