import unittest

from tools.dk.platforms.gc_mwcc.llmdata import (WEIGHTS, answer_debts, cap_generated_tokens, distinct_versions, leaks, local_names, mix,
                                                correct_rows, never_definitions, record, norm, orders, parse_weights, render, reordered, steps, zero_starts)
from tools.dk.platforms.gc_mwcc.llmtry import Context, Ingredients, build_messages, score_text

EXACT = """void Foo::Bar(int n) {
    int a = n + 1;
    float b;
    Thing* c = this->thing;
    b = 2.0f;
    c->Use(a, b);
}"""


def rec(**kw):
    r = {"source": "history", "unit": "game/x/Foo", "function": "Bar__3FooFi", "size": 0x40, "asm": "0x00 li r3, 0",
         "asm_shape": "S1", "decls": "struct Foo;", "wrong": EXACT.replace("n + 1", "n + 2"), "wrong_percent": 97.5,
         "rows": ["0x04 addi r3,r3,1 arg addi r3,r3,2"], "diff_class": "shape", "exact": EXACT,
         "provenance": {"commit": "abc", "time": 1}, "compiler": "GC/1.3.2", "flags": "-O4,p"}
    r.update(kw)
    return r


class VersionsTest(unittest.TestCase):
    def test_norm_ignores_comments_and_spacing(self):
        self.assertEqual(norm("int  a; // x\n/* y */ int b;"), "int a; int b;")

    def test_distinct_versions_oldest_commit_each_without_exact(self):
        vs = [({"commit": "c3", "time": 3}, "int f() { return 2; }"),       # newest first, as git log gives
              ({"commit": "c2", "time": 2}, "int f() {  return 1; }"),
              ({"commit": "c1", "time": 1}, "int f() { return 1; }"),
              ({"commit": "c0", "time": 0}, "int f() { return 3; }")]
        got = distinct_versions(vs, "int f() { return 3; }")
        self.assertEqual([m["commit"] for m, _ in got], ["c1", "c3"])

    def test_steps_next_score_and_improving(self):
        st = steps([({"t": 1}, "a", 50.0), ({"t": 2}, "b", 100.0), ({"t": 3}, "c", 90.0), ({"t": 4}, "d", 80.0)])
        self.assertEqual([(s["code"], s["next_percent"], s["improving"]) for s in st],
                         [("a", 100.0, True), ("c", 80.0, False), ("d", 100.0, True)])


class OrderTest(unittest.TestCase):
    def test_local_names_in_order(self):
        self.assertEqual(local_names(EXACT), ["a", "b", "c"])

    def test_orders_nearest_first_and_capped(self):
        got = orders(["a", "b", "c"], 10)
        self.assertEqual(got[0], ["b", "a", "c"])
        self.assertNotIn(["a", "b", "c"], got)
        self.assertEqual(len(orders(["a", "b", "c"], 2)), 2)

    def test_reordered_moves_declaration_only(self):
        got = reordered(EXACT, "Bar__3FooFi", ["b", "a", "c"])
        self.assertIsNotNone(got)
        self.assertLess(got.index("float b;"), got.index("int a"))
        self.assertEqual(sorted(norm(got).split()), sorted(norm(EXACT).split()))


class RenderTest(unittest.TestCase):
    def test_render_is_llm_try_round_one(self):
        r = rec()
        out = render(r)
        want = build_messages(Context(r["function"], r["asm"], r["decls"], "", r["wrong"]), r["wrong"],
                              r["wrong_percent"], r["rows"], Ingredients(), True)
        self.assertEqual(out["messages"][:2], want)
        self.assertEqual(out["messages"][2], {"role": "assistant", "content": "```cpp\n" + EXACT + "\n```"})
        self.assertIn("scores 97.50%", out["messages"][1]["content"])


class RowsFixTest(unittest.TestCase):
    """Records made while relative_rows left a function at offset 0 with offsets 4 too small."""
    OLD = ["  ...",
           "       li r3, 0x0                           arg  li r5, 0x0",
           "    0  cmpwi r3, 0x0                             cmpwi r3, 0x0",
           "    4  beq C                                     beq C"]
    NEW = ["  ...",
           "    0  li r3, 0x0                           arg  li r5, 0x0",
           "    4  cmpwi r3, 0x0                             cmpwi r3, 0x0",
           "    8  beq 10                                    beq 10"]
    ZERO = {("game/x/Foo", "Bar__3FooFi")}

    def test_zero_starts_come_from_the_splits_and_symbols(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "splits.txt").write_text("Sections:\n\t.text type:code align:4\n\n"
                                                "game/x/Foo.cpp:\n\t.text       start:0x80001000 end:0x80001100\n"
                                                "\t.data       start:0x80200000 end:0x80200010\n"
                                                "game/x/Baz.cpp:\n\t.text       start:0x80001100 end:0x80001200\n")
            (Path(d) / "symbols.txt").write_text("Bar__3FooFi = .text:0x80001000; // type:function size:0x20\n"
                                                 "Next__3FooFv = .text:0x80001020; // type:function size:0x20\n"
                                                 "Baz = .text:0x80001100; // type:function size:0x20\n"
                                                 "Tab = .data:0x80200000; // type:object size:0x10\n")
            cfg = SimpleNamespace(root=Path(d), platform_options={"splits": "splits.txt", "symbols": "symbols.txt"})
            self.assertEqual(zero_starts(cfg), {("game/x/Foo", "Bar__3FooFi"), ("game/x/Baz", "Baz")})

    def test_rows_of_a_function_at_offset_zero_are_moved_once(self):
        recs = [rec(unit="main/game/x/Foo", rows=list(self.OLD), source="synth"),
                rec(unit="main/game/x/Foo", function="Next__3FooFv", rows=list(self.OLD), source="synth"),   # not at offset 0: right already
                rec(unit="main/game/x/Foo", rows=list(self.OLD), rows_fixed=1),                               # made after the fix: kept
                rec(unit="main/game/x/Foo", rows=[], wrong="")]                                               # a direct pair: no rows
        self.assertEqual(correct_rows(recs, self.ZERO), {"synth": 1})
        self.assertEqual(recs[0]["rows"], self.NEW)
        self.assertEqual(recs[1]["rows"], self.OLD)
        self.assertEqual(recs[2]["rows"], self.OLD)
        self.assertEqual([r.get("rows_fixed") for r in recs], [1, 1, 1, None])
        self.assertEqual(correct_rows(recs, self.ZERO), {})                           # idempotent
        self.assertEqual(recs[0]["rows"], self.NEW)

    def test_rows_that_do_not_parse_stay_as_they_are(self):
        recs = [rec(unit="main/game/x/Foo", rows=["not a row"])]
        self.assertEqual(correct_rows(recs, self.ZERO), {"unparsed": 1})
        self.assertEqual(recs[0]["rows"], ["not a row"])
        self.assertNotIn("rows_fixed", recs[0])

    def test_new_records_carry_the_mark_of_the_rows_they_were_given(self):
        from types import SimpleNamespace
        from unittest import mock
        ctx = SimpleNamespace(asm="0 nop", decls="")
        f = {"unit": "main/game/x/Foo", "function": "Bar__3FooFi", "size": 4, "asm_shape": "S"}
        with mock.patch("tools.dk.platforms.gc_mwcc.llmtry.prepare", return_value=(None, None, None, None, None, ctx)):
            fresh = record(None, f, "history", "w", "t", {"percent": 90.0, "rows": self.NEW, "rows_fixed": 1}, "e", {}, {})
            cached = record(None, f, "history", "w", "t", {"percent": 90.0, "rows": self.OLD}, "e", {}, {})
        self.assertEqual(fresh["rows_fixed"], 1)
        self.assertEqual(cached["rows_fixed"], 0)                # made by the old code: export puts it right
        self.assertEqual(correct_rows([fresh, cached], self.ZERO), {"history": 1})
        self.assertEqual((fresh["rows"], cached["rows"]), (self.NEW, self.NEW))


class MixTest(unittest.TestCase):
    def never(self):
        return ({"game/bench/U"}, {"Held__3FooFv"}, {"BAD"})

    def test_leakage_by_unit_shape_and_text(self):
        pats = never_definitions(None, {"Held__3FooFv", "c_held"})
        tags = {"unit": "game/x/Foo", "function": "Bar__3FooFi", "asm_shape": "S1"}
        ok = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "f->Held();"}], "tags": tags}
        self.assertIsNone(leaks(ok, *self.never(), pats))
        call = {"role": "user", "content": "Foo::Held();\n"}
        self.assertIsNone(leaks({**ok, "messages": [ok["messages"][0], call]}, *self.never(), pats))
        self.assertEqual(leaks({**ok, "tags": {**tags, "unit": "game/bench/U"}}, *self.never(), pats), "unit")
        self.assertEqual(leaks({**ok, "tags": {**tags, "asm_shape": "BAD"}}, *self.never(), pats), "asm shape")
        body = {"role": "user", "content": "void Foo::Held() { }"}
        self.assertEqual(leaks({**ok, "messages": [ok["messages"][0], body]}, *self.never(), pats),
                         "never-train definition in text")
        cdef = {"role": "user", "content": "int c_held(int x) {\n return x; }"}
        self.assertTrue(leaks({**ok, "messages": [ok["messages"][0], cdef]}, *self.never(), pats))

    def test_mix_dedupes_weights_and_drops(self):
        base = {"messages": [{"role": "system", "content": "r"}, {"role": "user", "content": "asm"},
                             {"role": "assistant", "content": "```cpp\nint g() { return 0; }\n```"}],
                "tags": {"unit": "game/x/G", "function": "g", "asm_shape": "S2", "source": "base"}}
        regs = rec(source="regs", diff_class="register", wrong="void Foo::Bar(int n) { }")
        leak = rec(unit="game/bench/U", wrong="other")
        out, card = mix([rec(), rec(), regs, leak], [base], self.never(), [], 10**6, 2.2, upsample=True)
        self.assertEqual(card["records"], 3)                  # history (one duplicate gone), regs, base
        self.assertEqual(card["dropped"]["duplicate"], 1)
        self.assertEqual(card["leakage"], {"unit": 1})
        self.assertEqual(len(out), WEIGHTS["history"] + WEIGHTS["regs"] + WEIGHTS["base"])
        self.assertEqual({r["tags"]["source"] for r in out}, {"history", "regs", "base"})
        self.assertTrue(all(r["tags"]["approx_tokens"] > 0 for r in out))
        short, card2 = mix([rec()], [], self.never(), [], 5, 2.2, upsample=False)
        self.assertEqual((short, card2["dropped"]["too long"]), ([], 1))


    def test_weights_parse_over_the_defaults(self):
        w = parse_weights("synth=0.5, attempts=4")
        self.assertEqual((w["synth"], w["attempts"], w["base"]), (0.5, 4.0, WEIGHTS["base"]))
        self.assertEqual(parse_weights(""), WEIGHTS)
        for bad in ("nosuch=1", "synth", "synth=-1", "synth=x"):
            with self.assertRaises(ValueError):
                parse_weights(bad)

    def test_generated_share_of_trained_tokens_is_capped(self):
        tags = [{"source": "base", "approx_tokens": 100, "sample_weight": 1.0},
                {"source": "synth", "approx_tokens": 100, "sample_weight": 2.0},
                {"source": "attempts", "approx_tokens": 100, "sample_weight": 4.0}]
        f = cap_generated_tokens(tags, 0.5)
        self.assertAlmostEqual(f, 1 / 6, places=3)
        gen = sum(t["approx_tokens"] * t["sample_weight"] for t in tags[1:])
        self.assertAlmostEqual(gen / (gen + 100), 0.5, places=2)
        self.assertAlmostEqual(tags[2]["sample_weight"] / tags[1]["sample_weight"], 2.0, places=2)   # ratios kept
        self.assertEqual(cap_generated_tokens(tags, 0.9), 1.0)                 # under the cap: unchanged
        self.assertEqual(cap_generated_tokens(tags, 1.0), 1.0)

    def test_mix_card_has_trained_token_shares(self):
        regs = rec(source="regs", diff_class="register", wrong="void Foo::Bar(int n) { }")
        out, card = mix([rec(), regs], [], self.never(), [], 10**6, 2.2, upsample=False,
                        weights={**WEIGHTS, "regs": 5, "history": 1}, max_generated_share=0.5)
        src = card["sources"]
        self.assertEqual((src["regs"]["weight"], src["history"]["weight"]), (5, 1))
        self.assertLess(card["generated_weight_scale"], 1.0)
        self.assertAlmostEqual(card["generated_effective_share"], 0.5, delta=0.02)
        self.assertAlmostEqual(src["regs"]["effective_share"] + src["history"]["effective_share"], 1.0, places=2)
        self.assertEqual(card["effective_tokens_per_epoch"], sum(s["effective_tokens"] for s in src.values()))
        regs_row = next(r for r in out if r["tags"]["source"] == "regs")
        self.assertLess(regs_row["tags"]["sample_weight"], 5)
        _, free = mix([rec(), regs], [], self.never(), [], 10**6, 2.2, upsample=False,
                      weights={**WEIGHTS, "regs": 5, "history": 1})
        self.assertEqual(free["generated_weight_scale"], 1.0)                  # no cap unless asked
    def test_answer_debts(self):
        self.assertTrue(answer_debts("void f(S* h) {\n    if ((int)h == 0) g();\n}"))
        self.assertFalse(answer_debts("void f(float h) {\n    if ((int)h == 0) g();\n}"))      # a conversion
        self.assertTrue(answer_debts("void f() {\n    if ((int)m10 != 0) g();\n}", "struct S {\n    T* m10;\n};"))
        self.assertTrue(answer_debts("void f() {\n    int var_r31 = g();\n}"))
        self.assertTrue(answer_debts("void f() {\n    float f31 = x;\n}"))
        self.assertFalse(answer_debts("void f() {\n    float y = 1.0f; x->f31 = y;\n}"))
        self.assertFalse(answer_debts("asm void f() {\n    asm { li r3, 4 }\n}"))
        self.assertTrue(answer_debts("void f(int i) {\n    int off = i * 0x5C;\n    *(int*)((char*)&lbl_1 + off) = 0;\n}"))
        self.assertTrue(answer_debts("void f(int i) {\n    *(int*)((char*)p + i * 0x5C) = 0;\n}"))
        self.assertFalse(answer_debts("void f(S* o) {\n    o = (S*)((unsigned char*)o + ((o->m6 * 2 + 3) & ~3));\n}"))
        self.assertEqual(answer_debts("void f(S* p) {\n    if (p == 0) g(p->m4);\n}"), [])

    def test_scores_round_down(self):
        self.assertEqual([score_text(x) for x in (99.997, 99.99, 100.0, 97.5)], ["99.99%", "99.99%", "100.00%", "97.50%"])

if __name__ == "__main__":
    unittest.main()
