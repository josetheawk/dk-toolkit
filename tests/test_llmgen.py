import unittest
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc.llmdata import (AnswerJudge, Examples, LEVELS, cap_generated, cap_per_function, difficulty, dpo_pairs,
                                                features, levels, mix, render, stage_of)
from tools.dk.platforms.gc_mwcc.llmgen import (bt_fix, cap_rule_share, class_from_rows, clean_answer, literal_labels, synth_artifact, defines, forward_names, rename_all, signature_part, symbol_names, mutants, new_name, new_symbols, old_declarations,
                                               pick_spread, renamed, with_declarations)

OLD = """#include "game/Foo.h"

struct ObjThing {
    int a;
};

extern "C" void fn_80001234(ObjThing* p);
extern float lbl_80001000;

void Foo::Run(int n) {
    ObjThing* t = (ObjThing*)this->mPtr;
    fn_80001234(t);
}
"""

SRC = """struct Car { int speed; int gear; };

void Tick(Car* c, int n) {
    int i;
    int total = 0;
    for (i = 0; i < n; i++) {
        total += i;
    }
    c->speed = total;
    c->gear = n;
}
"""

ASM = """    0  stwu r1, -0x10(r1)
.L_80001000:
    4  bl fn_80002000
    8  cmpwi r3, 0
    C  bne .L_80001000
   10  bctrl
   14  lfs f1, 0x8(r3)
   18  blr"""


def rec(**kw):
    r = {"source": "history", "unit": "game/x/Foo", "function": "Bar__3FooFi", "size": 0x40, "asm": ASM,
         "asm_shape": "S1", "decls": "struct Foo;", "wrong": "void Foo::Bar(int n) { g(n + 2); }", "wrong_percent": 97.5,
         "rows": ["    4  addi r3, r3, 1  arg  addi r3, r3, 2"], "diff_class": "shape",
         "exact": "void Foo::Bar(int n) { g(n + 1); }", "provenance": {}}
    r.update(kw)
    return r


class HistoryRecoveryTest(unittest.TestCase):
    def test_defines_kinds(self):
        self.assertTrue(defines("struct ObjThing {\n int a;\n};", "ObjThing"))
        self.assertTrue(defines('extern "C" void fn_80001234(ObjThing* p);', "fn_80001234"))
        self.assertTrue(defines("extern float lbl_80001000;", "lbl_80001000"))
        self.assertTrue(defines("typedef int MyInt;", "MyInt"))
        self.assertFalse(defines("void g() { fn_80001234(0); }", "fn_80001234"))       # a use, not a declaration
        self.assertFalse(defines("int x = p->lbl_80001000;", "lbl_80001000"))

    def test_old_declarations_found_and_inserted(self):
        got = old_declarations(OLD, {"ObjThing", "fn_80001234"}, have="")
        self.assertEqual(len(got), 2)
        self.assertTrue(got[0].startswith("struct ObjThing"))
        self.assertEqual(old_declarations(OLD, {"ObjThing"}, have=got[0]), [])         # already there: not again
        text = "int a;\nvoid f() {}\n"
        self.assertEqual(with_declarations(text, 7, ["struct S;"]), "int a;\nstruct S;\n\nvoid f() {}\n")


class ForwardNamesTest(unittest.TestCase):
    OLD_SYMS = """fn_80001234 = .text:0x80001234; // type:function size:0x20
fn_80001300__3FooFv = .text:0x80001300; // type:function size:0x20
fn_80001400__3FooFv = .text:0x80001400; // type:function size:0x20
lbl_80300000 = .data:0x80300000; // type:object size:0x4
__ct__3FooFv = .text:0x80001500; // type:function size:0x20
fn_80001600__FPi = .text:0x80001600; // type:function size:0x20
fn_80001700 = .text:0x80001700; // type:function size:0x20
"""
    NEW_SYMS = """ResetAll = .text:0x80001234; // type:function size:0x20
Tick__3FooFv = .text:0x80001300; // type:function size:0x20
Draw__3BarFv = .text:0x80001400; // type:function size:0x20
gCarCount = .data:0x80300000; // type:object size:0x4
__ct__3FooFv = .text:0x80001500; // type:function size:0x20
Run__3FooFPi = .text:0x80001600; // type:function size:0x20
Step__3FooFv = .text:0x80001700; // type:function size:0x20
"""

    def test_maps_by_address_keeping_class_and_parameters(self):
        old, new = symbol_names(self.OLD_SYMS), symbol_names(self.NEW_SYMS)
        self.assertEqual(old["fn_80001234"], 0x80001234)
        got = forward_names(old, new, {"fn_80001234", "fn_80001300", "fn_80001400", "lbl_80300000", "fn_80001600",
                                       "fn_80001700", "nothing"})
        # a member that moved to another class keeps its call's spelling (same parameters): mapped;
        # a free function that became a member, or the reverse, is spelled differently: left alone
        self.assertEqual(got, {"fn_80001234": "ResetAll", "fn_80001300": "Tick", "fn_80001400": "Draw",
                               "lbl_80300000": "gCarCount"})

    def test_signature_part(self):
        self.assertEqual(signature_part("__Q22Ns9NsVehicleFi"), "Fi")
        self.assertEqual(signature_part("__3FooCFv"), "CFv")
        self.assertEqual(signature_part("__FP3Cari"), "FP3Cari")

    def test_rename_all_touches_members_too(self):
        code = "void Foo::Run() {\n    this->fn_80001300();\n    fn_80001234(lbl_80300000);\n}\n"
        out = rename_all(code, {"fn_80001300": "Tick", "fn_80001234": "ResetAll", "lbl_80300000": "gCarCount"})
        self.assertEqual(out, "void Foo::Run() {\n    this->Tick();\n    ResetAll(gCarCount);\n}\n")


class MineTest(unittest.TestCase):
    def test_class_from_rows(self):
        self.assertEqual(class_from_rows(["    4  mr r31, r3                arg  mr r30, r3", "  ..."]), "register")
        self.assertEqual(class_from_rows(["    4  addi r3, r3, 1            arg  addi r3, r3, 2"]), "shape")
        self.assertEqual(class_from_rows(["    4  lwz r3, 0x4(r3)           op  stw r3, 0x4(r3)"]), "shape")
        self.assertEqual(class_from_rows([]), "shape")

    def test_pick_spread_distinct_and_spread(self):
        items = [{"variant": f"int f() {{ return {i}; }}", "percent": float(i), "round": 0, "k": i} for i in range(10)]
        items.append({"variant": "int f() {  return 9; }", "percent": 9.0, "round": 1, "k": 0})     # same code again
        got = pick_spread(items, 3)
        self.assertEqual([a["percent"] for a in got], [9.0, 5.0, 0.0])
        self.assertEqual(len(pick_spread(items, 50)), 10)


class SynthTest(unittest.TestCase):
    def test_mutants_natural_and_capped(self):
        start = SRC.index("void Tick")
        span = (start, SRC.rindex("}") + 1)
        got = mutants(SRC, "Tick", span, True, 3, "Tick")
        self.assertTrue(got)
        self.assertLessEqual(len(got), 3)
        rules = {r for r, _l, _c in got}
        self.assertEqual(len(rules), len(got))                                   # one per rule
        for _r, _l, chunk in got:
            self.assertTrue(chunk.startswith("void Tick(Car* c, int n) {"))
            self.assertNotIn("__typeof__", chunk)
            self.assertNotEqual(chunk, SRC[span[0]:span[1]])
        many = mutants(SRC, "Tick", span, True, 20, "Tick")
        loop = [c for r, _l, c in many if r == "loop-form"]
        self.assertTrue(loop and "while (i < n) {" in loop[0] and "i++;" in loop[0] and "i = 0;" in loop[0])


class RuleShareTest(unittest.TestCase):
    def test_dominant_rule_capped_big_functions_kept(self):
        recs = [{"function": f"f{i}", "wrong": "w", "size": 0x40 + i, "provenance": {"rule": "statement-order"}} for i in range(20)]
        recs += [{"function": f"g{i}", "wrong": "w", "size": 0x40, "provenance": {"rule": r}} for i, r in
                 enumerate(["operand-swap"] * 3 + ["loop-form"] * 3)]
        out, dropped = cap_rule_share(recs, 0.4, caps={})
        so = [r for r in out if r["provenance"]["rule"] == "statement-order"]
        self.assertEqual(len(so), 4)                       # 0.4 * 6 / 0.6
        self.assertEqual(dropped, 16)
        self.assertEqual({r["function"] for r in so}, {"f19", "f18", "f17", "f16"})
        self.assertEqual(cap_rule_share(recs, 1.0, caps={}), (recs, 0))
        out, dropped = cap_rule_share(recs, 1.0)          # RULE_CAPS: statement order at most 20%
        self.assertEqual(sum(1 for r in out if r["provenance"]["rule"] == "statement-order"), 1)   # 0.2 * 6 / 0.8

    def test_synth_artifacts_of_existing_records(self):
        def r(rule, label, wrong="", exact=""):
            return {"provenance": {"rule": rule, "label": label}, "wrong": wrong, "exact": exact}
        self.assertTrue(synth_artifact(r("operand-swap", '"Me: " + text -> text + "Me: "')))
        self.assertTrue(synth_artifact(r("operand-swap", "(x - 3) / n / 12 * 12 -> 12 * (x - 3) / n / 12")))
        self.assertFalse(synth_artifact(r("operand-swap", 'f("a") != 0 -> 0 != f("a")')))
        self.assertFalse(synth_artifact(r("operand-swap", "a * b + c -> c + a * b")))
        self.assertTrue(synth_artifact(r("ternary-if", "x", "if (c) {\n  m = m;\n} else {\n  m = 0;\n}", "m = c ? m : 0;")))
        self.assertTrue(synth_artifact(r("ternary-if", "x", "return c ? 1 : 0;", "if (c) return 1;\nreturn 0;")))
        self.assertFalse(synth_artifact(r("ternary-if", "x", "a = c ? 1 : 0;", "a = c ? 1 : 0;")))      # the original's own
        self.assertTrue(synth_artifact(r("assign-in-condition", "(q = g(a)) in the condition", "if (h((q = g(a)), b)) {")))
        self.assertFalse(synth_artifact(r("assign-in-condition", "(q = g(a)) in the condition", "if ((q = g(a)) != 0) {")))
        calls = "void f() {\n    g(1);\n    g(2);\n    g(3);\n}"
        self.assertTrue(synth_artifact(r("statement-order", "swapped g(1); / g(2);", exact=calls)))
        self.assertFalse(synth_artifact(r("statement-order", "swapped g(1); / h(2);", exact=calls)))

    def test_backtrans_labels_and_whitespace(self):
        code = 'void f() {\n    g("a");   \n    \n    g("b");\n    g("a");\n}\n'
        self.assertEqual(literal_labels("g(&@412);\ng(&@413);\ng(&@412);", code), 'g("a");\ng("b");\ng("a");')
        self.assertEqual(literal_labels("g(&@412);", 'g("a"); g("b");'), "g(&@412);")       # counts differ: kept
        self.assertEqual(clean_answer(code), 'void f() {\n    g("a");\n\n    g("b");\n    g("a");\n}')


class RenameTest(unittest.TestCase):
    def test_new_name_natural(self):
        self.assertIn(new_name("i", {"j"}, "s"), set("kmn"))
        self.assertEqual(new_name("index", set(), "s"), "idx")
        self.assertEqual(new_name("carSpeed", {"car_speed"}, "s"), "vehicleSpeed")
        self.assertIsNone(new_name("zzq", set(), "s"))

    def test_renamed_skips_members_and_scopes(self):
        code = "void f(Car* c) { int speed = c->speed; obj.speed = speed; Speed::speed(); speed++; }"
        got = renamed(code, {"speed": "spd"})
        self.assertEqual(got, "void f(Car* c) { int spd = c->speed; obj.speed = spd; Speed::speed(); spd++; }")


class GradeTest(unittest.TestCase):
    def test_features(self):
        f = features(rec())
        self.assertEqual((f["insns"], f["calls"], f["vcalls"], f["branches"], f["loops"], f["floats"]), (7, 1, 1, 1, 1, 1))
        self.assertEqual(f["inlined"], 0)

    def test_difficulty_orders(self):
        small = rec(asm="    0  blr", wrong="", exact="void f() {}")
        far = rec(wrong_percent=20.0)
        near = rec(wrong_percent=99.0)
        self.assertLess(difficulty(small), difficulty(near))
        self.assertLess(difficulty(near), difficulty(far))
        self.assertGreater(difficulty(rec(diff_class="register")), difficulty(rec()))

    def test_levels_and_stages(self):
        lv = levels([float(i) for i in range(100)])
        self.assertEqual((lv[0], lv[50], lv[99]), (0, 1, 4))
        self.assertEqual(stage_of(1, 5), 1)
        moved = [stage_of(4, h) for h in range(0, 400000, 997)]
        self.assertTrue(any(s < 4 for s in moved) and moved.count(4) > len(moved) * 0.7)

    def test_caps(self):
        rs = [rec(source=s, wrong=f"w{i}") for i, s in enumerate(["synth", "base", "synth", "history", "regs"])]
        kept, dropped = cap_per_function(rs, 2)
        self.assertEqual(({r["source"] for r in kept}, dropped), ({"base", "history"}, 3))
        big = [dict(r, size=0x400) for r in rs]           # a big function keeps more
        self.assertEqual(cap_per_function(big, 2, 3)[1], 2)
        rs = [rec(source="base", function=f"b{i}") for i in range(35)] + [rec(source="synth", function=f"s{i}", wrong=f"w{i}")
                                                                        for i in range(100)]
        kept, dropped = cap_generated(rs, 0.65)
        gen = sum(1 for r in kept if r["source"] == "synth")
        self.assertEqual((gen, dropped), (65, 35))
        rs = [rec(source="base", function=f"b{i}") for i in range(35)] + \
             [rec(source="rename", function=f"n{i}", wrong="") for i in range(50)] + \
             [rec(source="attempts", function=f"a{i}", wrong=f"w{i}") for i in range(40)]
        kept, dropped = cap_generated(rs, 0.65)             # the model's attempts first, renamed copies last
        self.assertEqual(sum(1 for r in kept if r["source"] == "attempts"), 40)
        self.assertEqual(sum(1 for r in kept if r["source"] == "rename"), 25)


class ExamplesTest(unittest.TestCase):
    def make(self, zero):
        from tools.dk.platforms.gc_mwcc.retrieval import Example
        calls = []

        class FakeIndex:
            def top(self, asm, size, compiler, exclude, k):
                calls.append((size, set(exclude), k))
                return [Example("game/x/Other", "Other__3FooFv", 8, "GC/1.3.2", "    0  blr", "void Foo::Other() {}", set())]
        ex = Examples.__new__(Examples)
        ex.k, ex.zero, ex.memo, ex.comp, ex.index = 4, zero, {}, {"game/x/Foo": "GC/1.3.2"}, FakeIndex()
        return ex, calls

    def test_leave_one_out_and_memo(self):
        ex, calls = self.make(0.0)
        text, n = ex(rec())
        self.assertEqual(n, 1)
        self.assertIn("Other__3FooFv", text)
        ex(rec(wrong="another"))                         # the same function: one lookup
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"Bar__3FooFi"})   # never the record's own function
        self.assertEqual(calls[0][0], 7 * 4)             # size from the compact asm's instructions

    def test_zero_share(self):
        ex, _calls = self.make(0.2)
        got = [ex(rec(wrong=f"w{i}"))[1] for i in range(500)]
        self.assertTrue(0.12 < got.count(0) / 500 < 0.28)
        ex, _calls = self.make(1.0)
        self.assertEqual(ex(rec()), ("", 0))

    def test_render_puts_examples_first(self):
        from tools.dk.platforms.gc_mwcc.llmdata import render
        msgs = render(rec(), None, None, "## Similar functions of this game already matched\nEX")["messages"]
        self.assertIn("## Similar functions", msgs[1]["content"])
        self.assertLess(msgs[1]["content"].index("## Similar functions"), msgs[1]["content"].index("Bar"))


class RenderExportTest(unittest.TestCase):
    def test_render_direct_and_model_attempt(self):
        d = render(rec(wrong="", m2c="void Bar(Foo* this, int n) {}", source="base"))
        user = d["messages"][1]["content"]
        self.assertIn("Machine draft", user)
        self.assertIn("Write the definition", user)
        self.assertNotIn("Current definition", user)
        a = render(rec(source="attempts", first_round=False))
        self.assertIn("Best attempt so far (scores 97.50%)", a["messages"][1]["content"])
        self.assertIn("Current definition", render(rec())["messages"][1]["content"])

    def test_views_select_the_prompt_version(self):
        from tools.dk.platforms.gc_mwcc import llmtry
        from tools.dk.platforms.gc_mwcc.llmdata import builder
        for v in getattr(llmtry, "PROMPT_VERSIONS", (llmtry.PROMPT_VERSION,)):
            self.assertTrue(callable(builder(v)))
        with self.assertRaises(ValueError):
            builder(99)
        view = {"m2c": "void Bar(int n) { g(n->unk4); }", "decls": "struct Foo6;", "notes": "## Notes for this function\n- x"}
        d = render(rec(wrong="", source="base", m2c="old draft"), None, view)["messages"][1]["content"]
        self.assertIn("struct Foo6;", d)
        self.assertIn("n->unk4", d)
        self.assertIn("## Notes for this function", d)
        self.assertNotIn("old draft", d)
        r = render(rec(), None, view)["messages"][1]["content"]       # a repair takes the notes only
        self.assertIn("## Notes for this function", r)
        self.assertNotIn("struct Foo6;", r)

    def test_mix_drops_devices_judged_and_orders_stages(self):
        never = ({"game/bench/U"}, {"Held__3FooFv"}, {"BAD"})
        judge = lambda r: ["adds 1 volatile"] if "volatile" in r["exact"] else []
        rs = [rec(), rec(source="attempts", wrong="x", devices=["adds 1 pragma"]),
              rec(function="V__3FooFv", exact="void Foo::V() { volatile int x; }"),
              rec(source="synth", wrong="y", mutant_shape="BAD"),
              rec(source="base", wrong="", function="B__3FooFv", exact="void Foo::B() {}")]
        out, card = mix(rs, [], never, [], 10 ** 6, 2.2, False, None, judge, 8, 1.0)
        self.assertEqual(card["records"], 2)
        self.assertEqual(card["dropped"]["device attempt"], 1)
        self.assertEqual(card["dropped"]["judge"], 1)
        self.assertEqual(card["leakage"], {"mutant asm shape": 1})
        self.assertEqual([r["tags"]["stage"] for r in out], sorted(r["tags"]["stage"] for r in out))
        self.assertTrue(all(r["tags"]["difficulty"] in LEVELS for r in out))

    def test_dpo_device_attempt_first(self):
        rs = [rec(source="base", wrong=""), rec(source="history", wrong="h", wrong_percent=90.0),
              rec(source="attempts", wrong="d", wrong_percent=100.0, devices=["adds 1 pragma"]),
              rec(source="attempts", wrong="a", wrong_percent=95.0)]
        got = dpo_pairs(rs, None, 2)
        self.assertEqual([g["rejected"][0]["content"] for g in got], ["```cpp\nd\n```", "```cpp\na\n```"])
        self.assertEqual(got[0]["tags"]["rank_rejected"], [0, 0, 100.0])
        self.assertIn("Write the definition", got[0]["prompt"][1]["content"])     # the unwritten-function prompt

    def test_answer_judge(self):
        j = AnswerJudge.__new__(AnswerJudge)
        j.cfg, j.forbidden, j.types, j.memo = None, [], {"u": "struct Foo;\n"}, {}
        ok = rec(unit="u", function="Bar__3FooFi", exact="void Foo::Bar(int n) {\n    g(n + 1);\n}")
        self.assertEqual(j(ok), [])
        off = rec(unit="u", function="Bar__3FooFi", exact="void Foo::Bar(int n) {\n    *(int*)((char*)this + 0x10) = n;\n}")
        self.assertTrue(j(off))
        vol = rec(unit="u", function="Bar__3FooFi", exact="void Foo::Bar(int n) {\n    volatile int x = n;\n    g(x);\n}")
        self.assertTrue(j(vol))


class QuarantineSizeTest(unittest.TestCase):
    def test_quarantined_formats(self):
        import json
        import tempfile
        from pathlib import Path
        from tools.dk.platforms.gc_mwcc.llmdata import quarantined
        self.assertIsNone(quarantined(Path("/nonexistent/judge-audit.json")))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text(json.dumps({"made": "x", "functions": [{"unit": "u", "function": "A__3FooFv", "flags": ["f"]}],
                                     "flagged": ["B__3FooFv"], "notes": ["not a function"]}))
            self.assertEqual(quarantined(p), {"A__3FooFv", "B__3FooFv"})

    def test_quarantine_drops_every_source(self):
        never = (set(), set(), set())
        rs = [rec(source=s) for s in ("base", "history", "synth")] + [rec(function="Ok__3FooFv")]
        _out, card = mix(rs, [], never, [], 10 ** 6, 2.2, False, quarantine={"Bar__3FooFi"})
        self.assertEqual((card["records"], card["dropped"]["quarantined"]), (1, 3))
        self.assertEqual(sum(card["by_size"].values()), 1)

    def test_bands_and_big_kept_first(self):
        from tools.dk.platforms.gc_mwcc.llmdata import band
        self.assertEqual([band(x) for x in (0xFC, 0x100, 0x2FF, 0x300, 0x800)],
                         ["<0x100", "0x100-0x300", "0x100-0x300", "0x300-0x800", ">0x800"])
        rs = [rec(source="base", function=f"b{i}") for i in range(100)]
        rs += [rec(source="synth", function=f"s{i}", wrong=f"w{i}", size=0x40 if i % 2 else 0x900) for i in range(400)]
        kept, _ = cap_generated(rs, 0.5)
        big = sum(1 for r in kept if r["source"] == "synth" and r["size"] == 0x900)
        self.assertGreater(big, 60)                                             # of 100 kept, most are big

    def test_answer_flags(self):
        from tools.dk.platforms.gc_mwcc.llmdata import answer_flags
        flags = ["new local n only copies mCount", "odd shape: adds 1 goto", "redundant cast to its own type (int)x",
                 "adds a comma operator"]
        self.assertEqual(answer_flags(flags, True), ["adds a comma operator", "redundant cast to its own type (int)x"])
        self.assertIn("new local n only copies mCount", answer_flags(flags, False))


class BacktransTest(unittest.TestCase):
    def test_new_symbols(self):
        e = lambda k="fn": SimpleNamespace(kind=k)
        before = {"a": e(), "@1": e("obj")}
        after = {"a": e(), "b__FP3Car": e(), "@2": e(), "__sinit_x": e(), "lbl": e("obj")}
        self.assertEqual(new_symbols(before, after), ["b__FP3Car"])

    def test_fix_feedback(self):
        msg = bt_fix("line 12: x = m8C; | undefined identifier 'm8C'", 10)
        self.assertIn("line 10", msg)
        self.assertIn("undefined identifier 'm8C'", msg)
        self.assertIn("free function", msg)


class RepairBenchTest(unittest.TestCase):
    def test_summary_counts_repairs_apart(self):
        from tools.dk.platforms.gc_mwcc.llmtry import summarize
        rows = [{"kind": "run", "model": "m"},
                {"kind": "function", "function": "a", "bench_kind": "repair", "rid": "a#0", "start": 90.0, "best": 100.0, "exact": True},
                {"kind": "function", "function": "a", "bench_kind": "repair", "rid": "a#1", "start": 80.0, "best": 90.0, "exact": False},
                {"kind": "function", "function": "b", "bench_kind": "near", "start": 95.0, "best": 95.0, "exact": False}]
        s = summarize(rows)
        self.assertEqual(s["exact_repair"], "1/2")
        self.assertEqual(s["repair_gain"], 10.0)
        self.assertEqual(s["exact_near"], "0/1")


if __name__ == "__main__":
    unittest.main()
