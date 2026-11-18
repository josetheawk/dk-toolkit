import struct
import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import regpredict as rp
from tools.dk.platforms.gc_mwcc import regtruth as rt
from tools.dk.platforms.gc_mwcc.regs import Var, _slot, _type_size, parse_dwarf1

SRC = """
int D0(); int D1(); void U(int);
int h1a() { int x, y; y = D0(); x = D1(); U(0); U(x); U(y); return 0; }
int h1b() { int y, x; y = D0(); x = D1(); U(0); U(x); U(y); return 0; }
int h2b() { int x = D0(); int y = D1(); U(0); return x + y; }
int h2a() { int x = D0(); int y = D1(); U(0); U(x); U(y); return 0; }
int h3a(int a, int b) { int x = D0(); U(0); U(a); U(b); U(x); return 0; }
"""


def model(name: str, variables: list[tuple[str, str]], banks=None, unnamed=None):
    span = next(f for f in cbody.functions(SRC) if f.name == name)
    body = rp.Body(SRC, span.open, cbody.type_names(SRC))
    vals = rp.values(body, variables)
    bank = {n: "r" for n, _ in variables}
    a = rp.rank_rule(vals, bank)
    b, note = rp.behaviour_rule(body, vals, bank, unnamed)
    return a, b, note


class PredictorTests(unittest.TestCase):
    def test_declaration_order_decides(self):
        a, b, _ = model("h1a", [("x", "local"), ("y", "local")])
        self.assertEqual((a["x"], a["y"]), ("saved:r31", "saved:r30"))
        a, b, _ = model("h1b", [("y", "local"), ("x", "local")])
        self.assertEqual((b["y"], b["x"]), ("saved:r31", "saved:r30"))

    def test_local_becomes_temp_and_ranks_first(self):
        a, b, _ = model("h2b", [("x", "local"), ("y", "local")])
        self.assertEqual((b["x"], b["y"]), ("temp", "temp"))
        self.assertEqual(a["x"], "saved:r31")                 # the rank rule has no temporaries
        a, b, _ = model("h2a", [("x", "local"), ("y", "local")])
        self.assertEqual((b["x"], b["y"]), ("saved:r31", "saved:r30"))   # bare uses beyond arg0 of the next call

    def test_params_rank_last_first(self):
        a, b, _ = model("h3a", [("a", "param"), ("b", "param"), ("x", "local")])
        self.assertEqual((b["x"], b["b"], b["a"]), ("saved:r31", "saved:r30", "saved:r29"))

    def test_unexplained_temporaries_leave_registers_unpredicted(self):
        a, b, note = model("h3a", [("a", "param"), ("b", "param"), ("x", "local")], unnamed=2)
        self.assertTrue(all(p is None for p in b.values()))
        self.assertIn("temporaries", note)
        self.assertEqual(a["x"], "saved:r31")

    def test_a_call_in_the_other_branch_or_before_a_return_crosses_nothing(self):
        src = ("void U(int); void V(int);\n"
               "void f(int p, int q) { if (q) { V(0); } else { U(p); } }\n"
               "void g(int p, int q) { if (q) { V(0); return; } U(p); }\n"
               "void h(int p, int q) { while (q) { if (q > 1) { V(0); break; } q--; } U(p); }\n"
               "void k(int p, int q) { if (q) { U(p); } V(0); U(p); }\n")
        for name, cross in (("f", False), ("g", False), ("h", True), ("k", True)):
            span = next(s for s in cbody.functions(src) if s.name == name)
            body = rp.Body(src, span.open, set())
            p = rp.values(body, [("p", "param"), ("q", "param")])[0]
            self.assertEqual(p.cross, cross, name)

    def test_member_writes_use_this_and_call_free_inlines_are_no_calls(self):
        src = ("struct S { int m; void f(); void g(int); };\nvoid F(int); inline int Get(int x) { return x; }\n"
               "void S::f() { F(m); m = 0; }\nvoid S::g(int p) { Get(p); F(0); }\n")
        spans = cbody.functions(src)
        body = rp.Body(src, spans[-2].open, set())
        self.assertGreater(rp.implicit_this_end(body, {"this"}), body.calls[0].end)
        body = rp.Body(src, spans[-1].open, set())
        self.assertEqual(len(rp.real_calls(body, {"Get"})), 1)
        p = rp.values(body, [("p", "param")], {"Get"})[0]
        self.assertFalse(p.cross)

    def test_leaf_inline_names(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "include").mkdir()
            (root / "include" / "a.h").write_text("inline int Get(int x) { if (x) return x; return 0; }\n"
                                                  "inline void Put(int x) { F(x); }\n")
            cfg = type("Cfg", (), {"root": root})()
            self.assertEqual(rt.leaf_inline_names(cfg), {"Get"})

    def test_calls_the_object_never_branches_to_are_inlined(self):
        self.assertEqual(rt.base_name("fn_IsEmpty__6TxtStrCFv"), "fn_IsEmpty")
        self.assertEqual(rt.base_name("__ct__6TxtStrFv"), "__ct")
        self.assertEqual(rt.base_name("TRK_memcpy"), "TRK_memcpy")
        reloc = type("R", (), {})
        r1, r2 = reloc(), reloc()
        r1.type, r1.symbol, r1.addend, r1.target_section, r1.target_value = 10, "Out__Fi", 0, "", 0
        r2.type, r2.symbol, r2.addend, r2.target_section, r2.target_value = 10, ".text", 0x40, ".text", 0
        fn = type("F", (), {"relocs": [r1, r2]})()
        self.assertEqual(rt.branch_targets(fn, {(".text", 0x40): "Local__Fv"}), {"Out", "Local"})
        src = ("void Out(int); int Near(int x) { return x + 1; } void Wrap(int x) { Out(x); }\n"
               "struct S { virtual int V(); };\n"
               "void f(S* s, int p) { Near(p); Wrap(p); s->V(); Out(p); }\n")
        spans = cbody.functions(src)
        body = rp.Body(src, spans[-1].open, set())
        bodies = {s.name: src[s.open + 1:s.close] for s in spans}
        known = {"Near", "Wrap", "V"}
        # Near and Wrap copied in; Wrap's own call stays one; V is virtual (no bl either)
        self.assertEqual(rt.inlined_leaf_calls(body, {"Out"}, bodies, known, {"V"}), {"Near"})
        self.assertEqual(rt.inlined_leaf_calls(body, {"Out", "Near"}, bodies, known, {"V"}), set())

    def test_constructor_and_destructor_this(self):
        src = "struct S { int m; S(); ~S(); };\nS::S() { m = 0; }\nS::~S() { m = 1; }\n"
        spans = cbody.functions(src)
        for span, sym, want in ((spans[0], "__ct__1SFv", "saved:r31"), (spans[1], "__dt__1SFv", "saved:r30")):
            body = rp.Body(src, span.open, set())
            vals = rp.values(body, [("this", "param")])
            rp.special_members(sym, vals, span.close)
            self.assertEqual(rp.rank_rule(vals, {})["this"], want)

    def test_sizes_and_stack_order(self):
        src = "void f() { char c; int a; double d; short sh; char buf[0x10]; }"
        body = rp.Body(src, cbody.functions(src)[0].open, set())
        idx = {n: i for i, n in enumerate(["c", "a", "d", "sh", "buf"])}
        self.assertEqual(rp.stack_order(body, list(idx), idx), ["c", "sh", "a", "d", "buf"])
        self.assertIsNone(rp.stack_order(body, ["c", "zz"], {"c": 0, "zz": 1}))


class TruthTests(unittest.TestCase):
    def test_saved_registers_read_the_prologue(self):
        stmw = (47 << 26) | (27 << 21) | (1 << 16) | 0x14       # stmw r27, 0x14(r1)
        stfd = (54 << 26) | (31 << 21) | (1 << 16) | 0x30
        bl = (18 << 26) | 1
        code = struct.pack(">4I", stmw, stfd, bl, (36 << 26) | (30 << 21) | (1 << 16))
        got = rt.saved_registers(code)
        self.assertEqual((got["r"], got["f"]), (set(range(27, 32)), {31}))

    def test_stack_location_and_type_size_from_debug_info(self):
        self.assertEqual(_slot(bytes([2, 0, 0, 0, 1, 4, 0, 0, 0, 0x10, 7])), 16)
        self.assertIsNone(_slot(bytes([1, 0, 0, 0, 31])))
        types = {100: (0x13, {0xB: 12}), 200: (0x16, {7: 100})}
        self.assertEqual(_type_size({7: 200}, types), 12)
        self.assertEqual(_type_size({5: 14}, types), 4)
        self.assertEqual(_type_size({8: bytes([3, 1, 0, 0, 0, 0])}, types), 4)

    def test_classify_and_tally(self):
        fn = {"features": {"inline_call": True, "loop_index": False, "repeated_load": False, "struct_local": False},
              "unexplained_temps": 1}
        row = {"name": "p", "kind": "param", "cat": "reg", "actual": "saved:r29", "a": "saved:r30", "b": None}
        self.assertEqual(rt.classify(row, fn), "inline-param-temp")
        self.assertEqual(rt.classify({**row, "name": "this"}, fn), "this-high")
        self.assertEqual(rt.classify({**row, "cat": "stack"}, fn), "stack-slot-order")
        fn["unexplained_temps"] = 0
        self.assertEqual(rt.classify(row, fn), "param-merged-temp")
        row.update(a_ok=False, b_ok=None, scored=True, cls="param-merged-temp")
        ok = {"name": "q", "kind": "local", "cat": "reg", "actual": "saved:r31", "a": "saved:r31", "b": "saved:r31",
              "a_ok": True, "b_ok": True, "scored": True, "cls": "ok"}
        doc = rt.tally([{"functions": [{"function": "f", "address": 0x80000000, "vars": [row, ok]}]}])
        self.assertEqual(doc["predictors"]["a"], {"n": 2, "ok": 1, "unpredicted": 0})
        self.assertEqual(doc["predictors"]["b"], {"n": 2, "ok": 1, "unpredicted": 1})
        self.assertEqual(doc["classes"]["param-merged-temp"]["examples"][0]["address"], 0x80000000)
        meta = {"date": "x", "units_run": 1, "units_ok": 1, "mode": "m", "skipped_units": {}, "skipped_functions": {},
                "seconds": 1.0, "jobs": 1}
        self.assertIn("param-merged-temp", rt.render({"meta": meta, "totals": doc}))

    def test_stratified_sample_is_seeded(self):
        units = [{"name": f"main/{d}/{d}/u{i}"} for d in "abc" for i in range(5)]
        a, b = rt.stratified(units, 6, 1), rt.stratified(units, 6, 1)
        self.assertEqual(a, b)
        self.assertEqual(len({u["name"].split("/")[1] for u in a}), 3)


if __name__ == "__main__":
    unittest.main()
