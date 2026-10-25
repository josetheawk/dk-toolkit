import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.asmindex import AsmEntry
from tools.dk.platforms.gc_mwcc.dtk_symbols import Symbol
from tools.dk.platforms.gc_mwcc.propose import propose


def sym(name, section, address, size, type_="object"):
    return Symbol(name, section, address, {"type": type_, "size": f"{size:X}"})


def entry(name, kind, refs=(), calls=()):
    return AsmEntry(name, kind, Path("x.s"), "", list(refs), list(calls))


class ProposeTest(unittest.TestCase):
    def test_vtable_rtti_chain_weak_base_and_externs(self):
        symbols = [
            sym("f1", ".text", 0x1000, 0x10, "function"),
            sym("f2", ".text", 0x1010, 0x10, "function"),
            sym("f3", ".text", 0x1020, 0x10, "function"),       # another unit
            sym("name", ".data", 0x2000, 0x8),                  # class-name string
            sym("bases", ".data", 0x2008, 0xC),                 # base list -> base RTTI copy
            sym("vt", ".data", 0x2014, 0x10),                   # this unit's vtable
            sym("vt_base", ".data", 0x2024, 0x14),              # weak base vtable, also used by f3
            sym("other", ".data", 0x2038, 0x8),                 # next unit's data
            sym("rtti_base", ".sdata", 0x3000, 0x8),            # local copy, only reached via base list / weak vt
            sym("rtti", ".sdata", 0x3008, 0x8),
            sym("c", ".sdata2", 0x4000, 0x4),                   # float constant
            sym("g", ".sbss", 0x5000, 0x4),                     # global used by two units
        ]
        entries = {
            "f1": entry("f1", "fn", refs=["vt", "vt_base"]),
            "f2": entry("f2", "fn", refs=["c", "g"], calls=["f1", "ext_fn"]),
            "f3": entry("f3", "fn", refs=["vt_base", "g", "other"]),
            "vt": entry("vt", "obj", refs=["rtti", "f1"]),
            "vt_base": entry("vt_base", "obj", refs=["rtti_base"]),
            "rtti": entry("rtti", "obj", refs=["name", "bases"]),
            "bases": entry("bases", "obj", refs=["rtti_base"]),
            "name": entry("name", "obj"),
            "rtti_base": entry("rtti_base", "obj"),
            "c": entry("c", "obj"),
            "g": entry("g", "obj"),
            "other": entry("other", "obj"),
        }
        p = propose(symbols, entries, 0x1000, 0x1020)
        self.assertEqual(p.functions, ["f1", "f2"])
        self.assertEqual(p.sections[".data"], (0x2000, 0x2038))      # through the weak base vtable, not "other"
        self.assertEqual(p.sections[".sdata"], (0x3000, 0x3010))     # base RTTI copy pulled in via the weak vtable
        self.assertEqual(p.sections[".sdata2"], (0x4000, 0x4004))
        self.assertNotIn(".sbss", p.sections)
        self.assertEqual(p.extern_data, ["g"])
        self.assertEqual(p.extern_calls, ["ext_fn"])
        p = propose(symbols, entries, 0x1000, 0x1020, {0x1010: 0x6008, 0x1020: 0x600C})
        self.assertEqual(p.sections[".ctors"], (0x6008, 0x600C))     # f2's unnamed .ctors slot


if __name__ == "__main__":
    unittest.main()


class TuBoundsTest(unittest.TestCase):
    def test_tail_after_sinit_moves(self):
        from pathlib import Path as P
        from tools.dk.platforms.gc_mwcc.propose import adjust_tu_bounds
        def fn(name, addr):
            return Symbol(name, ".text", addr, {"type": "function", "size": "10"})
        def ent(name, file, calls=(), refs=(), text=""):
            return AsmEntry(name, "fn", P(file + ".s"), text, list(refs), list(calls))
        prev_body, prev_sinit = fn("p", 0x00), fn("__sinit_p_cpp", 0x10)
        thunk, a, b = fn("t", 0x20), fn("a", 0x30), fn("b", 0x40)
        own_sinit, tail, nxt = fn("__sinit_a_cpp", 0x50), fn("v", 0x60), fn("n", 0x70)
        thunk_text = "/* 80000020 */\taddi r3, r3, -0x8\n/* 80000024 */\tb p\n"
        entries = {
            "p": ent("p", "auto_03_p"), "__sinit_p_cpp": ent("__sinit_p_cpp", "auto_fn_p"),
            "t": ent("t", "auto_03_a", calls=["p"], text=thunk_text),
            "a": ent("a", "auto_03_a", refs=["vt_a"]), "b": ent("b", "auto_03_a"),
            "__sinit_a_cpp": ent("__sinit_a_cpp", "auto_fn_a", refs=["vt_a"]),
            "v": ent("v", "auto_03_n"), "n": ent("n", "auto_03_n"),
            "vt_a": AsmEntry("vt_a", "obj", P("x.s"), "", ["v"], []),          # vtable of this TU -> v
        }
        fns, out, inn = adjust_tu_bounds([thunk, a, b], [prev_body, prev_sinit], [own_sinit, tail, nxt], entries)
        self.assertEqual(out, ["t"])                         # thunk into the previous TU, right after its __sinit
        self.assertEqual(inn, ["__sinit_a_cpp", "v"])        # own initializer, then the inline virtual it points at
        self.assertEqual([f.name for f in fns], ["a", "b", "__sinit_a_cpp", "v"])

    def test_header_static_sinit_moves_in(self):
        from pathlib import Path as P
        from tools.dk.platforms.gc_mwcc.propose import adjust_tu_bounds
        f = lambda n, a: Symbol(n, ".text", a, {"type": "function", "size": "10"})
        e = lambda n, file, refs=(), calls=(): AsmEntry(n, "fn", P(file + ".s"), "", list(refs), list(calls))
        entries = {"a": e("a", "auto_03_a"), "s": e("s", "auto_fn_s", refs=["static"]),
                   "n": e("n", "auto_03_n"), "static": AsmEntry("static", "obj", P("x.s"), "", [], []),
                   "ctor": AsmEntry("ctor", "obj", P("x.s"), "", ["s"], [])}
        fns, _, inn = adjust_tu_bounds([f("a", 0)], [], [f("s", 0x10), f("n", 0x20)], entries)
        self.assertEqual(inn, ["s"])                          # touches only its own static: this TU's
        entries["n"] = e("n", "auto_03_n", refs=["static"])
        fns, _, inn = adjust_tu_bounds([f("a", 0)], [], [f("s", 0x10), f("n", 0x20)], entries)
        self.assertEqual(inn, [])                             # the static is shared with the next TU

    def test_no_move_without_sinit(self):
        from pathlib import Path as P
        from tools.dk.platforms.gc_mwcc.propose import adjust_tu_bounds
        f = lambda n, a: Symbol(n, ".text", a, {"type": "function", "size": "10"})
        entries = {n: AsmEntry(n, "fn", P("auto_03_x.s"), "", [], ["a"] if n == "z" else []) for n in "abz"}
        fns, out, inn = adjust_tu_bounds([f("a", 0), f("b", 0x10)], [], [f("z", 0x20)], entries)
        self.assertEqual((out, inn), ([], []))               # next unit has no __sinit first: nothing moves
