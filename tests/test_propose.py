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

    def test_declared_neighbours_bound_every_range(self):
        from tools.dk.platforms.gc_mwcc.dtk_symbols import Unit
        symbols = [
            sym("a", ".text", 0x0F00, 0x10, "function"),        # declared unit before us
            sym("f", ".text", 0x1000, 0x10, "function"),        # us
            sym("b", ".text", 0x1020, 0x10, "function"),        # declared unit after us
            sym("d0", ".data", 0x2000, 0x10),
            sym("d1", ".data", 0x2010, 0x10),                   # inside the next unit's range
            sym("d2", ".data", 0x2030, 0x8),                    # a stray beyond it
            sym("z0", ".bss", 0x4FF0, 0x8),                     # before the previous unit's .bss
            sym("z1", ".bss", 0x5008, 0x8),
        ]
        entries = {"f": entry("f", "fn", refs=["d0", "d1", "d2", "z0", "z1"]), "a": entry("a", "fn"),
                   "b": entry("b", "fn")} | {n: entry(n, "obj") for n in ("d0", "d1", "d2", "z0", "z1")}
        units = [Unit("A.cpp", [(".text", 0x0F00, 0x0F10), (".bss", 0x5000, 0x5008)]),
                 Unit("B.cpp", [(".text", 0x1020, 0x1030), (".data", 0x2010, 0x2030)]),
                 Unit("us.cpp", [(".text", 0x1000, 0x1010), (".data", 0x1F00, 0x3000)])]   # ignored: it is us
        p = propose(symbols, entries, 0x1000, 0x1010, units=units)
        self.assertEqual(p.sections[".data"], (0x2000, 0x2010))      # clipped at the next unit
        self.assertEqual(p.sections[".bss"], (0x5008, 0x5010))       # never before a unit that precedes us
        self.assertEqual(p.extern_data, ["d1", "d2", "z0"])
        self.assertTrue(any("d1" in n for n in p.notes))

    def test_own_vtable_rtti_and_strings_even_when_another_unit_uses_the_vtable(self):
        symbols = [
            sym("f", ".text", 0x1000, 0x10, "function"),        # a virtual of our class
            sym("ctor", ".text", 0x1010, 0x10, "function"),     # another unit's inlined ctor
            sym("str", ".data", 0x2000, 0x8),                   # our function's string
            sym("name", ".data", 0x2008, 0x8),                  # class name string
            sym("bases", ".data", 0x2010, 0x8),
            sym("vt", ".data", 0x2018, 0x10),
            sym("next", ".data", 0x2028, 0x8),                  # the next unit's
            sym("rtti", ".sdata", 0x3000, 0x8),
        ]
        entries = {"f": entry("f", "fn", refs=["str"]), "ctor": entry("ctor", "fn", refs=["vt", "next"]),
                   "vt": entry("vt", "obj", refs=["rtti", "f"]), "rtti": entry("rtti", "obj", refs=["name", "bases"]),
                   "str": entry("str", "obj"), "name": entry("name", "obj"), "bases": entry("bases", "obj"),
                   "next": entry("next", "obj")}
        p = propose(symbols, entries, 0x1000, 0x1010)
        self.assertEqual(p.sections[".data"], (0x2000, 0x2028))
        self.assertEqual(p.sections[".sdata"], (0x3000, 0x3008))

    def test_range_end_padding(self):
        symbols = [sym("f", ".text", 0x1000, 0x10, "function"), sym("g", ".text", 0x1010, 0x10, "function"),
                   sym("s", ".data", 0x2000, 0x13),             # unaligned end (a string)
                   sym("n", ".data", 0x2018, 0x8),
                   sym("w", ".sdata", 0x3000, 0x8),             # aligned end
                   sym("m", ".sdata", 0x3010, 0x8)]
        entries = {"f": entry("f", "fn", refs=["s", "w"]), "g": entry("g", "fn", refs=["n", "m"]),
                   "s": entry("s", "obj"), "n": entry("n", "obj"), "w": entry("w", "obj"), "m": entry("m", "obj")}
        p = propose(symbols, entries, 0x1000, 0x1010, read=lambda a, n: bytes(n))
        self.assertEqual(p.sections[".data"], (0x2000, 0x2018))      # a gap_ after 0x2013 must stay inside
        self.assertEqual(p.sections[".sdata"], (0x3000, 0x3008))     # the padding stays a hole

    def test_odd_start_takes_the_word(self):
        symbols = [sym("e", ".text", 0x0F00, 0x10, "function"), sym("f", ".text", 0x1000, 0x10, "function"),
                   sym("shared", ".sbss", 0x4000, 0x1), sym("mine", ".sbss", 0x4001, 0x1)]
        entries = {"e": entry("e", "fn", refs=["shared"]), "f": entry("f", "fn", refs=["shared", "mine"]),
                   "shared": entry("shared", "obj"), "mine": entry("mine", "obj")}
        p = propose(symbols, entries, 0x1000, 0x1010)
        self.assertEqual(p.sections[".sbss"], (0x4000, 0x4004))

    def test_bss_an_initializer_reaches_at_an_offset_and_strays(self):
        symbols = [sym("__sinit_x_cpp", ".text", 0x1000, 0x10, "function"),
                   sym("g", ".text", 0x1010, 0x10, "function"),
                   sym("base", ".bss", 0x5000, 0x8),            # constructed here, used by others too
                   sym("o1", ".bss", 0x5008, 0x8),              # reached at base+8: no reference
                   sym("own", ".bss", 0x5010, 0x8),
                   sym("x1", ".bss", 0x5018, 0x4), sym("x2", ".bss", 0x501C, 0x4), sym("x3", ".bss", 0x5020, 0x4),
                   sym("far", ".bss", 0x5024, 0x4)]             # after three of another unit's objects
        entries = {"__sinit_x_cpp": entry("__sinit_x_cpp", "fn", refs=["base", "own", "far"]),
                   "g": entry("g", "fn", refs=["base", "x1", "x2", "x3"])} | \
                  {n: entry(n, "obj") for n in ("base", "o1", "own", "x1", "x2", "x3", "far")}
        p = propose(symbols, entries, 0x1000, 0x1010)
        self.assertEqual(p.sections[".bss"], (0x5000, 0x5018))
        self.assertIn("far", p.extern_data)

    def test_leading_weak_copy_an_earlier_unit_reaches_is_its(self):
        symbols = [sym("e", ".text", 0x0F00, 0x10, "function"),     # earlier unit
                   sym("f", ".text", 0x1000, 0x10, "function"),
                   sym("vt_e", ".data", 0x1F00, 0x8),
                   sym("vt", ".data", 0x2000, 0x8),
                   sym("rtti_base", ".sdata", 0x3000, 0x8),          # weak copy, kept in the earlier unit
                   sym("rtti", ".sdata", 0x3008, 0x8)]
        entries = {"e": entry("e", "fn"), "f": entry("f", "fn"),
                   "vt_e": entry("vt_e", "obj", refs=["e", "rtti_base"]),
                   "vt": entry("vt", "obj", refs=["rtti", "f", "rtti_base"]),
                   "rtti": entry("rtti", "obj"), "rtti_base": entry("rtti_base", "obj")}
        p = propose(symbols, entries, 0x1000, 0x1010)
        self.assertEqual(p.sections[".sdata"], (0x3008, 0x3010))


class AsmIndexTest(unittest.TestCase):
    def test_quoted_names(self):
        import tempfile
        from tools.dk.platforms.gc_mwcc.asmindex import index
        text = (".fn f, global\n/* 0 */\tlis r3, \"@73\"@ha\n/* 4 */\taddi r3, r3, \"@73\"@l\n"
                "/* 8 */\tbl \"g<1>__Fv\"\n.endfn f\n.obj \"@73\", local\n\t.4byte \"@etb_80000000\"\n.endobj \"@73\"\n")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.s"
            path.write_text(text)
            entries = index([path])
        self.assertEqual(entries["f"].refs, ["@73"])
        self.assertEqual(entries["f"].calls, ["g<1>__Fv"])
        self.assertEqual(entries["@73"].refs, ["@etb_80000000"])


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


class LoneInitializerTest(unittest.TestCase):
    def test_lone_initializer_joins_the_tu_before(self):
        from pathlib import Path as P
        from tools.dk.platforms.gc_mwcc.propose import lone_initializer
        f = lambda n, a: Symbol(n, ".text", a, {"type": "function", "size": "10"})
        e = lambda n, file: AsmEntry(n, "fn", P(file + ".s"), "", [], [])
        entries = {"a": e("a", "game/unk/A"), "__sinit_x_cpp": e("__sinit_x_cpp", "game/unk/X"),
                   "__sinit_a_cpp": e("__sinit_a_cpp", "auto_fn_a"), "b": e("b", "game/unk/B")}
        self.assertTrue(lone_initializer([f("__sinit_x_cpp", 0x10)], [f("a", 0)], entries))
        self.assertFalse(lone_initializer([f("__sinit_x_cpp", 0x10)], [f("a", 0), f("__sinit_a_cpp", 8)], entries))
        self.assertFalse(lone_initializer([f("b", 0x10)], [f("a", 0)], entries))
