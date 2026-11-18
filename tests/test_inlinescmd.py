"""dk inlines: the command's pool, decoding and reports on synthetic functions (no game data)."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.test_asmflow import DTOR, asm
from tests.test_inlines import GIANT, PART_A, PART_B, renamed
from tools.dk.platforms.gc_mwcc import asmindex, inlines, inlinescmd


def entry(name: str, lines: str, file: str) -> asmindex.AsmEntry:
    text = asm(lines).replace(".fn f,", f".fn {name},").replace(".endfn f", f".endfn {name}")
    return asmindex.AsmEntry(name, "fn", Path(file), text)


def context() -> inlinescmd.Context:
    funcs = {
        "__dt__3FooFv": (DTOR, "a.s"),
        "fn_80401000": (DTOR, "b.s"),                       # identical code under an auto name
        "fn_giant": (GIANT + PART_A + PART_B, "g.s"),
        "fn_other": (renamed(PART_A + PART_B), "o.s"),
    }
    entries = {n: entry(n, text, f) for n, (text, f) in funcs.items()}
    return inlinescmd.Context(cfg=None, entries=entries,
                              unit_of_file={Path("a.s"): "game/Foo.cpp", Path("g.s"): "game/Big.cpp"},
                              exact={"__dt__3FooFv"}, sizes={"fn_giant": 0x900, "fn_other": 0x800},
                              canon=lambda v: v)


class DecodeTest(unittest.TestCase):
    def test_pool_keeps_identical_code_once_exact_first(self):
        templates, helpers = inlinescmd.build_pool(context(), {"fn_giant"}, inlines.MIN_WEIGHT, inlines.MAX_WEIGHT)
        self.assertIn("__dt__3FooFv", templates)
        self.assertNotIn("fn_80401000", templates)
        self.assertEqual(helpers["__dt__3FooFv"].same, ["fn_80401000"])

    def test_decode_splits_copies_repeats_and_frame(self):
        ctx = context()
        self.assertEqual(inlinescmd.giants(ctx), ["fn_giant", "fn_other"])     # the repeated-block corpus
        opts = {"min_weight": inlines.MIN_WEIGHT, "max_weight": inlines.MAX_WEIGHT, "min_conf": inlines.MIN_CONF,
                "min_block": inlines.KGRAM}
        r, = inlinescmd.decode_all(ctx, ["fn_giant"], opts, jobs=1)
        self.assertEqual((r["unit"], r["frame"]), ("game/Big.cpp", 9))
        h, = r["helpers"]
        self.assertEqual((h["helper"], h["pretty"], h["exact"], h["unit"]), ("__dt__3FooFv", "Foo::~Foo()", True,
                                                                                "game/Foo.cpp"))
        self.assertEqual([c["call"] for c in h["copies"]], ["Foo::~Foo(this=sp+0x20)", "Foo::~Foo(this=sp+0x34)"])
        self.assertEqual(r["copies_exact"], h["insns"])
        self.assertEqual(r["copies_open"], 0)
        self.assertEqual(r["repeated"], 20)                    # PART_A + PART_B, shared with fn_other
        block, = r["blocks"]
        self.assertEqual(block["partners"][0][0], "fn_other")
        text = "\n".join(inlinescmd.render_one(r, 5))
        self.assertIn("Foo::~Foo()  [__dt__3FooFv, src/game/Foo.cpp, exact]  2x", text)
        self.assertIn("same code: fn_80401000", text)
        table = "\n".join(inlinescmd.summary([r]))
        self.assertIn("| fn_giant | game/Big.cpp | 2304 |", table)
        self.assertIn("| Foo::~Foo() | __dt__3FooFv | game/Foo.cpp | exact | 2 | 1 |", table)


class CallTextTest(unittest.TestCase):
    def test_arguments_from_the_binding(self):
        sp = ("sp",)
        bind = {("arg", "r3"): (sp, 0x230), ("arg", "r4"): (("k",), 0x503)}
        self.assertEqual(inlinescmd.call_text("fn_8062A014__6TxtStrFUi", bind, None),
                         "TxtStr::fn_8062A014(this=sp+0x230, 0x503)")
        self.assertEqual(inlinescmd.call_text("__ct__3FooFPCcf", {("arg", "r3"): (sp, 8)}, None),
                         "Foo::Foo(this=sp+0x8, ?, ?)")
        self.assertEqual(inlinescmd.call_text("fn_80401234", {("arg", "r4"): (sp, 0)}, None), "fn_80401234(sp)")

    def test_callee_registers_come_from_the_mangled_name(self):
        self.assertEqual(inlines.callee_regs("bl fn_X__3FooFfi"), ("r3", "f1", "r4"))
        self.assertEqual(inlines.callee_regs("bl __dl__7XObjectFPv"), ("r3",))
        self.assertEqual(inlines.callee_regs("bl fn_Y__FPCce"), ("r3",))
        self.assertIsNone(inlines.callee_regs("bl fn_80401234"))
        self.assertIsNone(inlines.callee_regs("bctrl"))


if __name__ == "__main__":
    unittest.main()
