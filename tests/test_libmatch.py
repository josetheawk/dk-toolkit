import struct
import unittest

from tests.helpers import make_dol
from tools.dk.platforms.gc_mwcc.dol import Dol
from tools.dk.platforms.gc_mwcc.elf import Reloc
from tools.dk.platforms.gc_mwcc.libmatch import (
    DolIndex, ObjectMatch, RefFunction, locate_functions, masked_equal, masks_for, resolve_conflicts,
)
from tools.dk.platforms.gc_mwcc.dtk_symbols import overlap_victim

BASE = 0x80003100


def fn(name, words, relocs=(), offset=0):
    return RefFunction(name, offset, tuple(words), masks_for(4 * len(words), list(relocs)), len(relocs))


def body(seed, n):
    """n distinctive instruction words."""
    return [0x38600000 | ((seed * 31 + i) & 0xFFFF) for i in range(n - 1)] + [0x4E800020]


class MaskTest(unittest.TestCase):
    def test_reloc_fields_are_ignored(self):
        bl = Reloc(0, 10, "f", 0)       # REL24
        lo = Reloc(6, 4, "x", 0)        # ADDR16_LO on the second word (offset points at the halfword)
        sda = Reloc(8, 109, "y", 0)     # SDA21
        keep = masks_for(12, [bl, lo, sda])
        self.assertEqual(keep, [0xFC000003, 0xFFFF0000, 0xFFE00000])
        self.assertTrue(masked_equal((0x48000001, 0x38630000, 0x806D0000),
                                     (0x48123455, 0x3863ABCD, 0x80621234), keep))
        self.assertFalse(masked_equal((0x48000001,), (0x4C000001,), [0xFC000003]))


class LocateTest(unittest.TestCase):
    def setUp(self):
        # DOL: a(10 words) b(8) [stray 5] c(7) — reference object has a, x (dead-stripped), b, c.
        self.a, self.b, self.c, self.stray = body(1, 10), body(2, 8), body(3, 7), body(9, 5)
        code = self.a + self.b + self.stray + self.c
        dol = Dol.parse(make_dol([("T1", BASE, struct.pack(f">{len(code)}I", *code))]))
        self.addr = {"a": BASE, "b": BASE + 40, "stray": BASE + 72, "c": BASE + 92}
        self.index = DolIndex(dol, [(BASE, 40), (BASE + 40, 32), (BASE + 72, 20), (BASE + 92, 28)])

    def test_chain_skips_stripped_and_reports_gap(self):
        funcs = [fn("a", self.a), fn("x", body(5, 9), offset=32), fn("b", self.b, offset=68), fn("c", self.c, offset=92)]
        m = locate_functions(self.index, "ref:a.c", "v", funcs, 120)
        self.assertEqual([(n, a) for n, a, _ in m.matched], [("a", self.addr["a"]), ("b", self.addr["b"]), ("c", self.addr["c"])])
        self.assertEqual(m.unmatched, ["x"])
        self.assertEqual(m.gap_functions, [self.addr["stray"]])
        self.assertFalse(m.clean)
        self.assertEqual((m.start, m.end), (BASE, BASE + 120))

    def test_clean_when_span_fully_explained(self):
        m = locate_functions(self.index, "ref:b.c", "v", [fn("a", self.a), fn("b", self.b, offset=40)], 72)
        self.assertTrue(m.clean)

    def test_single_function_without_relocations_rejected(self):
        big = body(4, 20)
        self.assertEqual(locate_functions(self.index, "ref:z.c", "v", [fn("z", big)], 80).matched, [])

    def test_small_chain_rejected(self):
        m = locate_functions(self.index, "ref:s.c", "v", [fn("c", self.c)], 28)
        self.assertEqual(m.matched, [])

    def test_conflicts_go_to_strongest(self):
        strong = ObjectMatch("ref:big.c", "v", 0, 3, [("a", 1, 64), ("b", 2, 64), ("c", 3, 64)])
        weak = ObjectMatch("ref:small.c", "v", 0, 2, [("c", 3, 64), ("d", 4, 16)])
        kept = resolve_conflicts([weak, strong])
        self.assertEqual([m.source for m in kept], ["ref:big.c"])   # weak lost c, d alone is too small
        self.assertIn("c", weak.unmatched)


class OverlapTest(unittest.TestCase):
    def test_picks_auto_named_symbol(self):
        msg = "Symbol LastResetEnd (0x803928A8..0x803928B0) overlaps with symbol lbl_803928AC (0x803928AC..0x803928B0, align 1)"
        self.assertEqual(overlap_victim(msg), ("lbl_803928AC", 0x803928AC))
        self.assertIsNone(overlap_victim("Symbol A (0x1..0x2) overlaps with symbol B (0x1..0x2)"))
        self.assertIsNone(overlap_victim("something else"))


if __name__ == "__main__":
    unittest.main()


class BuildErrorsTest(unittest.TestCase):
    def test_objdiff_failure_beats_ninja_line_and_noise_is_dropped(self):
        from tools.dk.platforms.gc_mwcc.dtk_symbols import build_errors, caused_by
        out = ("FAILED: build/GH2E69/report.json\n"
               "Failed: Symbol gap_09_803909E3_sdata size out of section bounds (16 > 12)\n"
               + "Linker Warning: FORCEACTIVE symbol x not found\n" * 40)
        self.assertEqual(caused_by(out), "Failed: Symbol gap_09_803909E3_sdata size out of section bounds (16 > 12)")
        text = build_errors(out)
        self.assertNotIn("FORCEACTIVE", text)
        self.assertIn("extend the previous unit's split", text)
