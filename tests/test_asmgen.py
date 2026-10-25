import unittest

from tools.dk.platforms.gc_mwcc import asmgen
from tools.dk.platforms.gc_mwcc.gen import Unsupported

SECTIONS = {"var": ".sbss", "k": ".sdata2", "table": ".rodata"}.get

LISTING = """.fn fn_80148718, global
/* 80148718 00145638  80 6D 00 00 */\tlwz r3, var@sda21(r0)
/* 8014871C 0014563C  C0 02 00 00 */\tlfs f0, k@sda21(r0)
/* 80148720 00145640  3C 80 00 00 */\tlis r4, table@ha
/* 80148724 00145644  38 84 00 00 */\taddi r4, r4, table@l
.L_80148728:
/* 80148728 00145648  F1 C4 00 00 */\tpsq_st f14, 0x0(r4), 0, qr0
/* 8014872C 0014564C  4C C6 31 82 */\tcrclr cr1eq
/* 80148730 00145650  48 00 00 01 */\tbl fn_8013EB7C__5ClassFv
/* 80148734 00145654  42 00 FF F4 */\tbdnz .L_80148728
/* 80148738 00145658  38 A0 00 00 */\tli r5, var@sda21
/* 8014873C 0014565C  7C 70 E2 A6 */\tmfspr r3, GQR0
/* 80148740 00145660  4E 80 00 20 */\tblr
.endfn fn_80148718
"""


class AsmFunctionTest(unittest.TestCase):
    def test_listing_becomes_codewarrior_asm(self):
        src, defined = asmgen.asm_function("fn_80148718", LISTING, SECTIONS,
                                           lambda sym, size, addend=0: bytes.fromhex("3f800000") if sym == "k" else None)
        self.assertIn('extern "C" asm void fn_80148718(void) {\n    nofralloc\n', src)
        for line in ("lwz r3, var(r13)", "lfs f0, k(r2)", "lis r4, table@ha", "L_80148728:",
                     "psq_st f14, 0x0(r4), 0, 0", "crclr 6", "bl fn_8013EB7C__5ClassFv", "bdnz L_80148728",
                     "la r5, var(r13)", "mfspr r3, 912"):
            self.assertIn(line, src)
        self.assertIn('extern "C" void fn_8013EB7C__5ClassFv(void);', src)       # a C++ callee by its linker name
        self.assertIn('extern "C" const float k = 1.0f;', src)                    # a loaded literal carries its value
        self.assertIn('extern "C" int table;', src)                               # a table is the unit's data
        self.assertEqual(defined, {"k"})

    def test_member_function_head(self):
        pre, head = asmgen.declaration("fn_80029800__Q24EAGL5actorCFfPCc")
        self.assertEqual(head, "asm void EAGL::actor::fn_80029800(float, const char*) const")
        self.assertIn("namespace EAGL { struct actor { void fn_80029800(float, const char*) const; }; }", pre)

    def test_small_data_in_a_normal_section_is_refused(self):
        with self.assertRaises(Unsupported):
            asmgen.convert_line("lwz", "r3, x@sda21(r0)", {"x": ".data"}.get)


if __name__ == "__main__":
    unittest.main()
