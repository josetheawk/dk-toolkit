import unittest

from tools.dk.platforms.gc_mwcc import asmgen
from tools.dk.platforms.gc_mwcc.gen import Unsupported

SECTIONS = {"var": ".sbss", "k": ".sdata2", "table": ".rodata"}.get

LISTING = """.fn fn_80001000, global
/* 80001000 00000000  C0 02 00 00 */\tlfs f0, k@sda21(r0)
/* 80001004 00000004  80 6D 00 00 */\tlwz r3, var@sda21(r0)
/* 80001008 00000008  3C 80 00 00 */\tlis r4, table@ha
/* 8000100C 0000000C  38 84 00 00 */\taddi r4, r4, table@l
.L_80001010:
/* 80001010 00000010  F1 E4 10 08 */\tpsq_st f15, 0x8(r4), 0, qr1
/* 80001014 00000014  4C C6 31 82 */\tcrclr cr1eq
/* 80001018 00000018  48 00 00 01 */\tbl fn_80001400__5ClassFv
/* 8000101C 0000001C  42 00 FF F4 */\tbdnz .L_80001010
/* 80001020 00000020  38 AD 00 00 */\tli r5, var@sda21
/* 80001024 00000024  7C 71 E2 A6 */\tmfspr r3, GQR1
/* 80001028 00000028  4E 80 00 20 */\tblr
.endfn fn_80001000
"""


class AsmFunctionTest(unittest.TestCase):
    def test_listing_becomes_codewarrior_asm(self):
        src, defined = asmgen.asm_function("fn_80001000", LISTING, SECTIONS,
                                           lambda sym, size, addend=0: bytes.fromhex("3f800000") if sym == "k" else None)
        self.assertIn('extern "C" asm void fn_80001000(void) {\n    nofralloc\n', src)
        for line in ("lwz r3, var(r13)", "lfs f0, k(r2)", "lis r4, table@ha", "L_80001010:",
                     "psq_st f15, 0x8(r4), 0, 1", "crclr 6", "bl fn_80001400__5ClassFv", "bdnz L_80001010",
                     "la r5, var(r13)", "mfspr r3, 913"):
            self.assertIn(line, src)
        self.assertIn('extern "C" void fn_80001400__5ClassFv(void);', src)       # a C++ callee by its linker name
        self.assertIn('extern "C" const float k = 1.0f;', src)                    # a loaded literal carries its value
        self.assertIn('extern "C" int table;', src)                               # a table is the unit's data
        self.assertEqual(defined, {"k"})

    def test_member_function_head(self):
        pre, head = asmgen.declaration("fn_80001900__Q22Ns4ItemCFfPCc")
        self.assertEqual(head, "asm void Ns::Item::fn_80001900(float, const char*) const")
        self.assertIn("namespace Ns { struct Item { void fn_80001900(float, const char*) const; }; }", pre)

    def test_small_data_in_a_normal_section_is_refused(self):
        with self.assertRaises(Unsupported):
            asmgen.convert_line("lwz", "r3, x@sda21(r0)", {"x": ".data"}.get)


if __name__ == "__main__":
    unittest.main()
