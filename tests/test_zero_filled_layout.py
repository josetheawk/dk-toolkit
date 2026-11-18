"""Where the link keeps a discarded weak copy in place (its section is reached through the
section symbol by an object it keeps), dk check models the linked layout, not just the bytes."""

import unittest

from tests.test_zero_filled_kept import GLOBAL, LOCAL, WEAK, make_object
from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.matching import section_symbol_reached, zero_filled_layout, zero_filled_problem

VT = b"\0" * 8                     # a vtable: all relocations
OFFSET_VT = b"\0" * 4 + b"\xff\xff\xff\x48"   # a this-adjust word survives the link in place


def ours(data_objects, rtti_kept=True):
    """.data: data_objects (name, bytes, bind, relocated?); .sdata: an RTTI record pointing at
    .data's start through the section symbol (as GC 1.3 emits base lists)."""
    data, symbols, relocs = b"", [("...data.0", ".data", 0, 0, LOCAL)], []
    for name, body, bind, relocated in data_objects:
        symbols.append((name, ".data", len(data), len(body), bind))
        if relocated:
            relocs.append((".data", len(data), "__RTTI__1A"))
        data += body
    rtti = "__RTTI__1A" if rtti_kept else "__RTTI__4Weak"
    symbols.append((rtti, ".sdata", 0, 8, LOCAL))
    if not rtti_kept:
        symbols.append(("__RTTI__1A", ".sdata", 8, 8, LOCAL))
    relocs.append((".sdata", 4, "...data.0"))
    return Elf.parse(make_object({".data": data, ".sdata": b"\0" * 16}, symbols, relocs))


def target(data_objects):
    data, symbols, relocs = b"", [], []
    for name, body, relocated in data_objects:
        symbols.append((name, ".data", len(data), len(body), GLOBAL))
        if relocated:
            relocs.append((".data", len(data), name))
        data += body
    return Elf.parse(make_object({".data": data}, symbols, relocs))


class ZeroFilledLayoutTest(unittest.TestCase):
    def test_a_kept_reference_through_the_section_symbol_keeps_copies_in_place(self):
        base = ours([("__vt__1A", VT, GLOBAL, True), ("__vt__4Weak", VT, WEAK, True)])
        self.assertIn(".data", section_symbol_reached(base, set()))
        # only a discarded object reaches it: the link removes the copies instead
        self.assertNotIn(".data", section_symbol_reached(ours([("__vt__1A", VT, GLOBAL, True)], rtti_kept=False),
                                                         {"__RTTI__4Weak"}))

    def test_trailing_copies_make_the_section_longer(self):
        # the GC 1.3 shape: the target removed the copies, ours keeps them as zero blocks
        base = ours([("__vt__1A", VT, GLOBAL, True), ("__vt__4Weak", VT, WEAK, True)])
        why = zero_filled_problem(target([("__vt__1A", VT, True)]), base, ".data", {"__vt__4Weak"})
        self.assertIn("ours links 0x10 bytes with them, the target's .data is 0x8", why)

    def test_a_copy_before_the_target_s_zero_block_moves_the_vtable_after_it(self):
        base = ours([("__vt__1A", VT, GLOBAL, True), ("__vt__4Weak", VT, WEAK, True)])
        tgt = target([("lbl_zero", b"\0" * 8, False), ("__vt__1A", VT, True)])
        self.assertIn("__vt__1A lands at +0x0, target +0x8", zero_filled_problem(tgt, base, ".data", {"__vt__4Weak"}))

    def test_a_copy_where_the_target_has_its_block_lines_up_raw_words_included(self):
        base = ours([("__vt__4Weak", OFFSET_VT, WEAK, True), ("__vt__1A", VT, GLOBAL, True)])
        tgt = target([("lbl_zero", OFFSET_VT, False), ("__vt__1A", VT, True)])
        self.assertIsNone(zero_filled_problem(tgt, base, ".data", {"__vt__4Weak"}))

    def test_reported_only_for_a_section_scored_exact(self):
        base = ours([("__vt__1A", VT, GLOBAL, True), ("__vt__4Weak", VT, WEAK, True)])
        tgt = target([("__vt__1A", VT, True)])
        weak = {".data": [(8, 16, "__vt__4Weak")]}
        named = {"__vt__4Weak"}            # symbols.txt names the copy another unit keeps
        got = zero_filled_layout(tgt, base, weak, set(), {".data": 100.0}, named)
        self.assertEqual(len(got), 1)
        self.assertIn("reaches .data through its section symbol", got[0])
        self.assertIn("GC/1.3.2", got[0])
        self.assertEqual(zero_filled_layout(tgt, base, weak, set(), {".data": 97.0}, named), [])
        # no other definition by that name: the link may keep ours as the only copy, not judged
        self.assertEqual(zero_filled_layout(tgt, base, weak, set(), {".data": 100.0}, set()), [])

    def test_a_short_split_points_at_unit_propose(self):
        # the original holds ours' extra bytes right after the unit's range: dtk left them in auto data
        base = ours([("__vt__1A", VT, GLOBAL, True), ("__vt__4Weak", VT, WEAK, True)])
        tgt = target([("__vt__1A", VT, True)])
        weak = {".data": [(8, 16, "__vt__4Weak")]}
        args = (tgt, base, weak, set(), {".data": 100.0}, {"__vt__4Weak"})
        got = zero_filled_layout(*args, lambda sec, n: b"\0" * n, "game/a.cpp")
        self.assertIn("extra 0x8 bytes right after the unit's .data range", got[0])
        self.assertIn("./dk unit propose game/a.cpp", got[0])
        got = zero_filled_layout(*args, lambda sec, n: b"\x11" * n, "game/a.cpp")     # the next unit's data
        self.assertIn("GC/1.3.2", got[0])


if __name__ == "__main__":
    unittest.main()
