import struct
import unittest

from tools.dk.platforms.gc_mwcc import boundaries
from tools.dk.platforms.gc_mwcc.dtk_symbols import Symbol, Unit


def fn(name, address, size=0x40):
    return Symbol(name, ".text", address, {"type": "function", "size": f"0x{size:X}"})


def obj(name, section, address, size, data=None):
    attrs = {"type": "object", "size": f"0x{size:X}"}
    if data:
        attrs["data"] = data
    return Symbol(name, section, address, attrs)


class Memory:
    """The original bytes of a synthetic image: address -> bytes."""

    def __init__(self):
        self.blocks: dict[int, bytes] = {}

    def put(self, address: int, data: bytes) -> None:
        self.blocks[address] = data

    def read(self, address: int, size: int) -> bytes:
        for a, d in self.blocks.items():
            if a <= address and address + size <= a + len(d):
                return d[address - a:address - a + size]
        return bytes(size)


def f32(*values):
    return b"".join(struct.pack(">f", v) for v in values)


FUNCS = [fn("fA", 0x1000), fn("fB", 0x1040), fn("fC", 0x1080), fn("fD", 0x10C0)]
UNIT = Unit("u.cpp", [(".text", 0x1000, 0x1100)])


class LiteralTest(unittest.TestCase):
    def test_a_repeated_float_places_the_next_pool_and_its_first_user(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 2.0, 1.0, 3.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float"),
                        obj("l2", ".sdata2", 0x9008, 4, "float"), obj("l3", ".sdata2", 0x900C, 4, "float")]
        refs = {"fA": ["l0"], "fB": ["l1"], "fC": ["l2", "l3"], "fD": ["l3"]}
        rep = boundaries.analyse(UNIT, syms, refs, mem.read)
        self.assertEqual([(c.function.name, c.signals, c.clean) for c in rep.cuts], [("fC", ["literal"], True)])
        ev = rep.cuts[0].evidence[0]
        self.assertEqual((ev.section, ev.address), (".sdata2", 0x9008))
        self.assertIn("1f at 0x00009000 and 0x00009008", ev.detail)
        self.assertEqual(rep.cuts[0].earliest.name, "fC")
        self.assertEqual(boundaries.propose_line(rep, rep.cuts[0]), "./dk unit propose 0x00001080 0x00001100")

    def test_bit_patterns_are_not_pooled_literals(self):
        mem = Memory()
        mem.put(0x9000, b"\xff\xff\xff\xff" * 2)
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float")]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["l0"], "fC": ["l1"]}, mem.read)
        self.assertEqual(rep.evidence, [])

    def test_users_that_interleave_are_not_a_clean_cut(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 1.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float")]
        rep = boundaries.analyse(UNIT, syms, {"fB": ["l1"], "fC": ["l0"]}, mem.read)
        self.assertEqual([(c.function.name, c.clean) for c in rep.cuts], [("fB", False)])


class CallGraphTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()
        self.mem.put(0x9000, f32(1.0, 1.0))
        self.fns = [fn(f"f{k}", 0x1000 + 0x40 * k) for k in range(6)]
        self.syms = self.fns + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float")]
        self.unit = Unit("u.cpp", [(".text", 0x1000, 0x1180)])

    def test_callers_move_the_start_inside_the_data_range(self):
        # f0 uses the first copy, f4 the second: the next TU starts after f0 and by f4. f1 is f0's
        # helper; f2 calls f5 (past the cut, nothing before it uses f5): f2 starts the next TU
        refs = {"f0": ["l0"], "f4": ["l1"]}
        calls = {"f0": ["f1"], "f2": ["f5"]}
        rep = boundaries.analyse(self.unit, self.syms, refs, self.mem.read, calls=calls)
        cut = rep.cuts[0]
        self.assertEqual((cut.earliest.name, cut.function.name, cut.first.name), ("f1", "f4", "f2"))
        self.assertEqual(boundaries.propose_line(rep, cut), "./dk unit propose 0x00001080 0x00001180")
        self.assertIn("the call graph picks this one", boundaries.render(rep)[1])

    def test_without_evidence_the_data_cut_stands(self):
        rep = boundaries.analyse(self.unit, self.syms, {"f0": ["l0"], "f4": ["l1"]}, self.mem.read, calls={})
        self.assertIsNone(rep.cuts[0].start)
        self.assertEqual(rep.cuts[0].first.name, "f4")

    def test_a_function_both_sides_call_is_no_evidence(self):
        # f5 is called from before the range (f0), from it (f2) and after it (f4): a shared base, not a sign
        calls = {"f0": ["f5"], "f2": ["f5"], "f4": ["f5"]}
        rep = boundaries.analyse(self.unit, self.syms, {"f0": ["l0"], "f4": ["l1"]}, self.mem.read, calls=calls)
        self.assertIsNone(rep.cuts[0].start)


class PadTest(unittest.TestCase):
    def test_a_float_padded_to_eight_before_the_next_pool(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 0.0, 2.0, 1.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 8, "float"), obj("l1", ".sdata2", 0x9008, 4, "float"),
                        obj("l2", ".sdata2", 0x900C, 4, "float")]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["l0"], "fB": ["l1"], "fD": ["l2"]}, mem.read)
        self.assertEqual([(c.function.name, c.signals) for c in rep.cuts], [("fB", ["gap", "literal"])])
        self.assertEqual(rep.cuts[0].evidence[0].address, 0x9008)

    def test_a_repeat_across_a_placed_boundary_joins_it(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 0.0, 2.0, 1.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 8, "float"), obj("l1", ".sdata2", 0x9008, 4, "float"),
                        obj("l2", ".sdata2", 0x900C, 4, "float")]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["l0"], "fB": ["l1"], "fD": ["l2"]}, mem.read)
        self.assertEqual(len(rep.cuts), 1)
        self.assertEqual(rep.cuts[0].signals, ["gap", "literal"])
        literal = next(e for e in rep.evidence if e.signal == "literal")
        self.assertIs(literal.cut, rep.cuts[0].function)          # 1f at 0x9000 and 0x900C: the same TU start

    def test_an_array_reached_through_its_address_has_no_pad(self):
        # `Slot* table[2]` that dtk sized as one 4-byte element: code takes its address, so the
        # next 4 bytes are the array's second element, not a pad
        mem = Memory()
        syms = FUNCS + [obj("t", ".sdata", 0x9000, 8, "4byte"), obj("u", ".sdata", 0x9008, 4, "4byte")]
        refs = {"fA": ["t"], "fD": ["t", "u"]}
        rep = boundaries.analyse(UNIT, syms, refs, mem.read, asm={"fA": "li r30, t@sda21\n", "fD": "lwz r3, u@sda21(r13)\n"})
        self.assertEqual(rep.evidence, [])
        rep = boundaries.analyse(UNIT, syms, {"fA": ["t"], "fD": ["u"]}, mem.read,
                                 asm={"fA": "lwz r3, t@sda21(r13)\n", "fD": "lwz r3, u@sda21(r13)\n"})
        self.assertEqual([e.signal for e in rep.evidence], ["gap"])     # read only directly: still a pad

    def test_a_strings_pad_stays_a_pad_though_its_address_is_taken(self):
        mem = Memory()
        mem.put(0xA000, b"abc\0\0\0\0\0")
        syms = FUNCS + [obj("s", ".data", 0xA000, 4), obj("t", ".data", 0xA008, 4)]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["s"], "fC": ["t"]}, mem.read,
                                 asm={"fA": "addi r4, r3, s@l\n", "fC": "lwz r3, t@l(r3)\n"})
        self.assertEqual([(c.function.name, c.signals) for c in rep.cuts], [("fC", ["gap"])])

    def test_a_double_needs_its_eight_byte_alignment(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 0.0) + struct.pack(">d", 2.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("d0", ".sdata2", 0x9008, 8, "double")]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["l0"], "fC": ["d0"]}, mem.read)
        self.assertEqual(rep.evidence, [])


class VtableTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()
        self.mem.put(0xA000, b"Foo\0")
        self.mem.put(0xA010, b"%d items\0")

    def syms(self):
        return FUNCS + [obj("name_Foo", ".data", 0xA000, 4), obj("__vt__3Foo", ".data", 0xA004, 12),
                        obj("str", ".data", 0xA010, 9)]

    def test_code_data_after_the_last_vtable(self):
        refs = {"fA": ["__vt__3Foo"], "fC": ["str"]}
        rep = boundaries.analyse(UNIT, self.syms(), refs, self.mem.read, data_refs={"__vt__3Foo": ["name_Foo", "fB"]})
        self.assertEqual([(c.function.name, c.signals) for c in rep.cuts], [("fC", ["vtable"])])
        self.assertIn("follows __vt__3Foo, the unit's last vtable", rep.cuts[0].evidence[0].detail)

    def test_data_of_an_inline_virtual_is_no_sign(self):
        refs = {"fA": ["__vt__3Foo"], "fD": ["str"]}
        rep = boundaries.analyse(UNIT, self.syms(), refs, self.mem.read, data_refs={"__vt__3Foo": ["fD"]})
        self.assertEqual(rep.cuts, [])

    def test_rtti_after_a_vtable_is_no_sign(self):
        syms = FUNCS + [obj("__vt__3Foo", ".data", 0xA004, 12), obj("rtti", ".data", 0xA010, 8)]
        rep = boundaries.analyse(UNIT, syms, {"fA": ["__vt__3Foo"], "fC": ["rtti"]}, self.mem.read,
                                 data_refs={"__vt__3Foo": ["rtti"]})
        self.assertEqual(rep.cuts, [])


class FalsePositiveTest(unittest.TestCase):
    def test_array_initializers_copied_whole_are_not_pooled_literals(self):
        # `char name[0x20] = "actor";` in two functions: each has its own initializer, copied to the stack
        mem = Memory()
        mem.put(0x8000, b"actor".ljust(0x20, b"\0") * 2)
        syms = FUNCS + [obj("a0", ".rodata", 0x8000, 0x20), obj("a1", ".rodata", 0x8020, 0x20)]
        copy = lambda name: (f"lis r3, {name}@ha\naddi r7, r3, {name}@l\n"
                             + "".join(f"lwz r{8 + k}, 0x{4 * k:x}(r7)\n" for k in range(8)))
        refs = {"fA": ["a0"], "fC": ["a1"]}
        rep = boundaries.analyse(UNIT, syms, refs, mem.read, asm={"fA": copy("a0"), "fC": copy("a1")})
        self.assertEqual(rep.evidence, [])
        rep = boundaries.analyse(UNIT, syms, refs, mem.read, asm={"fA": "", "fC": ""})        # string literals: a sign
        self.assertEqual([e.signal for e in rep.evidence], ["literal"])

    def test_a_named_const_other_units_read_is_not_a_pooled_literal(self):
        mem = Memory()
        mem.put(0x9000, f32(-45.0, -45.0))
        syms = FUNCS + [fn("other", 0x2000), obj("k45", ".sdata2", 0x9000, 4, "float"),
                        obj("l45", ".sdata2", 0x9004, 4, "float")]
        refs = {"fA": ["k45"], "fC": ["l45"], "other": ["k45"]}      # `extern const float k45` in another unit
        self.assertEqual(boundaries.analyse(UNIT, syms, refs, mem.read).evidence, [])
        del refs["other"]
        self.assertEqual([e.signal for e in boundaries.analyse(UNIT, syms, refs, mem.read).evidence], ["literal"])

    def test_a_literal_used_on_both_sides_vetoes_the_cut(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 1.0))
        mem.put(0xB000, b"XWidget\0")
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float"),
                        obj("s0", ".sdata", 0xB000, 8)]
        refs = {"fA": ["l0", "s0"], "fC": ["l1"], "fD": ["s0"]}
        rep = boundaries.analyse(UNIT, syms, refs, mem.read)
        self.assertEqual(rep.cuts, [])
        self.assertEqual([(c.function.name, name) for c, name in rep.vetoed], [("fC", "s0")])
        self.assertTrue(any("is used on both sides" in line for line in boundaries.render(rep)))

    def test_data_after_the_last_vtable_used_by_the_sinit_and_factories(self):
        mem = Memory()
        mem.put(0xA010, b"XSimpleText\0")
        fns = [fn("fA", 0x1000), fn("fB", 0x1040), fn("__sinit_u_cpp", 0x1080), fn("fFactory", 0x10C0)]
        syms = fns + [obj("__vt__3Foo", ".data", 0xA004, 12), obj("name", ".data", 0xA010, 12)]
        refs = {"fA": ["__vt__3Foo"], "__sinit_u_cpp": ["name"], "fFactory": ["name"]}
        rep = boundaries.analyse(UNIT, syms, refs, mem.read)
        self.assertEqual(rep.cuts, [])


class AlignTest(unittest.TestCase):
    def test_a_declared_data_start_off_eight(self):
        unit = Unit("u.cpp", [(".text", 0x1000, 0x1100), (".data", 0xA004, 0xA020)])
        syms = FUNCS + [obj("a", ".data", 0xA004, 4), obj("b", ".data", 0xA008, 4)]
        rep = boundaries.analyse(unit, syms, {}, Memory().read)
        self.assertEqual([(e.signal, e.address) for e in rep.evidence], [("align", 0xA004)])
        self.assertTrue(any("off an 8-byte boundary" in line for line in boundaries.render(rep)))

    def test_another_units_data_is_left_out(self):
        mem = Memory()
        mem.put(0x9000, f32(1.0, 1.0))
        syms = FUNCS + [obj("l0", ".sdata2", 0x9000, 4, "float"), obj("l1", ".sdata2", 0x9004, 4, "float")]
        owners = {".sdata2": [(0x9000, 0x9004, "other.cpp")]}
        rep = boundaries.analyse(UNIT, syms, {"fA": ["l0"], "fC": ["l1"]}, mem.read, owners)
        self.assertEqual(rep.evidence, [])


if __name__ == "__main__":
    unittest.main()
