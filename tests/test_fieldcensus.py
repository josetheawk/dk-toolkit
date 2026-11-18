import json
import unittest
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import fieldcensus as fc


def asm(*lines):
    return "\n".join(f"/* {0x80001000 + 4 * k:08X} {4 * k:08X}  00 00 00 00 */\t{line}" for k, line in enumerate(lines))


ENTRIES = {
    # fn_A(Bar* r3, ...): reads through r3, passes r3 unchanged to a Bar method (the callee rule)
    "fn_A": SimpleNamespace(kind="fn", text=asm("lwz r4, 0x8(r3)", "lfs f1, 0x10(r3)", "lha r5, 0x4(r3)",
                                                 "lbz r6, 0x2(r3)", "extsb r6, r6", "lbz r7, 0x3(r3)",
                                                 "stfd f1, 0x18(r3)", "bl get__3BarFv", "blr")),
    # fn_B is only called with a typed value: fn_C (a Baz method) passes its this
    "fn_B": SimpleNamespace(kind="fn", text=asm("lwz r0, 0x0(r3)", "blr")),
    "run__3BazFv": SimpleNamespace(kind="fn", text=asm("bl fn_B", "blr")),
    # fn_D: called with two different classes: no type (a conflict is never voted)
    "fn_D": SimpleNamespace(kind="fn", text=asm("blr")),
    "a__3BazFv": SimpleNamespace(kind="fn", text=asm("bl fn_D", "blr")),
    "b__3QuxFv": SimpleNamespace(kind="fn", text=asm("bl fn_D", "blr")),
    "fn_V": SimpleNamespace(kind="fn", text=asm("lwz r3, 0x4(r3)", "blr")),
}


class FieldCensusTest(unittest.TestCase):
    def setUp(self):
        self.rows = fc.census(ENTRIES, {"fn_V": "Base"})

    def test_facts_width_kind_sign(self):
        f = {x["offset"]: x for x in self.rows["fn_A"]["facts"]}
        self.assertEqual((f[8]["width"], f[8]["kind"], f[8]["sign"]), (4, "int", None))
        self.assertEqual((f[16]["width"], f[16]["kind"]), (4, "float"))
        self.assertEqual(f[4]["sign"], "signed")                       # lha
        self.assertEqual(f[2]["sign"], "signed")                       # lbz + extsb of the same register
        self.assertIsNone(f[3]["sign"])                                # a bare lbz says nothing
        self.assertEqual((f[24]["kind"], f[24]["store"]), ("double", True))

    def test_roots(self):
        self.assertEqual(fc.typed_args(self.rows["fn_A"]), ["r3"])
        self.assertEqual(self.rows["fn_A"]["roots"]["r3"]["sources"], ["callee"])
        self.assertEqual(self.rows["run__3BazFv"]["roots"]["r3"]["sources"], ["name"])
        self.assertEqual(self.rows["fn_B"]["roots"]["r3"]["classes"], {"Baz": ["callers"]})
        self.assertNotIn("r3", self.rows["fn_D"]["roots"])             # callers disagree: left untyped
        self.assertEqual(self.rows["fn_V"]["roots"]["r3"]["classes"], {"Base": ["vtable"]})

    def test_demangled_parameters(self):
        self.assertEqual(fc.demangled_roots("set__3FooFP3BarfP3Baz"), {"r3": "Foo", "r4": "Bar", "r5": "Baz"})
        self.assertEqual(fc.demangled_roots("fn_80001000"), {})

    def test_vtable_slot_goes_to_the_base(self):
        doc = [{"name": "Base", "rtti": 1, "bases": [], "vtables": [{"offset": 0, "slots": [100, 104]}]},
               {"name": "Mid", "rtti": 2, "bases": [[1, 0]], "vtables": [{"offset": 0, "slots": [100, 108]}]},
               {"name": "Other", "rtti": 3, "bases": [], "vtables": [{"offset": 0, "slots": [112]},
                                                                      {"offset": -8, "slots": [104]}]}]
        names = {100: "fn_shared", 104: "fn_b", 108: "fn_m", 112: "fn_o"}
        self.assertEqual(fc.rtti_slots(doc, names), {"fn_b": "Base", "fn_m": "Mid", "fn_o": "Other", "fn_shared": "Base"})

    def test_kind_pairs(self):
        row = lambda t: {"instruction": {"formatted": t}}
        flag = lambda t: {"instruction": {"formatted": t}, "diff_kind": "DIFF_REPLACE"}
        left = [flag("lwz r3, 0x8(r31)"), row("lbz r0, 0x2(r3)"), row("extsb r0, r0"), flag("lhz r4, 0x6(r3)"), {}]
        right = [flag("lfs f1, 0x8(r30)"), row("lbz r0, 0x2(r3)"), row("mr r4, r0"), {}, flag("lha r4, 0x6(r3)")]
        self.assertEqual(fc.kind_pairs(left, right), ["lwz<->lfs @0x8", "lbz+extsb one side @0x2", "lhz<->lha @0x6"])
        self.assertEqual(fc.kind_pairs([flag("lwz r3, 0x8(r31)")], [flag("lwz r3, 0xc(r31)")]), [])
        self.assertEqual(fc.kind_pairs([row("lwz r3, 0x8(r31)")], [row("lfs f1, 0x8(r31)")]), [])   # unflagged rows

    def test_counts_and_determinism(self):
        route_rows = [{"unit": "u", "function": "fn_A", "class": "fresh"}, {"unit": "u", "function": "fn_D", "class": "fresh"}]
        c = fc.counts(self.rows, route_rows, {})
        self.assertEqual((c["c1_fresh_fn"], c["c1_typed"], c["c1_share"]), (2, 1, 0.5))
        self.assertEqual(json.dumps(fc.census(ENTRIES, {"fn_V": "Base"}), sort_keys=True),
                         json.dumps(fc.census(dict(reversed(list(ENTRIES.items()))), {"fn_V": "Base"}), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
