"""regs --residue's row classes (residue.py) on synthetic objdiff rows."""

import unittest

from tools.dk.platforms.gc_mwcc.residue import classify, summary


def row(text, kind=None, line=None):
    if text is None:
        return {"diff_kind": kind} if kind else {}
    r = {"instruction": {"formatted": text, "line_number": line}}
    if kind:
        r["diff_kind"] = kind
    return r


def run(pairs, pad=14):
    """pairs of (target, ours) texts; None = that side has no instruction. Padded with equal rows both ends."""
    same = [(row("nop"), row("nop"))] * pad
    body = []
    for t, o in pairs:
        kind = "DIFF_ARG_MISMATCH" if t and o and t.split()[0] == o.split()[0] else \
            "DIFF_INSERT" if t is None else "DIFF_DELETE" if o is None else "DIFF_REPLACE"
        body.append((row(t, kind, 7), row(o, kind, 7)))
    allrows = same + body + same
    return classify([t for t, _ in allrows], [o for _, o in allrows])


class ResidueTest(unittest.TestCase):
    def test_commutative_operands_swapped(self):
        got = run([("fcmpu cr0, f4, f1", "fcmpu cr0, f1, f4"), ("add r27, r0, r24", "add r27, r24, r0")])
        self.assertEqual([r["class"] for r in got], ["OPERAND-SWAP", "OPERAND-SWAP"])
        self.assertEqual(got[0]["line"], 7)

    def test_non_commutative_swap_is_not_operand_swap(self):
        got = run([("subf r3, r4, r5", "subf r3, r5, r4")])
        self.assertNotEqual(got[0]["class"], "OPERAND-SWAP")

    def test_copies(self):
        got = run([("fmr f1, f2", "fmr f2, f1"), ("mr r31, r3", None)])
        self.assertEqual([r["class"] for r in got], ["COALESCE", "COALESCE"])

    def test_extra_memory_access_is_stack(self):
        got = run([("lwz r0, 0x10(r31)", None), (None, "stw r3, 0x8(r1)")])
        self.assertEqual([r["class"] for r in got], ["STACK", "STACK"])

    def test_same_instruction_nearby_is_schedule(self):
        got = run([("lwz r4, 0x4(r3)", "addi r5, r5, 1"), ("addi r5, r5, 1", "lwz r4, 0x4(r3)")])
        self.assertEqual([r["class"] for r in got], ["SCHEDULE", "SCHEDULE"])

    def test_register_only_left_is_register(self):
        got = run([("lwz r25, 0x40(r3)", "lwz r28, 0x40(r3)")])
        self.assertEqual(got[0]["class"], "REGISTER")

    def test_prologue_rows_are_frame(self):
        rows = [row("stwu r1, -0x20(r1)", "DIFF_ARG_MISMATCH")] + [row("nop")] * 30
        ours = [row("stwu r1, -0x30(r1)", "DIFF_ARG_MISMATCH")] + [row("nop")] * 30
        self.assertEqual(classify(rows, ours)[0]["class"], "FRAME")

    def test_same_text_is_a_relocation(self):
        got = run([("addi r4, r26, 0xcc", "addi r4, r26, 0xcc")])
        self.assertEqual(got[0]["class"], "RELOCATION")

    def test_branch_to_another_address_is_a_relocation(self):
        got = run([("bne 0x8730", "bne 0x8748"), ("bne cr1, 0x10", "beq cr1, 0x10")])
        self.assertEqual([r["class"] for r in got][:1], ["RELOCATION"])
        self.assertNotEqual(got[1]["class"], "RELOCATION")

    def test_equal_rows_are_not_listed(self):
        self.assertEqual(run([]), [])
        self.assertEqual(summary(run([("fmr f1, f2", "fmr f2, f1")]))["COALESCE"], 1)


if __name__ == "__main__":
    unittest.main()
