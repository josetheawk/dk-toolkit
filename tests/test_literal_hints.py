import unittest

from tools.dk.platforms.gc_mwcc.diffcmd import stripped_hints


class StrippedHintTest(unittest.TestCase):
    """stripped_hints: pool objects whose first kept use comes after a later object's first use."""

    def test_uses_in_code_order_give_no_hint(self):
        uses = {0x0: (0, 0x10, "fA"), 0x4: (0, 0x20, "fA"), 0x8: (1, 0x4, "fB")}
        self.assertEqual(stripped_hints([0x0, 0x4, 0x8], uses), {})

    def test_an_object_first_used_by_a_later_function_was_used_first_by_stripped_code(self):
        # target pool: 0f, -1f, 4330 (vf8); the kept -1f user fn_80557BBC comes before vf8 in .text
        uses = {0x0: (0, 0x8, "fn_805575EC"), 0x8: (5, 0x30, "vf8"), 0x10: (2, 0xB4, "fn_80557BBC"),
                0x14: (3, 0x18, "fn_80557D9C")}
        hints = stripped_hints([0x0, 0x8, 0x10, 0x14], uses)
        self.assertEqual(list(hints), [0x8])
        self.assertIn("before +0x10", hints[0x8])
        self.assertIn("dead-stripped before fn_80557BBC", hints[0x8])

    def test_order_inside_one_function_is_no_evidence(self):
        # the scheduler hoists a double constant's load above the floats before it in the source
        uses = {0x0: (0, 0x40, "fn_80542AA0"), 0x4: (0, 0x14, "fn_80542AA0")}
        self.assertEqual(stripped_hints([0x0, 0x4], uses), {})

    def test_unused_objects_are_skipped(self):
        uses = {0x0: (1, 0x0, "fB"), 0x8: (0, 0x0, "fA")}
        self.assertEqual(list(stripped_hints([0x0, 0x4, 0x8], uses)), [0x0])


if __name__ == "__main__":
    unittest.main()
