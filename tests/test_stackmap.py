import unittest

from tools.dk.platforms.gc_mwcc.diffcmd import stack_rows


def rows(*texts):
    return {"instructions": [{"instruction": {"formatted": t}} for t in texts]}


class StackMapTest(unittest.TestCase):
    def test_runs_of_moved_slots_and_where_locals_are_missing(self):
        target = rows("stwu r1, -0xc0(r1)", "stw r0, 0xc4(r1)", "lwz r3, 0x8(r1)", "addi r4, r1, 0x14",
                      "stfd f0, 0x80(r1)", "lwz r0, 0x84(r1)", "lfd f31, 0xb0(r1)")
        ours = rows("stwu r1, -0x90(r1)", "stw r0, 0x94(r1)", "lwz r3, 0x8(r1)", "addi r4, r1, 0x14",
                    "stfd f0, 0x48(r1)", "lwz r0, 0x4c(r1)", "lfd f31, 0x80(r1)")
        got = stack_rows(target, ours)
        self.assertEqual(got[0], "  stack: frame 0x90 in ours, 0xC0 in the target (+0x30)")
        self.assertIn("    ours 0x8-0x14: same", got)
        self.assertIn("    ours 0x48-0x4C: target +0x38", got)
        self.assertTrue(any("the target has 0x38 more bytes of locals between ours 0x14 and 0x48" in r for r in got))
        self.assertTrue(any("ours has 0x8 more bytes of locals between 0x4C and 0x80" in r for r in got))

    def test_nothing_when_the_frames_agree(self):
        same = rows("stwu r1, -0x20(r1)", "lwz r3, 0x8(r1)")
        self.assertEqual(stack_rows(same, same), [])
        self.assertEqual(stack_rows(same, None), [])


if __name__ == "__main__":
    unittest.main()
