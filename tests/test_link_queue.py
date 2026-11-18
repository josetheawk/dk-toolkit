import unittest

from tools.dk.platforms.gc_mwcc.matchcmds import link_blockers
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore


class LinkBlockersTest(unittest.TestCase):
    def test_each_blocker_once_and_a_size_shift_is_the_functions(self):
        s = UnitScore("u", [FunctionScore("f", 0x100, 99.0), FunctionScore("g", 0x40, 100.0)],
                      {".data": 100.0, ".sdata2": 50.0},
                      layout_problems=["f: +0x4 bytes against the target, so the 1 function(s) after it sit that far off "
                                       "(its size, not the order)"],
                      link_notes=["undefined at link: ours declares 2 of the unit's own .sbss objects extern (a, b): "
                                  "define them in this unit"])
        got = [b for b, _ in link_blockers(s)]
        self.assertEqual(got, ["f 99.00% (0x100): close it (dk close / dk try / dk regs)",
                               ".sdata2 50.00%: literal order or values (dk diff --literals)",
                               "own data not defined (dk gen --data)"])

    def test_an_order_difference_is_an_emission_order_blocker(self):
        s = UnitScore("u", [FunctionScore("f", 0x10, 100.0)], {},
                      layout_problems=["__sinit_u_cpp: at .text+0x40 (after stripping), target .text+0x20"])
        self.assertEqual([b for b, _ in link_blockers(s)],
                         ["emission order: __sinit_u_cpp: at .text+0x40 (after stripping), target .text+0x20"])


if __name__ == "__main__":
    unittest.main()
