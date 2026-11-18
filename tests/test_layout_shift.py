import unittest

from tools.dk.platforms.gc_mwcc.matching import size_shifts


class SizeShiftTest(unittest.TestCase):
    def test_functions_pushed_by_one_bigger_function(self):
        want = {"a": (".text", 0x0, 0x20), "b": (".text", 0x20, 0x10), "c": (".text", 0x30, 0x8)}
        have = {"a": (".text", 0x0, 0x30), "b": (".text", 0x30, 0x10), "c": (".text", 0x40, 0x8)}
        shift = {n: v[1] for n, v in have.items()}
        self.assertEqual(size_shifts(want, have, shift),
                         ["a: +0x10 bytes against the target, so the 2 function(s) after it sit that far off "
                          "(its size, not the order)"])

    def test_another_order_is_not_a_size_shift(self):
        want = {"a": (".text", 0x0, 0x10), "b": (".text", 0x10, 0x10)}
        have = {"a": (".text", 0x10, 0x10), "b": (".text", 0x0, 0x10)}
        self.assertIsNone(size_shifts(want, have, {n: v[1] for n, v in have.items()}))


if __name__ == "__main__":
    unittest.main()
