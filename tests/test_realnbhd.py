"""dk dataset real-nbhd (realnbhd.py): the pure parts (tier order, direction, round-robin budget order, diff
class). The census itself compiles real units; it is measured by its run (T44)."""

import unittest

from tools.dk.platforms.gc_mwcc.realnbhd import better, diff_class, direction, round_robin, tier_key


class TierTest(unittest.TestCase):
    def test_order(self):
        exact, shape, rows = tier_key(100.0, 0, 0), tier_key(99.0, 1, 5), tier_key(98.0, 1, 3)
        self.assertTrue(better(exact, rows) and better(rows, shape))
        self.assertTrue(better(tier_key(90.0, 0, 9), tier_key(99.9, 1, 1)))       # fewer shape rows beats %
        self.assertFalse(better(tier_key(97.0, 1, 3), tier_key(98.0, 1, 3)))      # same tier: no record

    def test_direction(self):
        a, b = tier_key(95.0, 2, 10), tier_key(96.0, 1, 10)
        self.assertEqual((direction(a, b), direction(b, a), direction(a, a)), ("forward", "reverse", None))

    def test_round_robin_and_class(self):
        got = round_robin({"x": [1, 2, 3], "y": [4], "z": [5, 6]}, ["y", "x"])
        self.assertEqual(got, [("y", 4), ("x", 1), ("z", 5), ("x", 2), ("z", 6), ("x", 3)])
        self.assertEqual((diff_class(0, 4), diff_class(2, 4)), ("register", "shape"))


if __name__ == "__main__":
    unittest.main()
