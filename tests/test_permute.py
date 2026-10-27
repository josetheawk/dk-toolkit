import random
import unittest

from tools.dk.platforms.gc_mwcc.permute import body_span, mutate, take_slot

SRC = """int Foo::bar(int a) {
    int x = a;
    int y = 2;
    x += y;
    m4 = x * y;
    return x;
}
"""


class PermuteTest(unittest.TestCase):
    def test_body_span_and_rewrites_stay_in_the_body(self):
        lo, hi = body_span(SRC, "bar")
        lines = SRC.splitlines()
        self.assertEqual(lines[lo].strip(), "int x = a;")
        self.assertEqual(lines[hi].strip(), "return x;")
        rng = random.Random(1)
        seen = set()
        for _ in range(300):
            out = mutate(lines, lo, hi, rng)
            if out:
                self.assertEqual(out[0], lines[0])          # the signature never changes
                self.assertEqual(sorted(out), sorted(out))  # same number of lines
                seen.add("\n".join(out))
        self.assertTrue(any("x = x + y;" in s for s in seen))   # compound expanded
        self.assertTrue(any("y * x" in s for s in seen))         # operands commuted


if __name__ == "__main__":
    unittest.main()


def test_slots_are_exclusive(tmp_path):
    held = [take_slot(tmp_path, slots=2) for _ in range(2)]
    assert all(held)
    assert take_slot(tmp_path, slots=2) is None
    held[0].close()
    assert take_slot(tmp_path, slots=2) is not None
