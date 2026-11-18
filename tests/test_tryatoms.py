"""dk try --replace / --atoms / --grid / --oracle (tryatoms.py).

Synthetic: atom files, finding OLD, combining atoms, the placeholder-name guard, the grid order."""

import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import tryatoms
from tools.dk.platforms.gc_mwcc.tryatoms import Atom, compose, grid, locate, parse_atoms, placeholder_names

REPO = Path(__file__).resolve().parents[1]

SRC = ("static inline float h(float a) {\n    return a * 2.0f;\n}\n\n"
       "float f(float x, float y) {\n    float s = x + y;\n    s -= y / 2;\n    return h(s);\n}\n")


class AtomTest(unittest.TestCase):
    def span(self):
        return SRC.index("float f("), len(SRC)

    def test_atom_file(self):
        atoms = parse_atoms("// label: fold\n    float s = x + y;\n    s -= y / 2;\n// =>\n"
                            "    float s = x + y - y / 2;\n// ---\nreturn h(s);\n// =>\nreturn h(s + 0.0f);\n")
        self.assertEqual([a.label for a in atoms], ["fold", ""])
        self.assertEqual(atoms[0].new, "    float s = x + y - y / 2;")
        self.assertEqual(atoms[1].name(1), "a1")
        with self.assertRaises(ValueError):
            parse_atoms("no arrow here\n")

    def test_old_is_found_once_inside_the_function_first(self):
        span = self.span()
        self.assertEqual(SRC[slice(*locate(SRC, "s -= y / 2;", span))], "s -= y / 2;")
        self.assertIsNotNone(locate(SRC, "float s =   x+y;", span))           # whitespace folded
        self.assertEqual(locate(SRC, "return", span)[0], SRC.index("return h(s)"))   # the one inside the function
        self.assertIsNotNone(locate(SRC, "a * 2.0f", span))                  # outside, but once in the file
        self.assertIsNone(locate(SRC, "float", (0, len(SRC))))               # several: ambiguous
        self.assertIsNone(locate(SRC, "nowhere", span))

    def test_atoms_combine_unless_they_overlap(self):
        span = self.span()
        atoms = [Atom("s -= y / 2;", "s = s - y / 2;"), Atom("return h(s);", "return h(s) + 0.0f;"),
                 Atom("float s = x + y;\n    s -= y / 2;", "float s = x + y - y / 2;")]
        places = [locate(SRC, a.old, span) for a in atoms]
        both = compose(SRC, places, atoms, (0, 1))
        self.assertIn("s = s - y / 2;\n    return h(s) + 0.0f;", both)
        self.assertIsNone(compose(SRC, places, atoms, (0, 2)))                 # the same text twice
        self.assertIsNone(compose(SRC, places + [None], atoms + [Atom("x", "y")], (3,)))

    def test_new_locals_need_real_names(self):
        self.assertEqual(placeholder_names(Atom("s -= y / 2;", "float t0 = y / 2;\n    s -= t0;"), SRC), ["t0"])
        self.assertEqual(placeholder_names(Atom("s -= y / 2;", "__typeof__(y) half = y / 2;"), SRC), ["__typeof__"])
        self.assertEqual(placeholder_names(Atom("s -= y / 2;", "float half = y / 2;\n    s -= half;"), SRC), [])
        self.assertEqual(placeholder_names(Atom("s -= y / 2;", "s -= (int)y / 2;"), SRC), [])

    def test_grid_order(self):
        pairs = list(grid(4, 2, [3, 0, 1, 2]))
        self.assertEqual(pairs[0], (1, 2))                                     # the two best singles first
        self.assertEqual(len(pairs), 6)
        late = list(grid(4, 2, [0, 1, 2, 3], neutral={0}))
        self.assertTrue(all(0 in c for c in late[-3:]))                        # a neutral atom's pairs last
        self.assertEqual(len(list(grid(4, 3))), 6 + 4)

    def test_register_better_mark(self):
        base = tryatoms.Row((), "", right=313, known=314, rules_left=["r"])
        self.assertTrue(tryatoms.register_better(tryatoms.Row((1,), "", right=314, known=314), base))
        self.assertFalse(tryatoms.register_better(tryatoms.Row((1,), "", right=313, known=314), base))
        self.assertFalse(tryatoms.register_better(tryatoms.Row((1,), "", right=314, known=314, reachable=False), base))


if __name__ == "__main__":
    unittest.main()
