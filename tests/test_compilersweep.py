import unittest

from tools.dk.platforms.gc_mwcc import compilersweep as cs
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore


def unit_score(*pcts, data=None):
    return UnitScore("u", [FunctionScore(f"f{k}", 0x40, p) for k, p in enumerate(pcts)], data or {})


def sweep(base, *variants, path="src/game/u.cpp"):
    unit = {"name": "main/game/u", "metadata": {"source_path": path}}
    s = cs.Sweep(unit, cs.Variant("GC/1.3.2", "c++", base))
    for compiler, lang, score in variants:
        s.variants.append(cs.Variant(compiler, lang, score))
    return s


class CompareTest(unittest.TestCase):
    def test_a_gain_loses_no_exact_function(self):
        base = unit_score(100.0, 90.0, 80.0)
        self.assertTrue(cs.gains(base, unit_score(100.0, 100.0, 80.0)))
        self.assertTrue(cs.gains(base, unit_score(100.0, 95.0, 80.0)))            # a higher score is a gain too
        self.assertFalse(cs.gains(base, unit_score(99.0, 100.0, 100.0)))          # but not at an exact one's cost
        self.assertFalse(cs.gains(base, unit_score(100.0, 90.0, 80.0)))
        self.assertFalse(cs.gains(base, unit_score(100.0, 100.0, 10.0)))          # one more exact, the unit much worse
        self.assertEqual(cs.compare(base, unit_score(99.0, 100.0, 80.0))[:2], (["f1"], ["f0"]))

    def test_c_code_is_unmangled(self):
        self.assertTrue(cs.c_code(["fn_8057F53C", "__wcstoul", "memcpy"]))
        self.assertFalse(cs.c_code(["fn_8057F53C", "fn_80401234__5ClassFv"]))
        self.assertFalse(cs.c_code(["__sinit_snd_cpp"]))
        self.assertFalse(cs.c_code(["__ct__4ItemFv"]))
        self.assertFalse(cs.c_code([]))


class SourceTest(unittest.TestCase):
    def test_the_language_flag_is_replaced(self):
        argv = ["mwcc", "-O4,p", "-lang=c++", "-c", "src/u.cpp", "-o", "out"]
        self.assertEqual(cs.with_lang(argv, "c"), ["mwcc", "-O4,p", "-lang=c", "-c", "src/u.cpp", "-o", "out"])

    def test_extern_c_is_dropped_for_c(self):
        src = 'extern "C" int a[2] = {1, 2};\nextern "C" {\nvoid f(void);\nint g(int x) { return x; }\n}\nint h;\n'
        self.assertEqual(cs.as_c(src), "int a[2] = {1, 2};\n\nvoid f(void);\nint g(int x) { return x; }\n\nint h;\n")

    def test_current_compiler_and_language(self):
        cmd = "wibo sjiswrap.exe build/compilers/GC/1.3/mwcceppc.exe -O4,p -lang=c++ -c src/a.cpp -o x"
        self.assertEqual(cs.current(cmd, "src/a.cpp"), ("GC/1.3", "c++"))
        cmd = "wibo sjiswrap.exe build/compilers/GC/1.3.2/mwcceppc.exe -O4,p -lang=c -c src/a.c -o x"
        self.assertEqual(cs.current(cmd, "src/a.c"), ("GC/1.3.2", "c"))


class RenderTest(unittest.TestCase):
    def test_gains_print_the_unit_flags_line_and_nothing_else(self):
        base = unit_score(100.0, 90.0)
        s1 = sweep(base, ("GC/1.3", "c++", unit_score(100.0, 100.0)))
        s2 = sweep(base, ("GC/1.3", "c++", unit_score(90.0, 100.0)), path="src/game/v.cpp")
        lines, n = cs.render([s1, s2])
        self.assertEqual(n, 1)
        self.assertTrue(lines[0].startswith("  GAIN  game/u.cpp"))
        self.assertIn("exact 1 -> 2", lines[0])
        self.assertEqual(lines[2].strip(), "./dk unit flags game/u.cpp --mw-version GC/1.3")
        lines, _ = cs.render([s1, s2], every=True)
        self.assertTrue(any(line.startswith("  ----  game/v.cpp") and "-1 exact" in line for line in lines))

    def test_a_language_gain_says_to_rename(self):
        s = sweep(unit_score(90.0), ("GC/1.3.2", "c", unit_score(100.0)), path="src/lib/snd/a.cpp")
        lines, _ = cs.render([s])
        self.assertIn("rename src/lib/snd/a.cpp to src/lib/snd/a.c", lines[-1])
        self.assertTrue(lines[-1].strip().startswith("# lib/snd/a.cpp"))


class UniformTest(unittest.TestCase):
    def test_the_inline_mode_is_replaced(self):
        argv = ["mwcc", "-inline", "auto", "-O4,p", "-inline", "noauto", "-c", "src/u.cpp"]
        self.assertEqual(cs.with_inline(argv, "noauto"), ["mwcc", "-O4,p", "-inline", "noauto", "-c", "src/u.cpp"])
        self.assertEqual(cs.with_inline(argv, None), argv)

    def test_losses_name_what_gets_worse(self):
        base = unit_score(100.0, 90.0, data={".data": 100.0})
        self.assertEqual(cs.losses(base, unit_score(100.0, 95.0, data={".data": 100.0})), [])
        self.assertEqual(cs.losses(base, unit_score(99.0, 90.0, data={".data": 90.0})),
                         ["1 exact lost", "code -0.50", ".data -10.00"])
        self.assertEqual(cs.losses(base, None), ["does not compile"])

    def test_the_report_counts_and_groups_linked_units(self):
        grid = (("GC/1.3", "auto"), ("GC/1.3.2", "auto"))

        def row(path, lib, linked, *cells):
            unit = {"name": path, "metadata": {"source_path": "src/" + path, "complete": linked}}
            r = cs.Uniform(unit, lib, linked, cs.Variant("GC/1.3.2", "c++", unit_score(100.0)))
            r.cells = [cs.Variant(c, "c++", unit_score(p), inline=m) for (c, m), p in zip(grid, cells)]
            return r

        rows = [row("a.cpp", "game", True, 100.0, 100.0), row("b.cpp", "game", True, 90.0, 100.0),
                row("c.cpp", "sdk", True, 100.0, 90.0), row("d.cpp", "game", False, 100.0, 100.0)]
        lines = cs.render_uniform(rows, grid)
        self.assertEqual(lines[0], "GC/1.3 -inline auto      loses nothing in 3/4 units (linked 2/3)")
        self.assertIn("game: 1: b.cpp* (1 exact lost; code -10.00)", lines[1])
        self.assertIn("  1 GC/1.3 -inline auto | GC/1.3.2 -inline auto", "\n".join(lines))
        self.assertTrue(any(line.strip().startswith("1 GC/1.3.2 -inline auto ") and "game 1" in line for line in lines))


class BehindTest(unittest.TestCase):
    def test_commits_on_the_shared_branch_this_checkout_lacks(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock
        from tests.helpers import git
        from tools.dk.platforms.gc_mwcc import matchcmds
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git(root, "init", "-q", "-b", "shared")
            git(root, "config", "user.name", "t")
            git(root, "config", "user.email", "t@t.invalid")
            (root / "a").write_text("a")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "a")
            git(root, "branch", "mine")
            (root / "b").write_text("b")
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", "b")
            git(root, "switch", "-q", "mine")
            with mock.patch("tools.dk.core.landing.shared_ref", return_value="refs/heads/shared"):
                self.assertEqual(matchcmds.behind_shared(SimpleNamespace(root=root)), 1)


if __name__ == "__main__":
    unittest.main()
