import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.dk.platforms.gc_mwcc import stacked


class StackedTest(unittest.TestCase):
    def test_table_has_the_loop_rows(self):
        from tools.dk.platforms.gc_mwcc.rewrites import BY_NAME
        for r in stacked.TABLE:
            self.assertIn(r, BY_NAME)
        self.assertIn("for-init-list", stacked.TABLE)
        self.assertIn("iv-alias", stacked.TABLE)

    def test_valleys(self):
        self.assertEqual(stacked.valleys([98.0, 99.0, 100.0]), 0)
        self.assertEqual(stacked.valleys([98.0, 97.5, 100.0]), 1)
        self.assertEqual(stacked.valleys([99.0, 99.0, 99.0, 100.0]), 2)

    def test_never_functions_and_leak(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "datasets" / "shape-fixes" / "v0").mkdir(parents=True)
            (d / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl").write_text(json.dumps({"function": "fA"}) + "\n")
            (d / "heldout-split.json").write_text(json.dumps({"dev": [{"function": "fB"}], "final": ["fC"]}))
            with mock.patch("tools.dk.platforms.gc_mwcc.llmtry.data_dir", return_value=d):
                got = stacked.never_functions(None)
        self.assertEqual(got, {"fA": "shape-fixes", "fB": "heldout-dev", "fC": "heldout-final"})
        fz = {"functions": {"fG": "gate-lora2"}, "units": {}, "shapes": {}, "hashes": {}}
        self.assertEqual(stacked.leak({"function": "fB", "unit": "u"}, fz, got), "heldout-dev")
        self.assertEqual(stacked.leak({"function": "fG", "unit": "u"}, fz, got), "frozen")


if __name__ == "__main__":
    unittest.main()


BEFORE = "void f(int a) {\n    int x;\n    int y;\n    x = a;\n    if (x) {\n        y = 1;\n    } else {\n        y = 2;\n    }\n" + \
    "".join(f"    g({i});\n" for i in range(20)) + "    h(x, y);\n    k(x);\n}"
AFTER = BEFORE.replace("x = a;\n    if (x)", "x = a;\n    if (x != 0)").replace("    h(x, y);\n", "    h(y, x);\n")


class WindowTest(unittest.TestCase):
    def test_depth_weights(self):
        self.assertEqual(stacked.depth_weights("2-3"), ((2, 67), (3, 33)))
        self.assertEqual(stacked.depth_weights("3-5"), ((3, 50), (4, 33), (5, 17)))
        self.assertEqual(stacked.depth_weights("4"), ((4, 100),))
        with self.assertRaises(ValueError):
            stacked.depth_weights("5-3")

    def test_hunks_apply_to_the_exact_text(self):
        patch = stacked.hunks(BEFORE, AFTER)
        self.assertEqual(len(patch), 2)
        self.assertEqual(stacked.apply_hunks(BEFORE, patch), AFTER)
        for h in patch:
            self.assertEqual(BEFORE.count(h["search"]), 1)
        self.assertIsNone(stacked.hunks(BEFORE, BEFORE))

    def test_hunks_widen_a_repeated_line_until_unique(self):
        before = "void f() {\n    a();\n    b();\n    a();\n    b();\n}"
        after = "void f() {\n    a();\n    b();\n    a();\n    c();\n}"
        patch = stacked.hunks(before, after)
        self.assertEqual(stacked.apply_hunks(before, patch), after)
        self.assertEqual(len(patch), 1)
        self.assertIn("\n", patch[0]["search"])

    def test_window_shows_the_head_and_the_regions_with_context(self):
        patch = stacked.hunks(BEFORE, AFTER)
        win, shown = stacked.window(BEFORE, patch)
        self.assertTrue(win.startswith("void f(int a) {\n    int x;\n    int y;\n"))
        self.assertIn("if (x) {", win)
        self.assertIn("h(x, y);", win)
        self.assertIn("// ... lines ", win)
        self.assertNotIn("g(10);", win)
        self.assertEqual(shown[0][0], 1)
        for h in patch:                                     # every hunk line is shown
            self.assertTrue(any(a <= h["lines"][0] and h["lines"][1] <= b for a, b in shown), (h["lines"], shown))

    def test_render_window_and_kind_table(self):
        patch = stacked.hunks(BEFORE, AFTER)
        win, shown = stacked.window(BEFORE, patch)
        p = {"function": "f", "before": BEFORE, "before_percent": 97.5, "rows": ["  ...", "   10  li r0, 0x1   arg  li r0, 0x2"],
             "window": win, "patch": patch}
        msgs = stacked.render_window(p)
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
        self.assertIn("// ... lines ", msgs[1]["content"])
        self.assertNotIn("g(10);", msgs[1]["content"])
        self.assertEqual(msgs[2]["content"].count("<<<<<<< SEARCH"), 2)
        self.assertEqual(stacked.apply_hunks(BEFORE, patch), AFTER)
        tbl = stacked.kind_table([{"fix_kind": ["operator"]}, {"fix_kind": ["operator", "literal"]}], [{"fix_kind": ["literal"]}])
        self.assertEqual(tbl["operator"], {"puzzles": 2, "puzzle_share": 1.0, "real": 0, "real_share": 0.0})
        self.assertEqual(tbl["literal"]["real_share"], 1.0)
        self.assertIn("control-flow", tbl)


class FoldDrawTest(unittest.TestCase):
    def test_fold_draw_reaches_build_path(self):
        calls = []

        def fake_build_path(fn, s0, depth, rng, stats, fold_first=False):
            calls.append(fold_first)
            return []
        fn = mock.MagicMock()
        fn.defn.return_value = "void f() {}"
        fn.measure.return_value = {"compiled": True, "percent": 100.0}
        fn.compiles = 0
        f = {"u": {"metadata": {"source_path": "x.cpp"}}, "function": "f", "unit": "u", "size": 64, "asm_shape": ""}
        cfg = mock.MagicMock()
        cfg.root.__truediv__.return_value.read_text.return_value = "void f() {}"
        with mock.patch.object(stacked.ss, "Fn", return_value=fn), mock.patch.object(stacked.ss, "build_path", fake_build_path), \
                mock.patch.object(stacked.ss, "work_dir", return_value=Path("/nonexistent/x")), mock.patch("shutil.rmtree"):
            stacked.one_function(cfg, f, 1, 2, Path("/nonexistent"), ((2, 100),), False, 1.0)
            self.assertTrue(calls and all(calls))
            calls.clear()
            stacked.one_function(cfg, f, 1, 2, Path("/nonexistent"), ((2, 100),), False, 0.0)
            self.assertTrue(calls and not any(calls))
