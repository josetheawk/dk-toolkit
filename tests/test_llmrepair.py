"""llm-try repairs and ranking (llmrepair.py): declaration visibility, compile-error notes, shape-equal first."""

import unittest

from tools.dk.platforms.gc_mwcc import llmrepair as R
from tools.dk.platforms.gc_mwcc.llmtry import Context, Ollama, Settings, search

from .test_batch import have, real_compiler
from .test_llmtry import FakeOllama, fence

UNIT = """#include "a.h"
int f(int a) { return helper(a) + gT.x + K_TWO; }
struct T { int x; };
T gT = { 1 };
enum { K_ONE, K_TWO = 2 };
static int helper(int a) { return a * 2; }
int later(int);
"""


class VisibilityTest(unittest.TestCase):
    def test_items(self):
        kinds = [k for _s, _e, k in R.top_items(UNIT)]
        self.assertEqual(kinds, ["function", "type", "variable", "type", "function", "prototype"])

    def test_each_kind_is_made_visible_without_moving_a_definition(self):
        at = UNIT.index("int f(")
        new, label = R.visibility_fix(UNIT, at, "T")
        self.assertTrue(new.index("struct T") < new.index("int f("))
        self.assertEqual(new.count("struct T"), 1)                    # a type moves
        new, label = R.visibility_fix(UNIT, at, "gT")
        self.assertIn("extern T gT;\nint f(", new)                     # a global: an extern above, the definition stays
        self.assertIn("T gT = { 1 };", new)
        new, label = R.visibility_fix(UNIT, at, "helper")
        self.assertIn("static int helper(int a);\nint f(", new)       # a function: its prototype
        self.assertEqual(new.count("{ return a * 2; }"), 1)
        new, _ = R.visibility_fix(UNIT, at, "K_TWO")
        self.assertTrue(new.index("enum {") < new.index("int f("))
        self.assertIsNone(R.visibility_fix(UNIT, at, "nowhere"))
        self.assertIsNone(R.visibility_fix(UNIT, len(UNIT), "T"))     # declared before: nothing to do

    def test_error_names(self):
        self.assertEqual(R.undefined_names("line 3: x | undefined identifier 'gT'; line 4 | undefined identifier 'gT'"),
                         ["gT"])


class FeedbackTest(unittest.TestCase):
    def test_invented_names_and_prototypes(self):
        known = "struct S { int m10; };\nvoid fn_1(S* s, int n);\n"
        note = R.feedback("line 2: | undefined identifier 'unk14'; line 3: | function call 'fn_1(int)' does not match", known)
        self.assertIn("`unk14` is declared nowhere", note)
        self.assertIn("`fn_1` is declared as: void fn_1(S* s, int n);", note)
        self.assertEqual(R.feedback("line 2: | undefined identifier 'm10'", known), "")


def row(text, kind=None):
    return {"instruction": {"formatted": text}, **({"diff_kind": kind} if kind else {})}


class ShapeTest(unittest.TestCase):
    def test_register_only_rows_do_not_count(self):
        left = [row("lwz r3, 0x4(r31)", "DIFF_ARG_MISMATCH"), row("blr")]
        right = [row("lwz r4, 0x4(r30)", "DIFF_ARG_MISMATCH"), row("blr")]
        self.assertEqual(R.shape_rows(left, right), 0)
        right[0] = row("lwz r4, 0x8(r30)", "DIFF_ARG_MISMATCH")
        self.assertEqual(R.shape_rows(left, right), 1)
        self.assertEqual(R.shape_rows(left + [row("nop", "DIFF_INSERT")], right + [row("", "DIFF_INSERT")]), 2)

    def test_rank_key(self):
        a = {"compiled": True, "percent": 95.0, "shape_equal": True}
        b = {"compiled": True, "percent": 97.0, "shape_equal": False}
        self.assertGreater(R.rank_key(a), R.rank_key(b))
        self.assertLess(R.rank_key(a, "fuzzy"), R.rank_key(b, "fuzzy"))
        self.assertLess(R.rank_key({"compiled": False}), R.rank_key(b))


SHAPE = "int f(int a) {\n    return a + 2;\n}"
FUZZY = "int f(int a) {\n    return a + 3;\n}"


def evaluate(code, slot):
    if "a + 2" in code:
        return {"compiled": True, "percent": 90.0, "shape_equal": True, "devices": [], "exact": False, "rows": ["s"]}
    if "a + 3" in code:
        return {"compiled": True, "percent": 95.0, "shape_equal": False, "devices": [], "exact": False, "rows": ["f"]}
    return {"compiled": False, "error": "line 2: | undefined identifier 'unk8'", "devices": []}


class SearchRankTest(unittest.TestCase):
    def run_search(self, rank, replies, current="int f(int a) {\n    return a;\n}\n", repair=False):
        fake = FakeOllama(replies)
        rows = []
        try:
            st = Settings(model="fake", n=2, rounds=2, think=False, num_ctx=8192, prefill=False, rank=rank, repair=repair)
            ctx = Context("f", "    0  blr", "int g;\n", "", current)
            out = search(ctx, ctx.current, 50.0, [], evaluate, Ollama(fake.url), st, rows.append, jobs=1,
                         say=lambda *_: None)
        finally:
            fake.close()
        return out, fake.requests

    def test_shape_equal_feeds_the_next_round(self):
        out, reqs = self.run_search("shape", [fence(SHAPE), fence(FUZZY), fence(SHAPE), fence(SHAPE)])
        self.assertIn("a + 2", out.best_code)
        self.assertIn("a + 2", reqs[2]["messages"][-1]["content"])
        out, reqs = self.run_search("fuzzy", [fence(SHAPE), fence(FUZZY), fence(SHAPE), fence(SHAPE)])
        self.assertIn("a + 3", out.best_code)

    def test_failed_attempt_note_names_the_invented_member(self):
        _out, reqs = self.run_search("shape", [fence("int f(int a) {\n    return p->unk8;\n}")] * 4, current="",
                                     repair=True)
        self.assertIn("`unk8` is declared nowhere", reqs[2]["messages"][-1]["content"])
        _out, reqs = self.run_search("shape", [fence("int f(int a) {\n    return p->unk8;\n}")] * 4, current="")
        self.assertNotIn("declared nowhere", reqs[2]["messages"][-1]["content"])


@unittest.skipUnless(have(), "needs the project's compiler (./dk setup and a build)")
class GoldenTest(unittest.TestCase):
    def test_repairs_until_it_compiles(self):
        texts = []
        compile_ = real_compiler(texts)
        text = UNIT.replace('#include "a.h"\n', "")
        self.assertIsNone(compile_(text, "a"))
        done = []
        while compile_(text, "b") is None and len(done) < 4:
            fixed = None
            for name in ("helper", "gT", "T", "K_TWO"):
                if name not in done:
                    fixed = R.visibility_fix(text, text.index("int f("), name)
                    done.append(name)
                    break
            text = fixed[0]
        self.assertIsNotNone(compile_(text, "c"))
        self.assertLessEqual(text.index("struct T"), text.index("extern T gT"))   # the type moved above its use


if __name__ == "__main__":
    unittest.main()
