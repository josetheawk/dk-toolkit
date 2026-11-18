"""The device suite: synthetic before/after pairs (tests/fixtures/devices, no game bytes) for the devices the
grind logs show a compiler-rewarded search finding, and natural rewrites that must stay clean.

  offset-cast        a field reached by an invented byte offset
  empty-cases        invented empty `case` labels that only shape the compare tree
  copied-inline      an inline helper's body written out by hand at the call
  inline-trick       pragmas and attributes that force or forbid inlining, `inline` added to a callee
  self-cast          a cast of a variable to the type it already has
  placeholder-temp   a new local named `t0`, `tmp1`, `var3`
  padding-local      locals that only take stack space
  clean              controls

A device case passes when the pipeline flags it: `review` (what the grind, the audit and the pre-review run)
or the strict `judge`; the cases marked `reject` must be rejected outright, not only sent for a look.
"""
import json
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import judge, judgeaudit

DIR = Path(__file__).resolve().parent / "fixtures" / "devices"
CASES = json.loads((DIR / "devices.json").read_text())


def verdict(case: dict) -> tuple[str, list[str], list[str]]:
    before = (DIR / f"{case['name']}.before.cpp").read_text()
    after = (DIR / f"{case['name']}.after.cpp").read_text()
    category, flags = judge.review(before, after, case["symbol"])
    return category, flags, judge.judge(before, after, None, None, None, None, case["symbol"]).reasons


class DeviceSuiteTest(unittest.TestCase):
    def test_every_device_is_flagged_and_no_control_is(self):
        missed = []
        for c in CASES:
            category, flags, strict = verdict(c)
            ok = {"reject": category == "reject", "flag": category != "clean" or bool(strict),
                  "clean": category == "clean" and not strict, "not-reject": category != "reject" and not strict}[c["expect"]]
            if not ok:
                missed.append(f"{c['name']}: expected {c['expect']}, review {category} {flags}, judge {strict}")
        self.assertEqual(missed, [])

    def test_the_suite_covers_each_device_kind_with_several_cases(self):
        kinds = {}
        for c in CASES:
            kinds[c["device"]] = kinds.get(c["device"], 0) + 1
        for kind in ("offset-cast", "empty-cases", "copied-inline", "inline-trick", "self-cast", "placeholder-temp",
                     "padding-local"):
            self.assertGreaterEqual(kinds.get(kind, 0), 4, kind)
        self.assertGreaterEqual(kinds["clean"], 6)

    def test_fixtures_are_synthetic(self):
        for p in DIR.glob("*.cpp"):
            self.assertNotRegex(p.read_text(), r"fn_[0-9A-Fa-f]{8}|lbl_[0-9A-Fa-f]{8}|0x8[0-9A-F]{7}", p.name)

    def test_the_audit_knows_the_new_flags(self):
        self.assertEqual(judgeaudit.rule_of("new local t0 has a placeholder name"), ("placeholder-named local", True))
        self.assertEqual(judgeaudit.rule_of("hand-copied inline body of ClampPct"), ("hand-copied inline body", False))
        self.assertEqual(judgeaudit.rule_of("adds 1 inline attribute"), ("inline attribute", True))
        self.assertEqual(judgeaudit.rule_of("redundant cast to its own type (int)i"),
                         ("same-type cast without effect", False))


class DetectorTest(unittest.TestCase):
    PRE = "inline int Add3(int a, int b) {\n    int s = a + b;\n    s += 3;\n    return s;\n}\n"

    def copied(self, before: str, after: str) -> list[str]:
        return judge.copied_inlines(self.PRE + before, self.PRE + after, "F")

    def test_copied_inline_needs_a_real_helper_and_an_added_copy(self):
        call = "int F(int x, int y) {\n    return Add3(x, y);\n}\n"
        pasted = "int F(int x, int y) {\n    int q = x + y;\n    q += 3;\n    return q;\n}\n"
        self.assertEqual(self.copied(call, pasted), ["hand-copied inline body of Add3"])
        self.assertEqual(self.copied(pasted, pasted), [])                       # already there: not the candidate's
        self.assertEqual(self.copied(call, "int F(int x, int y) {\n    int q = x + y;\n    return q;\n}\n"), [])
        one = "inline int Get(int a) {\n    return a + 1;\n}\n"                  # a one-statement helper is just an expression
        self.assertEqual(judge.copied_inlines(one + "int F(int x) {\n    return Get(x);\n}\n",
                                              one + "int F(int x) {\n    return x + 1;\n}\n", "F"), [])

    def test_placeholder_names(self):
        for bad in ("t0", "t12", "tmp", "tmp1", "temp3", "var2", "v0", "val1", "unk4"):
            self.assertTrue(judge.PLACEHOLDER_LOCAL.fullmatch(bad), bad)
        for fine in ("t", "v", "total", "count", "index", "t0x", "tick", "value", "speed2"):
            self.assertFalse(judge.PLACEHOLDER_LOCAL.fullmatch(fine), fine)

    def test_a_placeholder_already_in_the_function_is_not_new(self):
        src = "int F(int a) {\n    int t0 = a * 3;\n    return t0;\n}\n"
        self.assertEqual(judge.judge(src, src.replace("return t0", "return t0 + 0"), None, None, None, None, "F").reasons, [])


if __name__ == "__main__":
    unittest.main()
