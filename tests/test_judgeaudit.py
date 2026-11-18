import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import judgeaudit
from tools.dk.platforms.gc_mwcc.judge import CLEAN, JUDGMENT, REJECT

DOC = """# Faithfulness

## `volatile`

| Where | What | Verdict |
| --- | --- | --- |
| `fe/A.cpp:10-12` (linked) | a flag | likely original |
| `race/B.cpp` | the settings pointer | device; the original probably re-read it |

## Pragmas
"""


class PragmaTest(unittest.TestCase):
    SRC = ("#include \"a.h\"\n#pragma force_active on\n#pragma optimization_level 0\n"
           "int f(void) { return 0; }\n#pragma dont_inline on\nint g(void) { return f(); }\n"
           "#pragma dont_inline reset\nint h(void) { return 1; }\n")

    def test_in_force_at(self):
        at = {n: self.SRC.index(f"int {n}(") for n in "fgh"}
        self.assertEqual([p[:2] for p in judgeaudit.pragmas_at(self.SRC, at["f"])], [("optimization_level", "0")])
        self.assertEqual(sorted(p[0] for p in judgeaudit.pragmas_at(self.SRC, at["g"])),
                         ["dont_inline", "optimization_level"])
        self.assertEqual([p[0] for p in judgeaudit.pragmas_at(self.SRC, at["h"])], ["optimization_level"])
        pushed = "#pragma push\n#pragma auto_inline off\n#pragma pop\nint k;"
        self.assertEqual(judgeaudit.pragmas_at(pushed, pushed.index("int k")), [])
        closed = "#pragma dont_inline on\nint g(void);\n#pragma dont_inline off\nint k;"   # off closes the on region
        self.assertEqual(judgeaudit.pragmas_at(closed, closed.index("int k")), [])
        self.assertEqual([p[:2] for p in judgeaudit.pragmas_at(closed, closed.index("int g"))], [("dont_inline", "on")])


class CategoryTest(unittest.TestCase):
    def cat(self, *flags):
        return judgeaudit.category_of([judgeaudit.rule_of(f) for f in flags])

    def test_rules_and_categories(self):
        self.assertEqual(self.cat(), CLEAN)
        self.assertEqual(self.cat("volatile, likely original (faithfulness.md)"), judgeaudit.LIKELY_ORIGINAL)
        self.assertEqual(self.cat("file-level #pragma optimization_level 0 (line 3)"), judgeaudit.LIKELY_ORIGINAL)
        self.assertEqual(self.cat("volatile, likely original (faithfulness.md)", "adds a cast to a raw pointer type"),
                         JUDGMENT)
        self.assertEqual(self.cat("struct packing #pragma pack (push, 1) (line 4)"), JUDGMENT)
        self.assertEqual(self.cat("compiled under #pragma dont_inline on (line 5)"), REJECT)
        self.assertEqual(self.cat("new local t only copies a"), REJECT)
        self.assertEqual(self.cat("something new"), REJECT)                    # unknown flags reject
        self.assertEqual(judgeaudit.rule_of("cast to its own type changes the code (A*)x"),
                         ("same-type cast that changes the code", True))


class TriageTest(unittest.TestCase):
    TABLE = ("| Unit | Function | Rule | Verdict | Reason |\n|---|---|---|---|---|\n"
             "| u/A | `f` | copy-only local | **device** | a parameter copy |\n"
             "| u/A | `g` | unused local, raw pointer cast | allowed | the target builds it |\n"
             "| u/B | `h` | hard-coded offset cast | false positive | a size |\n"
             "| u/B | `k` | comma operator | open | unexplained |\n")

    def rec(self, unit, fn, *flags):
        r = {"unit": unit, "function": fn, "flags": list(flags)}
        judgeaudit.apply_triage(r, judgeaudit.triage_rows(self.TABLE))
        return r["category"], r["quarantine"]

    def test_rows(self):
        rows = judgeaudit.triage_rows(self.TABLE)
        self.assertEqual(rows[("u/A", "f", "copy-only local")], "device")
        self.assertEqual(rows[("u/A", "g", "raw pointer cast")], "allowed")
        self.assertEqual(rows[("u/B", "h", "hard-coded offset cast")], "false positive")
        self.assertEqual(len(rows), 5)

    def test_verdicts_decide_the_quarantine(self):
        self.assertEqual(self.rec("u/A", "f", "new local t only copies a"), (REJECT, True))
        self.assertEqual(self.rec("u/A", "g", "new local x is never read"), (judgeaudit.LIKELY_ORIGINAL, False))
        self.assertEqual(self.rec("u/B", "h", "hard-coded offset cast", "adds a cast to a raw pointer type"),
                         (JUDGMENT, False))                     # raw pointer casts alone stay in training
        self.assertEqual(self.rec("u/B", "k", "adds a comma operator"), (REJECT, True))
        self.assertEqual(self.rec("u/C", "f", "new local t only copies a"), (REJECT, True))      # another unit
        self.assertEqual(self.rec("u/C", "m", "adds 1 volatile"), (JUDGMENT, True))              # not triaged
        self.assertEqual(self.rec("u/C", "n"), (CLEAN, False))

    def test_the_export_reads_the_quarantine_list(self):
        import json
        import tempfile
        from pathlib import Path
        from tools.dk.platforms.gc_mwcc.llmdata import quarantined
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "judge-audit.json"
            p.write_text(json.dumps({"flagged": [{"function": "a"}, {"function": "b"}], "quarantine": ["b"]}))
            self.assertEqual(quarantined(p), {"b"})


class VolatileRowsTest(unittest.TestCase):
    def test_rows_and_lookup(self):
        rows = judgeaudit.volatile_rows(DOC)
        self.assertEqual(rows, [("fe/A.cpp", {10, 11, 12}, True), ("race/B.cpp", None, False)])
        self.assertTrue(judgeaudit.volatile_original(rows, "src/game/fe/A.cpp", [11]))
        self.assertFalse(judgeaudit.volatile_original(rows, "src/game/fe/A.cpp", [20]))
        self.assertFalse(judgeaudit.volatile_original(rows, "src/game/race/B.cpp", [5]))


class SameTypeCastTest(unittest.TestCase):
    def test_bisects_to_the_same_type_casts(self):
        src = "struct A; struct B;\nA* a; B* b;\nvoid f(void) { use((A*)a); use((A*)b); use((B*)b); }\n"
        body = cbody.parse_body(src, src.index("{", src.index("void f")))
        casts = judgeaudit._casts(src, body)
        self.assertEqual([(t, op) for _, t, op in casts], [("A*", "a"), ("A*", "b"), ("B*", "b")])
        same = {("A*", "a"), ("B*", "b")}
        calls = []

        def compiles(text):                     # a fake compiler: DkNotSame<T, T> has no ::t
            calls.append(text)
            return not any(f"DkNotSame<{t}, __typeof__({op})>" in text for t, op in same) and \
                not any(f"*({t}**)0 = (__typeof__({op})*)0" in text for t, op in {("A*", "b")})
        hits = judgeaudit.same_type_casts(src, casts, compiles, cplusplus=True)
        self.assertEqual(sorted((t, op) for _, t, op in hits), sorted(same))
        self.assertEqual(judgeaudit.same_type_casts(src, [], compiles, True), [])


if __name__ == "__main__":
    unittest.main()
