"""dk inline-cost: a callee's distance from MWCC's auto-inline limit, measured with probes."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import inlinecost
from tools.dk.platforms.gc_mwcc.elf import Elf

REPO = Path(__file__).resolve().parents[1]
FLAGS = ["-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware", "-Cpp_exceptions", "off",
         "-RTTI", "off", "-enum", "int", "-lang=c++"]


def toy(statements: int, inline: str = "") -> str:
    body = "".join(f"    p[{i}] = x;\n" for i in range(statements))
    return (f"{inline}int callee(int* p, int x) {{\n{body}    return x;\n}}\n"
            "int caller(int* p, int x) {\n    return callee(p, x) + 1;\n}\n")


class ToyProbe:
    """dk inline-cost's probe interface over a toy file compiled with one compiler."""

    def __init__(self, text: str, version: str):
        self.text, self.version, self.callee = text, version, "callee__FPii"
        self.span = (text.index("int callee"), text.index("}\n") + 1)

    def expression_statements(self):
        return inlinecost.expression_statements(self.text, self.span)

    def compile(self, count=0, probe=inlinecost.STATEMENT, blank=()):
        work = Path(tempfile.mkdtemp())
        try:
            (work / "t.cpp").write_text(inlinecost.edited(self.text, self.span, count, probe, blank))
            mwcc = REPO / "build" / "compilers" / self.version / "mwcceppc.exe"
            proc = subprocess.run([str(REPO / "build" / "tools" / "wibo"), str(mwcc), *FLAGS, "-c", "t.cpp", "-o", "t.o"],
                                  cwd=work, capture_output=True)
            return Elf.parse((work / "t.o").read_bytes()) if proc.returncode == 0 else None
        finally:
            shutil.rmtree(work, ignore_errors=True)


def have(version: str) -> bool:
    return (REPO / "build" / "compilers" / version / "mwcceppc.exe").exists() and (REPO / "build" / "tools" / "wibo").exists()


class EditsTest(unittest.TestCase):
    def test_probes_and_blanks(self):
        text = toy(2)
        span = (0, text.index("}\n") + 1)
        self.assertEqual(len(inlinecost.expression_statements(text, span)), 2)
        out = inlinecost.edited(text, span, 2, blank=inlinecost.expression_statements(text, span)[1:])
        self.assertIn("{ ((void)0); ((void)0);\n    p[0] = x;\n    ;\n    return x;", out)

    def test_named_matches_the_written_name(self):
        self.assertTrue(inlinecost.named("fn_80001234__5ClassFv", "fn_80001234"))
        self.assertTrue(inlinecost.named("fn_80001234__5ClassFv", "fn_80001234__5ClassFv"))
        self.assertFalse(inlinecost.named("fn_80001234__5ClassFv", "fn_80001238"))


@unittest.skipUnless(have("GC/1.3") and have("GC/1.3.2"), "needs the project's compilers (./dk setup and a build)")
class MeasureTest(unittest.TestCase):
    def test_slack_of_an_inlined_callee(self):
        # GC 1.3 auto-inlines up to 14 statements (plus the return): 10 leave 4
        got = inlinecost.measure(ToyProbe(toy(10), "GC/1.3"), "caller__FPii")
        self.assertEqual(got, ["ours inlines it: 4 more statement(s) fit; with 5 more it stays out of line"])

    def test_statements_over_the_limit(self):
        got = inlinecost.measure(ToyProbe(toy(17), "GC/1.3"), "caller__FPii")
        self.assertEqual(got, ["ours calls it: 3 statement(s) over the limit (with its last 3 expression "
                               "statement(s) blanked it is inlined)"])
        got = inlinecost.measure(ToyProbe(toy(31), "GC/1.3.2"), "caller__FPii")      # 1.3.2: 29
        self.assertIn("ours calls it: 2 statement(s) over", got[0])

    def test_an_explicit_inline_is_judged_by_size(self):
        text = toy(40, "inline ")
        self.assertTrue(inlinecost.explicitly_inline(text, (0, len(text))))
        self.assertFalse(inlinecost.explicitly_inline(toy(40), (0, 10)))
        got = inlinecost.measure(ToyProbe(text, "GC/1.3.2"), "caller__FPii", explicit=True)
        self.assertRegex(got[0], r"^ours inlines it \(explicitly inline: judged by size\): \d+ more probe store")
        got = inlinecost.measure(ToyProbe(text, "GC/1.3"), "caller__FPii", explicit=True)
        self.assertIn("this compiler inlines it at any size", got[0])


def doc(checks, small=(), cap=512, totals=(300,)):
    return {"schema": "inline.v1", "compiler": "X", "limits": {"stmts": 30, "locals": 1024, "max_size": 512, "total": 7000},
            "functions": [{"name": "caller", "link": "", "obj": 1, "max": cap, "totals": list(totals), "checks": checks}],
            "small": list(small)}


def check(callee="callee", rnd=0, explicit=False, auto=True, est=None, cap=512, can=6, stmts=3):
    c = {"callee": callee, "link": "", "obj": 2, "round": rnd, "explicit": explicit, "auto": auto, "stmts": stmts, "can": can}
    if est is not None:
        c.update(est=est, max=cap)
    return c


class ExplainTest(unittest.TestCase):
    """The rule's verdicts from an oracle document (synthetic numbers)."""

    def test_inlined_auto_gives_both_slacks(self):
        got = inlinecost.explain(doc([check(est=40)], [{"name": "callee", "link": "", "obj": 2, "stmts": 12, "locals": 0}]),
                                 "caller__Fv", "callee__Fi")
        self.assertEqual(got[0], "inlined: auto-inline, cost 40 against the caller's limit 512, 472 unit(s) of slack")
        self.assertIn("19 more statement(s) (12 statement(s) of 30)", got[1])
        self.assertIn("473 more cost unit(s)", got[1])

    def test_over_the_cost_limit(self):
        got = inlinecost.explain(doc([check(explicit=True, auto=False, est=600)]), "caller", "callee")
        self.assertEqual(got[0], "not inlined: inline, cost 600 against the caller's limit 512, 88 over")

    def test_halved_limit_names_the_caller_budget(self):
        got = inlinecost.explain(doc([check(est=200, cap=128)], cap=128, totals=(7300,)), "caller", "callee")
        self.assertIn("72 over; the caller's limit is 128, not 512: its own size estimate is over 7000 (7300)", got[0])

    def test_not_a_candidate_reasons(self):
        many = {"name": "callee", "link": "", "obj": 2, "stmts": 38, "locals": 0}
        got = inlinecost.explain(doc([{"callee": "callee", "link": "", "obj": 2, "round": 0, "explicit": False,
                                       "auto": False}], [many]), "caller", "callee")
        self.assertIn("38 statement(s) as lowered, over the auto-inline limit of 30 by 8", got[0])
        got = inlinecost.explain(doc([{"callee": "callee", "link": "", "obj": 2, "round": 0, "explicit": False,
                                       "auto": False}]), "caller", "callee")
        self.assertIn("only a body defined before the caller", got[0])

    def test_last_round_inline_of_an_oversized_explicit(self):
        late = check(rnd=7, explicit=True, auto=False)
        got = inlinecost.explain(doc([check(rnd=1, explicit=True, auto=False, est=514), late]), "caller", "callee")
        self.assertTrue(got[0].startswith("inlined in round 7 only"))

    def test_records_match_by_link_name_first(self):
        d = doc([dict(check(est=5), link="callee__1AFv"), dict(check(est=9), link="callee__1BFv", obj=3)])
        self.assertEqual(inlinecost.callee_checks(d["functions"][0], "callee__1BFv")[0]["est"], 9)
        self.assertEqual(len(inlinecost.callee_checks(d["functions"][0], "callee")), 2)


if __name__ == "__main__":
    unittest.main()
