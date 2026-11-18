"""The RL reward gate (rlgate) and the in-span device detectors it shares with review(): synthetic code only."""
import hashlib
import unittest
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import judge, llmagent, rlgate

BEFORE = "struct S { int x; };\nint Func(int a, int* p) {\n    int r = a;\n    return r;\n}\n"


def after(body: str) -> str:
    return "struct S { int x; };\nint Func(int a, int* p) {\n" + body + "}\n"


class SpanDeviceTest(unittest.TestCase):
    def flags(self, body: str, before: str = BEFORE) -> tuple[list[str], list[str]]:
        return judge.span_device_flags(before, judge.bodies(before, before, "Func")[2], after(body),
                                       judge.bodies(after(body), after(body), "Func")[2])

    def test_each_device_is_found_when_added(self):
        cases = {
            "preprocessor": "    int r = a;\n#if 1\n    r++;\n#endif\n    return r;\n",
            "false-condition": "    int r = a;\n    if (0) {\n        r++;\n    }\n    return r;\n",
            "self-assign": "    int r = a;\n    r = r;\n    return r;\n",
            "identity": "    int r = a + 0;\n    return r * 1;\n",
            "void0": "    int r = a;\n    (void)0;\n    return r;\n",
            "not-not": "    int r = !!a;\n    return r;\n",
            "local-type-methods": "    struct L { int f() { return 1; } };\n    int r = a;\n    return r;\n",
            "address-pun": "    int r = ((S*)&p[1])->x;\n    return r;\n",
        }
        for kind, body in cases.items():
            reject, judgment = self.flags(body)
            flag, rej = judge.SPAN_DEVICES[kind]
            self.assertIn(flag, reject if rej else judgment, kind)

    def test_natural_code_is_not_flagged(self):
        body = ("    int r = a - 0x10;\n    r = a * 10 + p[0];\n    r += 2;\n    if (r != !a) {\n        r = p[1];\n    }\n"
                "    do {\n        r--;\n    } while (0);\n    struct T { int y; } t;\n    t.y = r;\n    return t.y;\n")
        self.assertEqual(self.flags(body), ([], []))

    def test_one_already_in_the_source_is_not_new(self):
        src = after("    int r = a + 0;\n    return r;\n")
        self.assertEqual(self.flags("    int r = a + 0;\n    return r;\n", src), ([], []))

    def test_review_rejects_the_reject_kinds_and_sends_the_others_for_a_look(self):
        self.assertEqual(judge.review(BEFORE, after("    int r = a;\n    (void)0;\n    return r;\n"), "Func")[0], judge.REJECT)
        self.assertEqual(judge.review(BEFORE, after("    int r = !!a;\n    return r;\n"), "Func")[0], judge.JUDGMENT)

    def test_exact_review_can_let_a_placeholder_through(self):
        cand = after("    int t0 = a * 3;\n    return t0 + p[0];\n")
        blocking, category, flags = judge.exact_review(BEFORE, cand, "Func", placeholders_ok=True)
        self.assertEqual((blocking, category), ([], judge.REJECT))
        self.assertTrue(any("placeholder" in f for f in flags))
        self.assertTrue(judge.exact_review(BEFORE, cand, "Func")[0])


def fake_measure(canon):
    """measure(text) -> an Eval-like object whose code is the hash of canon(text); a same-type probe
    (`__typeof__`) does not compile: no operand here has its cast's type."""
    def measure(text):
        if "__typeof__" in text:
            return None
        return SimpleNamespace(percent=100.0, code=hashlib.sha1("".join(canon(text).split()).encode()).hexdigest())
    return measure


class GateTest(unittest.TestCase):
    def test_a_clean_exact_pays_and_a_device_does_not(self):
        self.assertEqual(rlgate.gate(BEFORE, after("    return a;\n"), "Func", "exact").reward, 1.0)
        g = rlgate.gate(BEFORE, after("    int r = a;\n    r = r;\n    return r;\n"), "Func", "exact")
        self.assertLessEqual(g.reward, 0.0)
        self.assertEqual(g.category, judge.REJECT)

    def test_judgment_pays_nothing_and_goes_to_audit(self):
        g = rlgate.gate(BEFORE, after("    int r = !!a;\n    return r;\n"), "Func", "exact")
        self.assertEqual((g.reward, g.audit), (0.0, True))

    def test_tiers_and_malformed(self):
        self.assertEqual(rlgate.gate(BEFORE, after("    return a;\n"), "Func", "rows").reward, rlgate.REWARD["rows"])
        self.assertLess(rlgate.gate(BEFORE, BEFORE, "Func", "malformed").reward, 0.0)
        e = lambda p, s, r: SimpleNamespace(percent=p, shape_rows=s, rows=r)
        self.assertEqual(rlgate.tier(e(90, 3, 5), e(100.0, 0, 0)), "exact")
        self.assertEqual(rlgate.tier(e(90, 3, 5), e(91, 2, 6)), "shape_rows")
        self.assertEqual(rlgate.tier(e(90, 3, 5), e(91, 3, 4)), "rows")
        self.assertEqual(rlgate.tier(e(90, 3, 5), e(95, 4, 1)), "none")
        self.assertEqual(rlgate.tier(e(90, 3, 5), None), "none")

    def test_deletion_twin_strips_a_byte_neutral_construct(self):
        cand = after("    int r = a;\n    (void)0;\n    return r;\n")
        g = rlgate.gate(BEFORE, cand, "Func", "exact", measure=fake_measure(lambda t: t.replace("(void)0;", "")))
        self.assertEqual((g.reward, g.twins.get("deletion")), (1.0, "stripped"))
        self.assertNotIn("(void)0", g.text)
        g = rlgate.gate(BEFORE, cand, "Func", "exact", measure=fake_measure(lambda t: t))     # the construct matters
        self.assertEqual((g.reward, g.twins.get("deletion")), (0.0, "changes the code"))

    def test_declarative_twin_catches_a_cast_standing_in_for_a_declaration(self):
        before = "int Func(int* p) {\n    unsigned int n = 0;\n    return p[n];\n}\n"
        cand = "int Func(int* p) {\n    unsigned int n = 0;\n    return p[(int)n];\n}\n"
        twins = rlgate.declarative_twins(before, cand, "Func")
        self.assertEqual([(v, t) for v, t, _ in twins], [("n", "int")])
        self.assertIn("int n = 0;", twins[0][2])
        self.assertIn("p[n]", twins[0][2])
        canon = lambda t: t.replace("unsigned int n", "int n").replace("(int)n", "n")
        g = rlgate.gate(before, cand, "Func", "exact", measure=fake_measure(canon), declarative=True)
        self.assertEqual((g.reward, g.audit), (rlgate.REWARD["declarative-twin"], True))
        g = rlgate.gate(before, cand, "Func", "exact", measure=fake_measure(lambda t: t), declarative=True)  # differs:
        self.assertEqual((g.reward, g.audit), (0.0, True))                                    # still a look
        self.assertEqual(g.twins["declarative"][0]["result"], "other code")
        self.assertTrue(any(f.startswith("adds an integer cast (int)n") for f in g.flags))
        narrowing = cand.replace("(int)n", "(unsigned char)n")                               # a real conversion
        self.assertEqual(rlgate.gate(before, narrowing, "Func", "exact").reward, 1.0)

    def test_a_same_type_cast_the_probe_cannot_check_still_gets_a_look(self):
        cand = after("    int r = (int)a;\n    return r;\n")
        g = rlgate.gate(BEFORE, cand, "Func", "exact", measure=fake_measure(lambda t: t))     # the probe never compiles
        self.assertEqual(g.reward, 0.0)
        self.assertTrue(any(f.startswith("redundant cast to its own type") for f in g.flags))

    def test_declarative_twin_retypes_a_parameter(self):
        before = "int Func(int* p, unsigned int n) {\n    return p[n];\n}\n"
        cand = "int Func(int* p, unsigned int n) {\n    return p[(int)n];\n}\n"
        (_v, _t, twin), = rlgate.declarative_twins(before, cand, "Func")
        self.assertIn("int Func(int* p, int n)", " ".join(twin.split()).replace("( ", "("))
        self.assertIn("p[n]", twin)


class ActionTest(unittest.TestCase):
    def test_allow_list_apply_and_repeats(self):
        A = llmagent.Action
        self.assertIsNone(rlgate.check_action(A("structured", transform="else-return", anchor=3), "x", set()))
        self.assertIn("not in the shape table", rlgate.check_action(A("structured", transform="cast-call", anchor=3), "x", set()))
        self.assertNotIn("inline-helper", rlgate.allowed_transforms())
        self.assertEqual(rlgate.check_action(A("edit", old="a", new="b"), None, set()), "the action does not apply")
        self.assertEqual(rlgate.check_action(A("edit", old="a", new="b"), "s", {"s"}), "repeats a state already seen")
        self.assertIsNone(rlgate.check_action(A("stop", label="no-idea"), None, set()))
        self.assertEqual(rlgate.check_action(A(None, error="no action in the reply"), None, set()), "no action in the reply")


class RenderTest(unittest.TestCase):
    def test_no_oracle_anywhere_and_the_sft_score_line(self):
        sc = llmagent.Score(True, 97.5, (1, 2), ["  12  addi r3, r3, 1   arg  addi r4, r4, 1"], shape_rows=2, right=3, known=5)
        ex = {"function": "Other", "before": "int Other() {\n    return 1;\n}", "after": "int Other() {\n    return 2;\n}",
              "rows": [], "diff_class": "shape", "before_percent": 90.0, "fix_kind": ["literal"],
              "oracle": {"explain": "v1 coloured r4, target r3", "reach": "reachable"}}
        msgs = rlgate.render("Func", "int Func() {\n    return 1;\n}", sc, [("edit `a` -> `b`", "90.00%, 3 rows shown differ")], [ex, ex], 1)
        user = msgs[1]["content"]
        self.assertNotIn("Oracle:", user)
        self.assertIn("## Func now: 97.50%, 1 rows shown differ\n", user)
        self.assertEqual(user.count("### Example"), 1)
        want = llmagent.build_prompt("Func", "int Func() {\n    return 1;\n}", llmagent.Score(True, 97.5, (), sc.rows), "",
                                     [{**ex, "oracle": None}], [("edit `a` -> `b`", "90.00%, 3 rows shown differ")], "both")
        self.assertEqual(msgs, want)
        self.assertNotIn("### Example", rlgate.render("Func", "int Func() {}", sc, [], [ex], 0)[1]["content"])

    def test_outcome_lines_use_the_sft_words(self):
        new, best = llmagent.Score(True, 99.0, (), ["x"], shape_rows=1), llmagent.Score(True, 98.0, (), ["x", "y"])
        self.assertEqual(rlgate.outcome_line(new, best, True, True), "99.00%, 1 rows shown differ (kept: the new best)")
        self.assertEqual(rlgate.outcome_line(best, new, False, False),
                         "98.00%, 2 rows shown differ (worse than the best 99.00%: reverted)")


if __name__ == "__main__":
    unittest.main()
