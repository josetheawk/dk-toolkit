import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import rewrites
from tools.dk.platforms.gc_mwcc.rewrites import BY_NAME, RULES, Site, candidates, ranked, symptoms


def site(body: str, cplusplus: bool = True, focus=None) -> Site:
    src = "void f(int a, int b, float x, S* p) {\n" + body + "}\n"
    return Site(src, cbody.parse_body(src, src.index("{")), focus, cplusplus)


def outs(rule: str, body: str, **kw) -> list[str]:
    s = site(body, **kw)
    return [new[new.index("{") + 2:-3] for _, _, new in candidates(s, [BY_NAME[rule]])]


def ins(mnemonic, *args, diff=None, line=None):
    parts = [{"opcode": {"mnemonic": mnemonic}}] + [{"arg": {"opaque": a}} for a in args]
    e = {"instruction": {"formatted": mnemonic + " " + ", ".join(args), "parts": parts}}
    if line:
        e["instruction"]["line_number"] = line
    if diff:
        e["diff_kind"] = diff
    return e


class SymptomTest(unittest.TestCase):
    def test_signs_from_aligned_rows(self):
        target = [ins("add", "r3", "r4", "r5", diff="DIFF_ARG_MISMATCH"), ins("cmplw", "r0", "r3", diff="DIFF_ARG_MISMATCH"),
                  ins("beq", "x", diff="DIFF_REPLACE"), ins("cmplwi", "r3", "3", diff="DIFF_REPLACE"),
                  ins("li", "r30", "0", diff="DIFF_REPLACE"), ins("lwz", "r3", "r4", diff="DIFF_ARG_MISMATCH")]
        ours = [ins("add", "r3", "r5", "r4", diff="DIFF_ARG_MISMATCH", line=10),
                ins("cmplw", "r3", "r0", diff="DIFF_ARG_MISMATCH", line=11), ins("bne", "x", diff="DIFF_REPLACE", line=11),
                ins("cmpwi", "r3", "3", diff="DIFF_REPLACE", line=12), ins("mr", "r30", "r29", diff="DIFF_REPLACE", line=13),
                ins("lwz", "r5", "r4", diff="DIFF_ARG_MISMATCH", line=14)]
        s = symptoms(target, ours)
        self.assertEqual(dict(s.signs), {"swap": 1, "compare": 1, "branch": 1, "signed": 1, "zero": 1, "regs": 1})
        self.assertEqual(sorted(s.lines), [10, 11, 12, 13, 14])

    def test_order_count_and_relocations(self):
        target = [ins("stw", "r0", "8", "r1", diff="DIFF_REPLACE"), ins("lfs", "f1", "lbl_1", diff="DIFF_ARG_MISMATCH"),
                  ins("b", "x", diff="DIFF_INSERT"), {"diff_kind": "DIFF_DELETE"}]
        ours = [ins("lwz", "r0", "8", "r1", diff="DIFF_REPLACE", line=3), ins("lfs", "f1", "@72", diff="DIFF_ARG_MISMATCH"),
                {"diff_kind": "DIFF_INSERT"}, ins("stw", "r0", "8", "r1", diff="DIFF_DELETE")]
        s = symptoms(target, ours)
        self.assertEqual(s.signs["reloc"], 1)
        self.assertEqual(s.signs["order"], 1)          # stw on both sides, at other places
        self.assertGreaterEqual(s.signs["count"], 1)

    def test_rules_are_ranked_by_the_signs_they_answer(self):
        sym = rewrites.Symptoms()
        sym.signs.update({"branch": 3})
        top = [r.name for r in ranked(sym)[:3]]
        self.assertIn("one-case-switch", top)
        self.assertEqual(len(ranked(sym)), len(RULES))


class TableTest(unittest.TestCase):
    def test_every_row_names_its_symptom_idiom_and_known_signs(self):
        for r in RULES:
            self.assertTrue(r.symptom and r.idiom, r.name)
            self.assertTrue(set(r.signs) <= set(rewrites.SIGNS), r.name)
        self.assertEqual(len({r.name for r in RULES}), len(RULES))

    def test_one_shape_at_several_sites_is_also_one_candidate(self):
        body = "    p->a = (float)(a * 1) * x;\n    p->b = (float)(a * 2) * x;\n"
        got = [lab for _, lab, _ in candidates(site(body), [BY_NAME["operand-swap"]])]
        self.assertTrue(any("(and 1 more like it)" in lab for lab in got))
        none = [lab for _, lab, _ in candidates(site(body), [BY_NAME["operand-swap"]], groups=False)]
        self.assertFalse(any("more like it" in lab for lab in none))
        both = next(new for _, lab, new in candidates(site(body), [BY_NAME["operand-swap"]])
                    if lab.startswith("named") and "more like it" in lab and ": t0 * x" in lab)
        self.assertIn("t0 = (float)(a * 1);", both)
        self.assertIn("t1 = (float)(a * 2);", both)

    def test_each_candidate_numbers_its_new_locals_from_zero(self):
        body = "    p->a = a * 2 + p->c;\n    p->b = b * 3 + p->d;\n"
        second = [new for _, lab, new in candidates(site(body), [BY_NAME["operand-swap"]]) if "named b * 3" in lab]
        self.assertTrue(second and all("t0 = b * 3" in new for new in second))

    def test_focus_limits_the_sites(self):
        s = site("    a = a + b;\n    b = x * 2.0f + a;\n", focus={2})
        labels = [lab for _, lab, _ in candidates(s, [BY_NAME["operand-swap"]])]
        self.assertTrue(labels and all("a + b" not in lab for lab in labels))


class RuleTest(unittest.TestCase):
    def test_operand_swap_plain_and_named(self):
        got = outs("operand-swap", "    a = b + a;\n    x = x * 2.0f + p->m;\n")
        self.assertIn("    a = a + b;\n    x = x * 2.0f + p->m;", got)
        self.assertIn("    a = b + a;\n    __typeof__(x * 2.0f) t0 = x * 2.0f;\n    x = p->m + t0;", got)
        self.assertIn("    a = b + a;\n    __typeof__(x * 2.0f) t0 = x * 2.0f;\n    x = t0 + p->m;", got)

    def test_operand_swap_in_c_declares_at_the_top(self):
        got = outs("operand-swap", "    int k = 0;\n    k = a * b + p->m;\n", cplusplus=False)
        self.assertIn("    int k = 0;\n    __typeof__(a * b) t0;\n    t0 = a * b;\n    k = p->m + t0;", got)

    def test_operand_swap_never_hoists_a_guarded_operand(self):
        got = outs("operand-swap", "    if (a && b * 2 + a) a = 1;\n")
        self.assertFalse(any("t0" in g for g in got))

    def test_compare_flip_and_truth_tests(self):
        got = outs("compare-flip", "    if (a < b) a = 1;\n    if (p) b = 2;\n    if (x != 0.0f) b = 3;\n")
        self.assertIn("    if (b > a) a = 1;\n    if (p) b = 2;\n    if (x != 0.0f) b = 3;", got)
        self.assertIn("    if (a < b) a = 1;\n    if (p != 0) b = 2;\n    if (x != 0.0f) b = 3;", got)
        self.assertIn("    if (a < b) a = 1;\n    if (p) b = 2;\n    if (x) b = 3;", got)
        self.assertIn("    if (a < b) a = 1;\n    if (p) b = 2;\n    if (0.0f != x) b = 3;", got)

    def test_reassociate(self):
        got = outs("reassociate", "    a = a + (b + 4);\n")
        self.assertEqual(got, ["    a = a + b + 4;"])
        self.assertIn("    a = a + (b + 4);", outs("reassociate", "    a = a + b + 4;\n"))

    def test_half(self):
        self.assertEqual(outs("half", "    x = x * 0.5f;\n"), ["    x = x / 2.0f;"])
        self.assertEqual(outs("half", "    x = (x + 1.0f) / 2.0f;\n"), ["    x = (x + 1.0f) * 0.5f;", "    x = 0.5f * (x + 1.0f);"])
        self.assertEqual(outs("half", "    x *= 0.5f;\n"), ["    x /= 2.0f;"])

    def test_ternary_and_if_both_ways(self):
        got = outs("ternary-if", "    a = b < 0 ? b : 0;\n")
        self.assertEqual(got, ["    if (b < 0) {\n        a = b;\n    } else {\n        a = 0;\n    }"])
        got = outs("ternary-if", "    if (b < 0) {\n        a = b;\n    } else {\n        a = 0;\n    }\n")
        self.assertEqual(got, ["    a = b < 0 ? b : 0;"])
        got = outs("ternary-if", "    if (a) return b;\n    return 0;\n")
        self.assertEqual(got, ["    return a ? b : 0;"])
        got = outs("ternary-if", "    int k = a ? b : 2;\n")
        self.assertEqual(got, ["    int k;\n    if (a) {\n        k = b;\n    } else {\n        k = 2;\n    }"])

    def test_one_case_switch_both_ways(self):
        got = outs("one-case-switch", "    if (a == 3) {\n        b = 1;\n    }\n")
        self.assertEqual(got, ["    switch (a) {\n    case 3:\n        b = 1;\n        break;\n    }"])
        got = outs("one-case-switch", "    switch (a) {\n    case 3:\n        b = 1;\n        break;\n    }\n")
        self.assertEqual(got, ["    if (a == 3) {\n        b = 1;\n    }"])

    def test_one_case_switch_keeps_a_loop_break_out(self):
        self.assertEqual(outs("one-case-switch", "    for (;;) {\n        if (a == 3) break;\n    }\n"), [])

    def test_or_return_both_ways(self):
        got = outs("or-return", "    if (!a) return;\n    if (!b) return;\n")
        self.assertEqual(got, ["    if (!a || !b) return;"])
        got = outs("or-return", "    if (!a || !b) return;\n")
        self.assertEqual(got, ["    if (!a) return;\n    if (!b) return;"])

    def test_chained_assignment_both_ways(self):
        got = outs("chained-assign", "    a = 0;\n    b = 0;\n    p->m = 0;\n")
        self.assertEqual(got, ["    a = b = p->m = 0;", "    p->m = b = a = 0;"])
        got = outs("chained-assign", "    a = b = 0;\n")
        self.assertEqual(got, ["    a = 0;\n    b = 0;", "    b = 0;\n    a = 0;"])

    def test_op_assign_both_ways(self):
        self.assertEqual(outs("op-assign", "    x = x * 2.0f;\n"), ["    x *= 2.0f;"])
        self.assertEqual(outs("op-assign", "    x = 2.0f * x;\n"), ["    x *= 2.0f;"])
        self.assertEqual(outs("op-assign", "    x *= a + b;\n"), ["    x = x * (a + b);", "    x = (a + b) * x;"])

    def test_unsigned_index(self):
        got = outs("unsigned-index", "    for (int i = 0; i < a; i++) {\n        p->v[i] = 0;\n    }\n")
        self.assertIn("    for (int i = 0; i < a; i++) {\n        p->v[(unsigned int)i] = 0;\n    }", got)
        self.assertIn("    for (unsigned int i = 0; i < a; i++) {\n        p->v[i] = 0;\n    }", got)

    def test_reference_element_both_ways(self):
        got = outs("ref-element", "    p->v[a].x = 1.0f;\n    p->v[a].y = x;\n    a++;\n    p->v[a].x = 0;\n")
        self.assertEqual(got, ["    __typeof__(p->v[a])& e0 = p->v[a];\n    e0.x = 1.0f;\n    e0.y = x;\n    a++;\n"
                               "    p->v[a].x = 0;"])
        got = outs("ref-element", "    T& e = p->v[a];\n    e.x = 1.0f;\n    e.y = x;\n")
        self.assertEqual(got, ["    p->v[a].x = 1.0f;\n    p->v[a].y = x;"])

    def test_flag_form_both_ways(self):
        got = outs("flag-form", "    bool k = false;\n    if (a || b) k = true;\n    g(k);\n")
        self.assertEqual(got, ["    bool k = a || b;\n    g(k);"])
        got = outs("flag-form", "    bool k = a < b;\n    g(k);\n")
        self.assertEqual(got, ["    bool k = false;\n    if (a < b) {\n        k = true;\n    }\n    g(k);"])

    def test_assignment_in_condition_both_ways(self):
        got = outs("assign-in-condition", "    S* q = p->next;\n    if (q != 0) g(q);\n")
        self.assertEqual(got, ["    S* q;\n    if ((q = p->next) != 0) g(q);"])
        got = outs("assign-in-condition", "    S* q;\n    if ((q = p->next) != 0) g(q);\n")
        self.assertEqual(got, ["    S* q;\n    q = p->next;\n    if (q != 0) g(q);"])

    def test_assignment_never_moves_into_a_loop_condition(self):
        self.assertEqual(outs("assign-in-condition", "    a = g(a);\n    while (a) a--;\n"), [])

    def test_declaration_order(self):
        got = outs("decl-order", "    int k = a;\n    float y;\n    y = x;\n    g(k, y);\n")
        self.assertEqual(got, ["    float y;\n    int k = a;\n    y = x;\n    g(k, y);"])
        got = outs("decl-order", "    int k = a;\n    {\n        float y = x;\n        g(k, y);\n    }\n")
        self.assertEqual(got, ["    float y;\n    int k = a;\n    {\n        y = x;\n        g(k, y);\n    }"])

    def test_declaration_order_keeps_const_and_references_in_place(self):
        self.assertEqual(outs("decl-order", "    int k = a;\n    const int y = b;\n    T& r = p->v[0];\n"), [])

    def test_statement_order_swaps_independent_neighbours(self):
        got = outs("statement-order", "    p->a = a;\n    b = p->c;\n    a = b + 1;\n")
        self.assertEqual(got, ["    b = p->c;\n    p->a = a;\n    a = b + 1;"])       # b then a depend on each other
        self.assertEqual(outs("statement-order", "    int k = a;\n    g(k);\n"), [])

    def test_synth_artifacts_never_made(self):                 # the 2026-11-02 spot check
        got = outs("operand-swap", "    a = b - a + 12;\n    x = (x - 3) / b + 12;\n")
        self.assertIn("    a = 12 + (b - a);\n    x = (x - 3) / b + 12;", got)   # same precedence: grouped
        self.assertIn("    a = b - a + 12;\n    x = 12 + (x - 3) / b;", got)     # tighter: no parentheses needed
        self.assertEqual(outs("operand-swap", "    a = (a - 3) / b / 12 * 12;\n"), [])     # integer rounding
        self.assertEqual(outs("operand-swap", '    g(p->s + ":" + p->t);\n'), [])         # string concatenation
        self.assertEqual(outs("ternary-if", "    a = b < 0 ? a : 0;\n"), [])            # `a = a;`
        self.assertEqual(outs("ternary-if", "    if (b) {\n        a = 1;\n    } else {\n        a = 0;\n    }\n"), [])
        self.assertEqual(outs("ternary-if", "    if (b) return true;\n    return false;\n"), [])
        self.assertEqual(outs("assign-in-condition", "    a = g(b);\n    if (h(a, b)) b = 0;\n"), [])
        calls = "    g(1);\n    g(2);\n    g(3);\n"
        self.assertEqual(outs("statement-order", calls), [])                      # a run of identical calls
        self.assertEqual(len(outs("statement-order", "    g(1);\n    g(2);\n")), 1)

    def test_split_cast_compare_cast_if_invert(self):
        self.assertEqual(outs("split-init", "    int k = a;\n"), ["    int k;\n    k = a;"])
        self.assertEqual(outs("cast-call", "    S* q = Get(a);\n"), ["    S* q = (S*)Get(a);"])
        self.assertEqual(outs("cast-call", "    S* q = (S*)Get(a);\n"), ["    S* q = Get(a);"])
        self.assertEqual(outs("compare-cast", "    if (p->c == 3) a = 1;\n"),
                         ["    if ((int)p->c == 3) a = 1;", "    if ((unsigned int)p->c == 3) a = 1;"])
        self.assertEqual(outs("if-invert", "    if (a < b) {\n        a = 1;\n    } else {\n        b = 1;\n    }\n"),
                         ["    if (a >= b) {\n        b = 1;\n    } else {\n        a = 1;\n    }"])


if __name__ == "__main__":
    unittest.main()
