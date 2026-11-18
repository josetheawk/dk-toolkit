import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import oracle
from tools.dk.platforms.gc_mwcc.rewrites import Site


def row(text: str) -> dict:
    """An objdiff instruction row: `mnemonic a, b, c`."""
    mn, _, rest = text.partition(" ")
    parts = [{"opcode": {"mnemonic": mn}}]
    for a in [x.strip() for x in rest.split(",") if x.strip()]:
        parts.append({"arg": {"opaque": a}})
    return {"instruction": {"formatted": text, "address": "0", "parts": parts}}


# Two temporaries created on one line take each other's registers (r4/r5); a named local is right.
TARGET = [row("lwz r4, r1"), row("neg r5, r6"), row("add r3, r4, r5"), row("mr r31, r3")]
OURS = [row("lwz r5, r1"), row("neg r4, r6"), row("add r3, r5, r4"), row("mr r31, r3")]
RECORD = {"name": "f", "insns": 4, "vregs": [
    {"index": 40, "cls": "gpr", "kind": "temp", "name": "", "color": 5, "coalesced_into": None, "spilled": False,
     "defs": [0], "reads": [2], "line": 3},
    {"index": 41, "cls": "gpr", "kind": "temp", "name": "", "color": 4, "coalesced_into": None, "spilled": False,
     "defs": [1], "reads": [2], "line": 3},
    {"index": 33, "cls": "gpr", "kind": "local", "name": "x", "color": 31, "coalesced_into": None, "spilled": False,
     "defs": [3], "reads": [], "line": 4},
    {"index": 34, "cls": "gpr", "kind": "temp", "name": "", "color": 3, "coalesced_into": 3, "spilled": False,
     "defs": [], "reads": [], "line": None},
]}


class OracleTest(unittest.TestCase):
    def test_compiler_names(self):
        self.assertEqual(oracle.oracle_name("fn_80001234__5ClassFv"), "fn_80001234")
        self.assertEqual(oracle.oracle_name("__ct__5ClassFv"), "__ct")
        self.assertEqual(oracle.oracle_name("memcpy"), "memcpy")

    def test_runner_absent_or_configured(self):
        class Cfg:
            platform_options = {}
        with mock.patch.dict(os.environ, {"DK_ORACLE": ""}):
            self.assertIsNone(oracle.runner(Cfg()))
        with mock.patch.dict(os.environ, {"DK_ORACLE": "/no/such/runner"}):
            self.assertIsNone(oracle.runner(Cfg()))
        with mock.patch.dict(os.environ, {"DK_ORACLE": "true"}):
            self.assertEqual(oracle.runner(Cfg()), ["true"])
        self.assertIn("DK_ORACLE", oracle.UNAVAILABLE)

    def test_wrap_runs_the_compile_under_the_runner(self):
        argv = oracle.wrap(["run"], Path("/t/v.json"), "fn_1__Fv")(["cc", "-c", "a.cpp"])
        self.assertEqual(argv, ["run", "/t/v.json", "--funcs", "fn_1", "--", "cc", "-c", "a.cpp"])

    def test_find_record_by_name_size_and_order(self):
        doc = {"schema": oracle.SCHEMA, "functions": [{"name": "__dt", "insns": 5}, {"name": "__dt", "insns": 5, "k": 2},
                                                      {"name": "__dt", "insns": 7}]}
        self.assertEqual(oracle.find_record(doc, "__dt__1AFv", 20, 1).get("k"), 2)
        self.assertEqual(oracle.find_record(doc, "__dt__1BFv", 28)["insns"], 7)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            p.write_text(json.dumps({"schema": "other"}))
            self.assertIn("error", oracle.load(p))
            self.assertIn("error", oracle.load(Path(d) / "missing.json"))

    def test_values_line_up_with_the_target_and_name_the_swap(self):
        vals = oracle.values(RECORD, TARGET, OURS)
        by = {v.index: v for v in vals}
        self.assertEqual((by[40].ours, by[40].target), ("r5", "r4"))
        self.assertEqual((by[41].ours, by[41].target), ("r4", "r5"))
        self.assertTrue(by[33].ok)
        self.assertNotIn(34, by)                          # coalesced into a real register: no value of its own
        self.assertEqual(oracle.registers_right(vals), (1, 3))
        cs = oracle.causes(vals, lambda n: "x = a[i] + -b;")
        self.assertEqual([c.kind for c in cs], ["swap"])
        self.assertIn("ours creates #41 temp", cs[0].text)
        self.assertEqual(oracle.blamed_lines(vals), {3})
        self.assertTrue(any("!! #40" in line for line in oracle.render(vals, cs)))
        self.assertIsNone(oracle.values(dict(RECORD, insns=5), TARGET, OURS))   # rows do not fit the record

    def test_dropped_branch_and_operand_order(self):
        rec = dict(RECORD, insns=7, ops=[34, 34, 0, 0, 139, 34, 139])  # one of two listed branches is not emitted
        self.assertIn(oracle.emitted_rows(rec, ["lwz", "lwz", "b", "mr", "lwz", "mr"]),
                      ([0, 1, 2, None, 3, 4, 5], [0, 1, None, 2, 3, 4, 5]))
        self.assertIsNone(oracle.emitted_rows(dict(rec, insns=9, ops=[0] * 9), ["lwz"]))
        target = TARGET[:2] + [row("add r3, r4, r5")] + TARGET[3:]
        ours = [row("lwz r4, r1"), row("neg r5, r6"), row("add r3, r5, r4"), row("mr r31, r3")]
        rec2 = json.loads(json.dumps(RECORD))
        rec2["vregs"][0]["color"], rec2["vregs"][1]["color"] = 4, 5
        rec2["lines"] = [3, 3, 7, 4]
        vals = oracle.values(rec2, target, ours)
        self.assertEqual(oracle.registers_right(vals), (3, 3))
        self.assertEqual(vals.orders, [(7, "add r3, r5, r4", "add r3, r4, r5")])
        self.assertEqual([c.kind for c in oracle.causes(vals)], ["operand-order"])
        self.assertEqual(oracle.blamed_lines(vals), {7})

    def test_directed_moves_name_a_condition_operand(self):
        src = ("int g(int);\n"
               "int f(int* p, int d) {\n"
               "    if (p[d > 0] == g(d)) {\n"
               "        return 1;\n"
               "    }\n"
               "    return 0;\n"
               "}\n")
        span = cbody.functions(src)[-1]
        site = Site(src, cbody.parse_body(src, span.open), span=span)
        v = oracle.Value(40, "gpr", "temp", "", 3, "r4", "r5", "neg r4, r6")
        w = oracle.Value(39, "gpr", "temp", "", 3, "r5", "r4", "lwz r5, 0(r3)")
        got = oracle.moves(site, [v, w])
        labels = [label for _kind, label, _new in got]
        self.assertTrue(any("named d > 0 as t0 before the if" in x for x in labels), labels)
        self.assertTrue(any(k == "operand-swap" for k, _l, _n in got))
        named = next(new for _k, label, new in got if "named d > 0" in label)
        self.assertIn("__typeof__(d > 0) t0 = d > 0;", named)


if __name__ == "__main__":
    unittest.main()
