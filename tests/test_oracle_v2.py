"""Oracle contract v2 (colouring order, order solver, write mode) and the close search it directs.

Synthetic: a fake runner script answers the `--solve` contract. Golden (skipped without the project's compiler
and a v2 runner in DK_ORACLE): a toy function compiled under the real runner."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import closecmd, oracle
from tools.dk.platforms.gc_mwcc.blockorders import block_orders
from tools.dk.platforms.gc_mwcc.oracle import Value
from tools.dk.platforms.gc_mwcc.rewrites import Site

REPO = Path(__file__).resolve().parents[1]

FAKE = textwrap.dedent('''\
    import json, sys
    a = sys.argv[1:]
    if a[0] != "--solve":
        sys.exit(2)
    req = json.load(open(a[1]))
    if req.get("probe"):
        out = {"schema": "reach.v1", "modes": ["extra", "force", "solve"]}
    else:
        t = req["targets"]["gpr"]
        out = {"schema": "reach.v1", "reachable": True, "seen": t, "classes": {"gpr": {
            "reachable": True, "mode": "strict", "force": {"order": [41, 40], "call": 1},
            "rules": [{"before": {"index": 41, "kind": "temp", "line": 3}, "after": {"index": 40, "kind": "temp", "line": 3},
                       "flip": True, "cell": "temp-temp"}],
            "decl_order": {"changed": True, "declare": [["b", 41, 3], ["a", 40, 2]], "force": {"order": [41, 40]}}}}}
    json.dump(out, open(a[2], "w"))
    ''')


def rec(order, operands):
    """A function record: values 40/41 (gpr) with operand positions and a colouring order."""
    return {"name": "f", "vregs": [{"index": i, "cls": "gpr", "kind": "temp", "line": 3 + k, "operands": ops,
                                    "coalesced_into": None} for k, (i, ops) in enumerate(operands.items())],
            "graph": {"gpr": {"order": order}}}


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "fake.py").write_text(FAKE)
        self.cmd = [sys.executable, str(self.tmp / "fake.py")]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wrap_passes_extra_and_force(self):
        argv = oracle.wrap(["run"], Path("/t/v.json"), "fn_1__Fv", extra=True, force=Path("/t/f.json"))(["cc"])
        self.assertEqual(argv, ["run", "/t/v.json", "--funcs", "fn_1", "--extra", "--force", "/t/f.json", "--", "cc"])

    def test_probe_and_solve_round_trip(self):
        self.assertTrue(oracle.supports_solve(self.cmd, self.tmp / "w"))
        self.assertFalse(oracle.supports_solve(["false"], self.tmp / "w2"))
        vals = [Value(40, "gpr", "temp", "", 3, "r5", "r4"), Value(41, "gpr", "temp", "", 3, "r4", "r5"),
                Value(42, "fpr", "temp", "", 4, "f1", None), Value(43, "gpr", "temp", "", 4, "r6", "f2")]
        self.assertEqual(oracle.target_colours(vals), {"gpr": {"40": 4, "41": 5}})
        self.assertIn("error", oracle.solve(self.cmd, {"vregs": []}, vals, self.tmp / "w"))     # no --extra graph
        an = oracle.solve(self.cmd, rec([40, 41], {40: [[0, 0]], 41: [[1, 0]]}), vals, self.tmp / "w")
        self.assertEqual(an["seen"], {"40": 4, "41": 5})
        self.assertEqual(oracle.force_spec("f__Fv", an), {"f": {"gpr": {"order": [41, 40], "call": 1}}})
        self.assertEqual(oracle.force_spec("f__Fv", an, "decl"), {"f": {"gpr": {"order": [41, 40]}}})
        self.assertEqual(oracle.colour_spec("f__Fv", vals), {"f": {"gpr": {"colors": {"40": 4, "41": 5}}}})
        rules = oracle.flat_rules(an)
        self.assertEqual([(r["a"], r["b"], r["flip"], r["cell"]) for r in rules], [(41, 40, True, "temp-temp")])
        self.assertEqual(oracle.wanted_orders(an), [[("b", 3), ("a", 2)]])

    def test_a_runner_without_solve_says_so(self):
        doc = oracle._call_solve(["true"], {"schema": oracle.REACH, "probe": True}, self.tmp / "w")
        self.assertIn("no `--solve` mode", doc["error"])

    def test_rule_status_follows_values_by_operand_position(self):
        rules = [{"cls": "gpr", "a": 41, "b": 40}]
        base = rec([40, 41], {40: [[0, 0]], 41: [[1, 0]]})
        # renumbered values (a new compile), now coloured 41's value first
        new = rec([7, 6], {6: [[0, 0]], 7: [[1, 0]]})
        self.assertEqual(oracle.rule_status(rules, base, base), [False])
        self.assertEqual(oracle.rule_status(rules, base, new), [True])
        self.assertEqual(oracle.map_values(base, new), {("gpr", 40): 6, ("gpr", 41): 7})
        gone = rec([6], {6: [[0, 0]]})
        self.assertEqual(oracle.rule_status(rules, base, gone), [None])


class RuleDirectedTest(unittest.TestCase):
    def test_changed_lines_and_order(self):
        old = "a\nb\nc\nd\n"
        self.assertEqual(closecmd.changed_lines(old, "a\nb\nX\nd\n"), (3, 3))
        cands = [("r", "far", "Z\nb\nc\nd\n"), ("r", "near", "a\nb\nX\nd\n"), ("r", "x declared", "a\nb\nc\nd\ne\n")]
        got = closecmd.rule_directed(old, cands, {3}, {"x"})
        self.assertEqual([c[1] for c in got], ["near", "x declared", "far"])

    def test_eval_key_puts_rules_after_shape(self):
        sym = mock.Mock()
        a = closecmd.Eval("", 99.0, 2, sym, 1, right=5, wrong=1, shape_rows=0, sat=2)
        b = closecmd.Eval("", 99.5, 2, sym, 1, right=5, wrong=0, shape_rows=0, sat=1)
        c = closecmd.Eval("", 99.9, 1, sym, 1, right=6, wrong=0, shape_rows=1, sat=3)
        self.assertEqual(sorted([c, b, a], key=lambda e: e.key), [a, b, c])


def site(src: str) -> Site:
    span = cbody.functions(src)[0]
    return Site(src, cbody.parse_body(src, span.open, cbody.type_names(src)), None, True)


BLOCKS = """int f(int* p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        int v = p[i];
        int w = v + 1;
        s += w;
    }
    for (int i = 0; i < n; i++) {
        int k = p[i];
        s -= k;
    }
    return s;
}
"""


class BlockOrdersTest(unittest.TestCase):
    def test_repeated_names_map_to_their_own_block(self):
        st = site(BLOCKS)
        got = block_orders(st, [("w", 5), ("v", 4)])
        labels = [g[1] for g in got]
        self.assertIn("block@3: declare w, v at the top (split)", labels)
        top = next(g[2] for g in got if g[1] == "block@3: declare w, v at the top (split)")
        self.assertIn("int w;\n        int v;\n        v = p[i];\n        w = v + 1;", top)
        # `i` of the second loop (line 8) is not the first loop's
        got2 = block_orders(st, [("i", 8), ("s", 2)])
        self.assertTrue(any("declare i, s" in g[1] for g in got2))
        moved = next(g[2] for g in got2 if "declare i, s" in g[1])
        self.assertIn("for (i = 0; i < n; i++) {\n        int k", moved)       # the second loop's i, not the first
        self.assertIn("for (int i = 0; i < n; i++) {\n        int v", moved)

    def test_initialisers_kept_where_legal(self):
        src = "int g(int a) {\n    int x = 1;\n    int y = 2;\n    return x + y + a;\n}\n"
        got = block_orders(site(src), [("y", 3), ("x", 2)])
        kept = next(g[2] for g in got if "initialisers kept" in g[1])
        self.assertIn("int y = 2;\n    int x = 1;", kept)
        self.assertEqual(block_orders(site(src), [("x", 2), ("y", 3)], {"zz"}), [])    # already in order


def _v2_runner() -> list[str] | None:
    cmd = oracle.runner(None)
    if cmd is None:
        return None
    work = Path(tempfile.mkdtemp())
    try:
        return cmd if oracle.supports_solve(cmd, work) else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


FLAGS = ["-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware", "-Cpp_exceptions", "off",
         "-RTTI", "off", "-enum", "int", "-lang=c++"]
TOY = "int g(int);\nint toy(int a, int b) {\n    int x = g(a);\n    int y = g(b);\n    return g(x + y) + x * y;\n}\n"


@unittest.skipUnless((REPO / "build/compilers/GC/1.3.2/mwcceppc.exe").exists() and _v2_runner(),
                     "needs the project's compiler and an oracle runner with contract v2 in DK_ORACLE")
class GoldenTest(unittest.TestCase):
    def compile(self, work: Path, extra=True, force=None) -> tuple[dict, bytes]:
        (work / "t.cpp").write_text(TOY)
        argv = [str(REPO / "build/tools/wibo"), str(REPO / "build/compilers/GC/1.3.2/mwcceppc.exe"), *FLAGS,
                "-c", "t.cpp", "-o", "t.o"]
        spec = None
        if force is not None:
            spec = work / "force.json"
            spec.write_text(json.dumps(force))
        subprocess.run(oracle.wrap(_v2_runner(), work / "v.json", "toy__Fii", extra, spec)(argv), cwd=work,
                       capture_output=True)
        doc = oracle.load(work / "v.json")
        return oracle.find_record(doc, "toy__Fii", None), (work / "t.o").read_bytes()

    def test_own_colouring_is_already_reached_and_force_changes_registers(self):
        work = Path(tempfile.mkdtemp())
        try:
            record, obj = self.compile(work)
            self.assertIn("order", record["graph"]["gpr"])
            kept = [v for v in record["vregs"] if v["cls"] == "gpr" and v.get("coalesced_into") is None
                    and v.get("color", 0) >= 14]                      # values in nonvolatile registers
            self.assertGreaterEqual(len(kept), 2)
            vals = [Value(v["index"], "gpr", v["kind"], v.get("name", ""), v.get("line"), f"r{v['color']}",
                          f"r{v['color']}") for v in kept]
            an = oracle.solve(_v2_runner(), record, vals, work / "s")
            self.assertTrue(an["reachable"])
            self.assertEqual(an["classes"]["gpr"]["mode"], "already")
            x, y = kept[:2]
            swap = {"toy": {"gpr": {"colors": {str(x["index"]): y["color"], str(y["index"]): x["color"]}}}}
            _rec2, obj2 = self.compile(work, force=swap)
            self.assertNotEqual(obj, obj2)                      # a forced colouring changes the code
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class DegreeLeverTest(unittest.TestCase):
    """H10: the solver's degree levers are rendered under `regs --reach`."""

    def test_levers_and_deferred_values_render(self):
        from tools.dk.platforms.gc_mwcc.regs import degree_lines
        dg = {"K": 29, "model_ok": True,
              "deferred": [{"index": 35, "kind": "param", "name": "value", "line": 148, "scan": 2}],
              "levers": [{"values": [{"index": 35, "kind": "param", "name": "value", "line": 148},
                                     {"index": 36, "kind": "param", "name": "size", "line": 148}], "delta": 1}]}
        lines = degree_lines(dg)
        self.assertIn("raw degree >= 29", lines[0])
        self.assertIn("#35 param value (line 148) (scan 2)", lines[0])
        self.assertIn("one more value(s) live across #35 param value (line 148) and #36 param size", lines[1])

    def test_nothing_without_degree_data(self):
        from tools.dk.platforms.gc_mwcc.regs import degree_lines
        self.assertEqual(degree_lines(None), [])
        self.assertEqual(degree_lines({"K": 29, "deferred": [], "levers": []}), [])
        self.assertEqual(degree_lines({"K": 29, "deferred": [{"index": 1, "kind": "temp", "scan": 2}], "levers": []})[-1],
                         "    no one- or two-value degree lever")
