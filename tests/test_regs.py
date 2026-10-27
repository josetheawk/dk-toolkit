import struct
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.regs import (DwarfFunction, Row, Var, advice, debug_refs, find_function, parse_dwarf1,
                                             register_map, reorder, rows, tie_orders, wanted_order)

REPO = Path(__file__).resolve().parents[1]


def die(tag: int, attrs: bytes) -> bytes:
    return struct.pack(">IH", 6 + len(attrs), tag) + attrs


def name(s: str) -> bytes:
    return struct.pack(">H", 0x0038) + s.encode() + b"\0"


def mangled(s: str) -> bytes:
    return struct.pack(">H", 0x2008) + s.encode() + b"\0"


def sibling(v: int) -> bytes:
    return struct.pack(">HI", 0x0012, v)


def reg(n: int) -> bytes:
    return struct.pack(">HH", 0x0023, 5) + struct.pack(">BI", 1, n)


def build_debug() -> tuple[bytes, dict[int, int]]:
    """fn(param self in r29) { local a in r31; { local b in f30 } } then fn2(); siblings relocated."""
    parts, refs = [], {}

    def add(blob: bytes, sib_at: int | None = None) -> int:
        off = sum(len(p) for p in parts)
        parts.append(blob)
        return off

    fn = add(die(0x06, sibling(0) + name("fn") + mangled("fn__FP1S")))
    add(die(0x05, sibling(0) + name("self") + reg(29)))
    add(die(0x0C, sibling(0) + name("a") + reg(31)))
    blk = add(die(0x0B, sibling(0)))
    add(die(0x0C, sibling(0) + name("b") + reg(62)))
    add(struct.pack(">I", 4))                      # end of the block's children
    blk_end = sum(len(p) for p in parts)
    add(struct.pack(">I", 4))                      # end of fn's children
    fn_end = sum(len(p) for p in parts)
    add(die(0x06, sibling(0) + name("fn2")))
    refs[fn + 6 + 2] = fn_end                      # sibling value sits after tag + attribute name
    refs[blk + 6 + 2] = blk_end
    return b"".join(parts), refs


class DwarfTest(unittest.TestCase):
    def test_variables_are_scoped_by_sibling_references(self):
        data, refs = build_debug()
        fns = parse_dwarf1(data, refs)
        self.assertEqual([f.name for f in fns], ["fn", "fn2"])
        f = find_function(fns, "fn__FP1S")
        self.assertEqual([(v.kind, v.name, v.reg, v.depth) for v in f.vars],
                         [("param", "self", "r29", 0), ("local", "a", "r31", 0), ("local", "b", "f30", 1)])
        self.assertEqual(fns[1].vars, [])
        self.assertIs(find_function(fns, "fn"), f)        # by plain name when unique


def ins(mnemonic: str, *regs: str) -> dict:
    parts = [{"opcode": {"mnemonic": mnemonic}}] + [{"arg": {"opaque": r}} for r in regs]
    return {"instruction": {"parts": parts}}


class MappingTest(unittest.TestCase):
    def test_register_map_pairs_aligned_operands(self):
        target = [ins("mr", "r29", "r3"), ins("lwz", "r0", "r29"), {}, ins("addi", "r31", "r31")]
        ours = [ins("mr", "r26", "r3"), ins("lwz", "r0", "r26"), ins("nop"), ins("add", "r30", "r30")]
        m = register_map(target, ours)
        self.assertEqual(m["r26"], Counter({"r29": 2}))
        self.assertNotIn("r30", m)                       # different mnemonics are not paired
        self.assertEqual(m["r3"], Counter({"r3": 1}))

    def test_rows_and_declaration_advice(self):
        fn = DwarfFunction("f", "f", [Var("param", "list", "r27"), Var("local", "low", "r31"), Var("local", "high", "r30"),
                                      Var("local", "last", "r29"), Var("local", "mid", "r28")])
        pairs = {"r31": Counter({"r31": 3}), "r30": Counter({"r30": 2}), "r29": Counter({"r28": 4}),
                 "r28": Counter({"r29": 5}), "r27": Counter({"r27": 1}), "r3": Counter({"r4": 1})}
        table = rows(fn, pairs, "r")
        self.assertEqual([r.ours for r in table], ["r31", "r30", "r29", "r28", "r27"])   # volatile r3 left out
        current, want, _ = wanted_order(fn, table)
        self.assertEqual((current, want), (["last", "mid"], ["mid", "last"]))
        self.assertTrue(advice(fn, table)[0].startswith("declare these locals in this order (first gets the highest register): mid, last"))

    def test_no_advice_when_registers_agree(self):
        fn = DwarfFunction("f", "f", [Var("local", "a", "r31")])
        self.assertEqual(advice(fn, [Row("r31", "r31", ["local a"], [])]), [])

    def test_tie_orders_permute_shared_target_registers(self):
        out = tie_orders(["a", "b", "c"], {"a": 31, "b": 30, "c": 30})
        self.assertEqual(out, [["a", "b", "c"], ["a", "c", "b"]])


SRC = """int f(int n) {
    int low = 0;
    int high = n;
    int last = -1;
    for (int i = 0; i < n; i++) {
        int mid = (low + high) / 2;
        last = mid;
    }
    return last;
}
"""


class ReorderTest(unittest.TestCase):
    def test_minimal_move_declares_one_local_earlier(self):
        out = reorder(SRC, "f", ["mid", "last"], minimal=True)
        lines = out.splitlines()
        self.assertEqual(lines[3], "    int mid;")
        self.assertEqual(lines[4], "    int last = -1;")
        self.assertIn("        mid = (low + high) / 2;", lines)
        self.assertIsNone(reorder(SRC, "f", ["low", "high"], minimal=True))   # already in that order

    def test_hoist_declares_all_at_the_top_and_handles_for(self):
        out = reorder(SRC, "f", ["i", "last", "low"], minimal=False)
        lines = out.splitlines()
        self.assertEqual(lines[1:4], ["    int i;", "    int last;", "    int low;"])
        self.assertIn("    low = 0;", lines)
        self.assertIn("    for (i = 0; i < n; i++) {", lines)

    def test_duplicate_or_complex_declarations_are_refused(self):
        dup = SRC.replace("        last = mid;", "        int low = mid;")
        self.assertIsNone(reorder(dup, "f", ["low", "last"], minimal=False))


class CompilerRuleTest(unittest.TestCase):
    """The allocation rule itself, against the real compiler (skipped without the project's tools)."""

    def compile(self, code: str) -> Elf:
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            self.skipTest("compiler not installed (./dk setup)")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.cpp"
            src.write_text(code)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware",
                   "-lang=c++", "-sym", "on", "-c", str(src), "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return Elf.parse((Path(tmp) / "t.o").read_bytes())

    def test_declaration_order_then_parameters_last_param_first(self):
        elf = self.compile("extern int h(int); extern void g(int,int,int,int,int);\n"
                           "void t(int x, int y) { int c, b, a; a = h(1); b = h(2); c = h(3); h(4); g(a, b, c, x, y); }\n")
        f = find_function(parse_dwarf1(elf.section(".debug").data, debug_refs(elf)), "t__Fii")
        regs = {v.name: v.reg for v in f.vars}
        self.assertEqual(regs, {"c": "r31", "b": "r30", "a": "r29", "y": "r28", "x": "r27"})

    def test_a_free_value_reuses_the_lowest_register_already_handed_out(self):
        # v2 overlaps v0 only; v3 overlaps nothing, so it takes r30 (lowest in use), not r31.
        elf = self.compile("extern int h(int); extern void g(int,int);\n"
                           "void t() { int v0, v2, v3; v0 = h(1); v2 = h(2); h(3); g(v0, 1); g(v2, 2);"
                           " v3 = h(4); h(5); g(v3, 3); }\n")
        f = find_function(parse_dwarf1(elf.section(".debug").data, debug_refs(elf)), "t__Fv")
        self.assertEqual({v.name: v.reg for v in f.vars}, {"v0": "r31", "v2": "r30", "v3": "r30"})


if __name__ == "__main__":
    unittest.main()
