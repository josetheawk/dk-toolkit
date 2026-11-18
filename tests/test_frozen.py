"""The frozen union (core/frozen.py + gate.frozen_keys) and the commit guard. Synthetic fixtures only."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.dk.core import frozen
from tools.dk.platforms.gc_mwcc import gate


def rows(*names, unit="u/x"):
    return {"functions": [{"unit": unit, "function": n} for n in names]}


class Union(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("DK_LLM_DATA")
        os.environ["DK_LLM_DATA"] = self.tmp.name
        self.d = Path(self.tmp.name)
        (self.d / "eval").mkdir()
        (self.d / "gate-lora2").mkdir()
        self.cfg = SimpleNamespace(root=self.d, platform_options={})

    def tearDown(self):
        if self.old is None:
            os.environ.pop("DK_LLM_DATA", None)
        else:
            os.environ["DK_LLM_DATA"] = self.old
        self.tmp.cleanup()

    def write(self, rel, doc):
        (self.d / rel).write_text(json.dumps(doc))

    def test_every_set_counts_and_released_leave(self):
        self.write("gate-lora2/gate.json", rows("g1"))
        self.write("near-sample-60.json", rows("n1"))
        self.write("bench.json", rows("b1", "g1"))
        self.write("heldout.json", rows("h1"))
        self.write("heldout-split.json", {"dev": ["d1"], "final": ["f1"], "excluded": ["x1"]})
        self.write("repair-eval.json", rows("r1"))
        self.write("review-eval.json", {"units": ["u/x"], "functions": [{"function": "v1"}]})
        self.write("eval/h-giant-40.json", rows("e1"))
        self.write("eval/released.json", {"b1": {"date": "2026-11-18", "reason": "test"},
                                         "__ct__3UnkFv": {"date": "d", "reason": "r"}, "_note": "text"})
        self.write("near-sample-fresh-30.json", rows("__ct__3UnkFv"))
        u = gate.frozen_keys(self.cfg)
        self.assertEqual(dict(u), {"g1": "gate-lora2", "n1": "near-sample-60", "e1": "eval:h-giant-40", "h1": "heldout",
                                   "d1": "heldout-split", "f1": "heldout-split", "r1": "repair-eval", "v1": "review-eval"})
        self.assertNotIn("b1", u)                       # released
        self.assertNotIn("x1", u)                       # only the named lists of heldout-split
        self.assertIn(("any/unit", "g1"), u)
        self.assertEqual(set(gate.released_functions(self.cfg)), {"b1", "__ct__3UnkFv"})

    def test_dk_toml_sets_override(self):
        self.write("bench.json", rows("b1"))
        self.write("mine.json", rows("m1"))
        self.cfg.platform_options = {"frozen": {"sets": {"mine": ["mine.json"]}, "released": "none.json"}}
        self.assertEqual(dict(gate.frozen_keys(self.cfg)), {"m1": "mine"})


SRC = """struct Unk { Unk(); int Get(int a); int Get(); };
Unk::Unk() : m(0) {
    m = 1;
}
int Unk::Get(int a) {
    return a + 1;
}
int Unk::Get() {
    return 2;
}
extern "C" int fn_80001000(int x) {
    return x;
}
"""
WANT = {"__ct__3UnkFv": [("Unk::Unk", 0)], "Get__3UnkFi": [("Unk::Get", 1)], "fn_80001000": [("fn_80001000", None)]}


class Guard(unittest.TestCase):
    def changed(self, before, after):
        return frozen.changed_frozen(before, after, WANT)

    def test_unchanged_and_moved_are_fine(self):
        self.assertEqual(self.changed({"a.cpp": SRC.encode()}, {"a.cpp": SRC.encode()}), [])
        body = "extern \"C\" int fn_80001000(int x) {\n    return x;\n}\n"
        moved_from = SRC.replace(body, "")
        self.assertEqual(self.changed({"a.cpp": SRC.encode(), "b.cpp": b""},
                                      {"a.cpp": moved_from.encode(), "b.cpp": body.encode()}), [])

    def test_edit_names_the_function_and_overloads_apart(self):
        after = SRC.replace("return a + 1;", "return a + 2;")
        self.assertEqual(self.changed({"a.cpp": SRC.encode()}, {"a.cpp": after.encode()}), ["Get__3UnkFi"])
        after = SRC.replace("return 2;", "return 3;")              # the other overload is not frozen
        self.assertEqual(self.changed({"a.cpp": SRC.encode()}, {"a.cpp": after.encode()}), [])
        after = SRC.replace("m = 1;", "m = 2;")
        self.assertEqual(self.changed({"a.cpp": SRC.encode()}, {"a.cpp": after.encode()}), ["__ct__3UnkFv"])

    def test_new_definition_and_deletion_count(self):
        new = "int fn_80001000(int x) {\n    return x;\n}\n"
        self.assertEqual(self.changed({"n.cpp": None}, {"n.cpp": new.encode()}), ["fn_80001000"])
        self.assertEqual(self.changed({"n.cpp": new.encode()}, {"n.cpp": None}), ["fn_80001000"])
        self.assertEqual(self.changed({"n.txt": None}, {"n.txt": new.encode()}), [])   # not a source file

    def test_guard_message_says_how_to_release(self):
        mod = SimpleNamespace(frozen_definitions=lambda cfg: (WANT, "llm-data/eval/released.json"))
        import tools.dk.platforms as platforms
        orig = platforms.load
        platforms.load = lambda name: mod
        try:
            out = frozen.guard(SimpleNamespace(platform="x"), {"a.cpp": SRC.replace("return x;", "return 0;").encode()},
                               lambda rel: SRC.encode())
        finally:
            platforms.load = orig
        self.assertEqual(len(out), 1)
        self.assertIn("fn_80001000", out[0])
        self.assertIn("released.json", out[0])

    def test_definition_forms_from_mangled_names(self):
        tmp = tempfile.TemporaryDirectory()
        old = os.environ.get("DK_LLM_DATA")
        os.environ["DK_LLM_DATA"] = tmp.name
        try:
            d = Path(tmp.name)
            (d / "bench.json").write_text(json.dumps(rows("__ct__3UnkFv", "Get__3UnkFi", "fn_80001000")))
            (d / "symbols.txt").write_text("__ct__3UnkFv = .text:0x80002000; // type:function size:0x4\n")
            cfg = SimpleNamespace(root=d, platform_options={"symbols": "symbols.txt"})
            want, _ = gate.frozen_definitions(cfg)
            self.assertEqual(want["__ct__3UnkFv"], [("Unk::Unk", 0), ("fn_80002000", None)])
            self.assertEqual(want["Get__3UnkFi"], [("Unk::Get", 1)])
            self.assertEqual(want["fn_80001000"], [("fn_80001000", None)])
        finally:
            if old is None:
                os.environ.pop("DK_LLM_DATA", None)
            else:
                os.environ["DK_LLM_DATA"] = old
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
