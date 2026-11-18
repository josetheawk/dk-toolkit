"""Batched candidate compiles (batch.py): versions of one function as renamed copies in one unit."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.batch import Batcher, rename

REPO = Path(__file__).resolve().parents[1]
FLAGS = ["-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware", "-Cpp_exceptions", "off",
         "-RTTI", "off", "-enum", "int", "-lang=c++"]
BASE = "int g(int);\nint f(int a, int b) {\n    int x = g(a);\n    return x + b;\n}\nint h(int a) { return f(a, 1); }\n"


def have() -> bool:
    return (REPO / "build/compilers/GC/1.3.2/mwcceppc.exe").exists() and (REPO / "build/tools/wibo").exists()


def real_compiler(texts: list):
    def go(text: str, slot: str):
        texts.append(text)
        work = Path(tempfile.mkdtemp())
        try:
            (work / "t.cpp").write_text(text)
            proc = subprocess.run([str(REPO / "build/tools/wibo"), str(REPO / "build/compilers/GC/1.3.2/mwcceppc.exe"),
                                   *FLAGS, "-c", "t.cpp", "-o", "t.o"], cwd=work, capture_output=True)
            return (work / "t.o").read_bytes() if proc.returncode == 0 else None
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return go


class SyntheticTest(unittest.TestCase):
    def test_rename_only_touches_identifiers(self):
        self.assertEqual(rename('int f(int f2) { return f(f2) + "f"[0]; } // f', "f", "f_dkb1"),
                         'int f_dkb1(int f2) { return f_dkb1(f2) + "f"[0]; } // f')

    def test_only_edits_inside_the_function_are_batched(self):
        b = Batcher(lambda t, s: None, BASE, "f__Fii", False)
        self.assertTrue(b.ready)
        inside = BASE.replace("return x + b;", "return b + x;")
        self.assertIn("return b + x;", b.body_of(inside))
        self.assertIsNone(b.body_of("static int k;\n" + inside))         # an edit above the function
        self.assertIsNone(b.body_of(inside + "int z;\n"))                  # an edit below it

    def test_ineligible_functions(self):
        self.assertFalse(Batcher(lambda t, s: None, BASE, "f__Fii", False, size=4096).ready)
        self.assertFalse(Batcher(lambda t, s: None, BASE.replace("int f(", "static int f("), "f__Fii", False).ready)
        self.assertFalse(Batcher(lambda t, s: None, BASE, "__ct__1AFv", False).ready)
        two = BASE + "int f(int a) { return a; }\n"
        self.assertFalse(Batcher(lambda t, s: None, two, "f__Fii", False).ready)

    def test_a_failed_batch_answers_nothing(self):
        b = Batcher(lambda t, s: None, BASE, "f__Fii", False)
        self.assertEqual(b.hashes([BASE.replace("x + b", "b + x")]), [None])


@unittest.skipUnless(have(), "needs the project's compiler (./dk setup and a build)")
class GoldenTest(unittest.TestCase):
    def test_same_code_same_hash_and_one_compile(self):
        texts = []
        b = Batcher(real_compiler(texts), BASE, "f__Fii", False, jobs=1)
        versions = [BASE.replace("return x + b;", "return b + x;"),          # commutative: same code
                    BASE.replace("return x + b;", "return x + b; /* a comment */"),
                    BASE.replace("return x + b;", "return x - b;"),          # different code
                    "int k;\n" + BASE]                                       # not batchable
        got = b.hashes([BASE] + versions)
        self.assertEqual(len(texts), 1)                                      # one compile for all of them
        self.assertTrue(b.ready)
        self.assertIsNotNone(got[0])
        self.assertNotEqual(got[0], got[3])
        self.assertIsNone(got[4])
        self.assertEqual(got[0], got[2])                                     # a comment: same code


if __name__ == "__main__":
    unittest.main()
