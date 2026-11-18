import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_project
from tools.dk.core import quality
from tools.dk.core.config import load

RULES = """
[quality]
paths = ["src"]

[[quality.debt]]
name = "fn_"
pattern = '\\bfn_[0-9A-F]{8}\\b'

[[quality.debt]]
name = "dup-decl"
pattern = '^(?:extern [^{\\n]*;|(?:struct|class) [\\w:]+;)[ \\t]*$'
duplicates = true
"""


class DuplicateDebtTest(unittest.TestCase):
    def test_only_repeats_of_a_declaration_count(self):
        source = ('struct Foo;\nextern "C" void fn_80001000(int);\nextern int x;\n'
                  'void f() { fn_80001000(1); }\n'
                  'extern "C" void fn_80001000(int);  \nstruct Foo;\nstruct Foo;\n'
                  'extern "C" void g() { }\n')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_project(root, files={"src/a.cpp": source})
            with open(root / "dk.toml", "a") as f:
                f.write(RULES)
            cfg = load(root)
            [report] = quality.scan(cfg, quality.judged_files(cfg))
            self.assertEqual(report.debt["dup-decl"], 3)             # one extern, two forward declarations
            self.assertEqual(report.debt["fn_"], 3)                  # plain rules still count every match


if __name__ == "__main__":
    unittest.main()
